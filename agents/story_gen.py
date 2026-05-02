from dotenv import load_dotenv
from langchain.messages import HumanMessage, ToolMessage, SystemMessage
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Pipeline
from peft import PeftModel
from langgraph.types import Command
from flask import Flask, Response, json, request, stream_with_context
from flask_cors import CORS
from flask import jsonify
import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
import io
import base64
from PIL import Image
import os
from huggingface_hub import InferenceClient

load_dotenv()

#load finetuned model
def load_synopsis_model() -> tuple[Pipeline, PeftModel]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    lora_adapter_path = "Moris258/Meta-Llama-3.1-8B-Instruct-Manga-Synopsis-v.4-LORA"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
    lora_loaded_model = PeftModel.from_pretrained(model, lora_adapter_path, device_map="auto", trust_remote_code=True)
    pipe = pipeline(
        "text-generation",
        model=lora_loaded_model,
        tokenizer=tokenizer,
        model_kwargs={
            "quantization_config": quantization_config,
        }
    )
    return pipe, lora_loaded_model

def load_image_model():
    dtype = torch.bfloat16

    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    return pipe

def generate_synopsis(input: str, pipe: Pipeline):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that creates manga synopses based on the given description. The generated synopsis should be at least 500 characters."},
        {"role": "user", "content": input},
    ]

    outputs = pipe(
        messages,
    )

    return outputs[0]["generated_text"][-1]["content"].split("synopsis: ")[1]


client = InferenceClient(
    provider="fal-ai",
    api_key=os.environ["HF_ACCESS_TOKEN"],
)

story = ""

#load model
model = ChatOllama(model="llama3.1")

MANAGER_AGENT_PROMPT = """
    Pretend you are a writer creating a manga. Based on the provided bullet point, generate the story.
    The scene also includes information about the scene setting, characters in the scene and the last panel of the story.
    You are also provided all information about all future events. The future events information may be empty. Do not generate your own upcoming scenes.
    Only generate story for the provided bullet point, not for future events.
    For every bullet point in the scene, follow this step:
        Step 1. use the "generate_story" tool to generate a section of the story based on the current bullet point in the scene, the last panel of the story generated so far, the character and setting information and information about future events.
        If there are no future scenes, provide an empty string for future events field.
        The format of "generate_story" should be: generate_story(bullet_point=bullet_point, last_panel=last_panel, future_events=future_events, characters=characters, setting=setting).

    Make only 1 tool call at a time, and wait for the response before making the next tool call.
    Return all generated panels for the current bullet point.
    Do not return any extra information, just the panels.
"""

SUMMARY_AGENT_PROMPT = """
    You are an agent whose job it is to summarize a story provided in the prompt.
    The summary should be no longer than 5 sentences and should capture the main plot points and themes of the story.
    The summary should be concise and should not include any unnecessary details.
    Return only the summary, do not include any extra text.
    If no story text is provided, return an empty string.
"""

STORY_AGENT_PROMPT = """
    Pretend you are a writer making panels for a manga. Continue the story from the last panel, following the outline title and keeping in mind the overall story summary.
    Do not include the last panel in the newly generated panels.
    Make sure that the generated story fits the provided genres, if any are provided.
    You are also provided with a list of future events. Make sure that the generated story does not clash with any of the future events and that the story can flow naturally into the next event.
    Only generate story content that is relevant to the current bullet point in the outline. Do not include any content that is not relevant to the current bullet point, even if it is relevant to the overall story.
    Include the characters provided. You don't have to include all characters, only the most relevant ones to the current section of the story.
    Try to incorporate the scene setting into the panel description.
    Generate a sequence of manga panels in text form that continue the story. Each panel should have a description of the scene and any dialogue between characters. The panels should be formatted as follows:
        **Panel 1**:
        *Scene Description: Description of the scene.
        *Character Name: Dialogue here.
    Generate at least 3 panels for each section of the story, but feel free to generate more if you think it is necessary to continue the story in a compelling way, but do not generate more than 10.
    Return only the generated panels, do not include any extra text.
"""

CHARACTER_DETECT_AGENT = """
    You are an agent whose job it is to return information about the characters.
    You are given a list of all characters, return information about those characters that are included in mentioned characters.
    Structure the output as such for every character mentioned:
        "Full Name", "Gender (male/female)"
        Physical Description: "the character's physical description"
        Personality Description: "the character's personality description"
        Connection to Characters: "their connection to other characters"
    Return only the character information, do not include any extra text.
"""

@dataclass
class InputData:
    synopsis: str = ""
    characters: str = ""
    outline: str = ""
    genres: str = ""

class StoryState(AgentState):
    text: str


@tool
def summarize_story(runtime: ToolRuntime) -> str:
    """Summarize the story saved in the state generated so far."""
    global story
    try:
        text = story
    except KeyError:
        text = ""
    response = summary_agent.invoke({
        "messages": [HumanMessage(content=text)]
    })
    return response["messages"][-1].content

@tool
def update_story(story_text: str, runtime: ToolRuntime) -> Command:
    """Update the story text in the state with the newly generated section."""
    global story
    story += story_text + "\n"
    try:
        return Command(update={
            "text": runtime.state["text"] + "\n" + story_text,
            "messages": [ToolMessage("Successfully updated story text", tool_call_id=runtime.tool_call_id)]
            })
    except KeyError:
        return Command(update={
            "text": story_text,
            "messages": [ToolMessage("Successfully updated story text", tool_call_id=runtime.tool_call_id)]
            })

@tool
def get_story(runtime: ToolRuntime) -> str:
    """Get the story text from the state."""
    global story
    try:
        return story
    except KeyError:
        return ""

# @tool
# def get_outline(runtime: ToolRuntime) -> str:
#     """Get the outline from the context."""
#     return runtime.context.outline



CHARACTER_AGENT_SYSTEM_PROMPT = """
    Pretend you are a writer creating a manga. Based on the provided synposis in the user's prompt, create a list of characters that would be suitable for the story.
    The characters should be diverse and interesting, with unique personalities and backgrounds that fit the story's synopsis.
    Format the character description as such:
        "Full Name", "Gender (male/female)"
        Physical Description: "the character's physical description"
        Personality Description: "the character's personality description"
        Connection to Characters: "their connection to other characters"
    Return the generated characters as a string.
    Return only information about the characters, do not generate any extra text at the start or the end.
"""
#

OUTLINE_AGENT_SYSTEM_PROMPT = """
    Pretend that you are a writer creating a general outline for a manga. Base the outline on the provided synopsis in the user's prompt.
    Generate an amount of scenes equal to the requested amount in the user's prompt if it is specified.
    The outline should include characters also provided in the user's prompt. The outline should be in a structured format, with clear sections and bullet points for each part of the story.
    Also include information about which characters are present for each scene. Include information about the scene setting. Include only one specific setting, not an option between two or more.
    Generate the scene outlines one by one.
    The outline should be in a structured as such:
        **"Scene number and name"**
            Setting: "place where the scene is set"
            Characters: "characters in scene"
            Bullet Points:
                *"first bullet point for scene"
                *"second bullet point for scene"
                etc.
    The outline should be detailed enough to provide a clear roadmap for writing the manga, but it should not include any actual story content or dialogue.
    Focus on creating a high-level overview of the story's structure and key elements based on the provided synopsis and characters.
    Every scene should have atleast 3 bullet points.
    Return the generated outline as a list of strings, with each string representing a section of the outline.
    Return only the outline, do not generate any extra text.
"""
    #     The outline should include at least these sections:
    # 1. Story premise: A starting point that sets the stage for the story.
    # 2. Middle development: A section that describes the main events and conflicts that occur in the middle of the story and lead to the story's ending.
    # 3. Ending: A section that describes how the story concludes and resolves the main conflicts.
    # The above sections are just a starting point. You can add more sections if you think they are necessary to create a comprehensive outline for the manga.

OUTLINE_MANAGER_AGENT_SYSTEM_PROMPT = """
    Pretend you are a writer creating a manga. Based on the provided synopsis by the user, you will first generate a list of characters that would be suitable for the story using the "generate_characters" tool.
    Wait for the tool to generate the characters before proceeding to the next step.
    Only after the characters have been generated, you will then create a general outline for the manga based on the provided synopsis and the generated characters.
    The character information should be the same as the output from the "generate_characters" tool. Do not modify or add any extra information about the characters.
    Return the generated story outline as a list of strings, with each string representing a section of the outline.
    The outline should be long enough to generate the designated amount of chapters by the user, and it must include an ending.
    Return only the outline, do not generate any extra text at the start or end.
"""

HELP_AGENT_SYSTEM_PROMPT = """
    You are a helpful agent designed to guide the user through this application.
    The purpose of this application is to generate manga panels from an initial prompt and a selection of genres.
    The user can also generate just the manga synopsis, characters, outline, panel information or image prompts without creating the entire manga.
    The user can also enter context information from the panel on the left.
    If the user asks for help, explain this to them.
    You can also answer any general questions they may have or help with any tasks.
"""

    # To generate the characters, you will use the tool "generate_characters" which takes the synopsis as input and returns a list of characters.
    # To generate the outline, you will use the tool "generate_outline" which takes the synopsis and the generated characters as input and returns a structured outline for the manga.

PANEL_PROMPT_AGENT_SYSTEM_PROMPT = """
    You are a helpful agent that creates prompts to generate images through image generation software.
    You are given a panel that contains the scene description of that panel.
    You are also given a series of character descriptions including their physical description.
    You are also given the last prompt generated, which you can use to maintain consistency in the generated images.
    Based on the scene description and the character's physical description, generate a prompt for the panel that could
    be used to generate an image. The prompt should be about a maximum of 2 sentences.
    Structure the message as such:
        **Panel "number of panel"**
        Prompt: "the image prompt"
    If you encounter a character not included in the prompt, generate an appearance for them and use that same appearance going forward.
    Do not include character names, instead replace them with their physical description.
    Do not include any characters that aren't present in the scene description.
    In the prompt, do not include information about character conversations.
    Return only the prompts with no extra text.
"""

@tool
def generate_characters(synopsis: str) -> str:
    """Generate characters based on the provided synopsis."""
    response = character_agent.invoke({
        "messages": [HumanMessage(content=synopsis)]
    })
    return response["messages"][-1].content

@tool
def generate_outline(synopsis: str, characters: str) -> str:
    """Generate an outline based on the provided synopsis and characters."""
    response = outline_agent.invoke({
        "messages": [HumanMessage(content=f"Synopsis: {synopsis}\nCharacters: {characters}")]
    })
    return response["messages"][-1].content

help_agent = create_agent(
    model=model,
    name="help_agent",
    system_prompt=HELP_AGENT_SYSTEM_PROMPT,
)

summary_agent = create_agent(
    model=model,
    name="summary_agent",
    system_prompt=SUMMARY_AGENT_PROMPT,
)

story_agent = create_agent(
    model=model,
    name="story_agent",
    system_prompt=STORY_AGENT_PROMPT,
)

character_detect_agent = create_agent(
    model=model,
    name="character_detect_agent",
    system_prompt=CHARACTER_DETECT_AGENT
)

# manager_agent = create_agent(
#     model = model,
#     name="manager_agent",
#     system_prompt=MANAGER_AGENT_PROMPT,
#     tools=[update_story, get_story],
#     context_schema=InputData,
#     state_schema=StoryState,
# )

character_agent = create_agent(
    model=model,
    name="character_agent",
    system_prompt=CHARACTER_AGENT_SYSTEM_PROMPT
)

outline_agent = create_agent(
    model=model,
    name="outline_agent",
    system_prompt=OUTLINE_AGENT_SYSTEM_PROMPT
)

outline_manager_agent = create_agent(
    model=model,
    name="outline_manager_agent",
    tools = [generate_characters],
    system_prompt=OUTLINE_MANAGER_AGENT_SYSTEM_PROMPT,
)

image_prompt_agent = create_agent(
    model=model,
    name="image_prompt_agent",
    system_prompt=PANEL_PROMPT_AGENT_SYSTEM_PROMPT,
)


app = Flask(__name__)
CORS(app)

def generate_story(bullet_point: str, last_panel: str, future_events: str, characters: str, setting:str, genres: str) -> str:
    """Generate a section of the story based on the provided bullet point, last sentence, story summary, and characters."""
    global story
    if story != "":
        story_summary = summary_agent.invoke({
            "messages": [HumanMessage(content=story)]
        })["messages"][-1].content
    else:
        story_summary = ""

    prompt = f"Bullet Point: {bullet_point}\nLast Panel: {last_panel}\nFuture Events: {future_events}\nStory Summary: {story_summary}\nCharacters: {characters}\nGenres: {genres}\nSetting: {setting}"
    response = story_agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })
    return response["messages"][-1].content + "\n\n"

def generate_story_panels(outline: str, synopsis: str, characters: str, genres: str) -> str:
    """Generates story panels from input."""
    global story
    story = ""
    scenes = outline.split("**Scene")
    print(scenes)
    for scene in scenes:
        index = scenes.index(scene)
        
        if(index == 0):
            continue
        
        scene_characters_index = scene.index("Characters")
        scene_setting_index = scene.index("Setting")
        bullet_points_index = scene.index("Bullet Points")
        scene_setting = scene[scene_setting_index + len("Setting: "):scene_characters_index]
        scene_characters = scene[scene_characters_index + len("Characters: "):bullet_points_index]
        bullet_points = scene[scene_characters_index:].split("* ")[1:]
        
        future_scenes = ""
        future_bullet_points = ""
        for i in range(index + 1, len(scenes)):
            future_scene = scenes[i]
            future_scenes += "**Scene" + scenes[i] + "\n"
            future_bullet_points_index = future_scene.index("Bullet Points") + len("Bullet Points:\n")
            future_bullet_points += future_scene[future_bullet_points_index:]


        
        scene_characters = character_detect_agent.invoke({
            "messages": [HumanMessage(content="all characters: " + characters + "\nmentioned characters: " + scene_characters)]
        })["messages"][-1].content
        
        for point in bullet_points:
            index = bullet_points.index(point)
            future_points = "\n"
            for i in range(index + 1, len(bullet_points)):
                future_points += "* " + bullet_points[i]

            last_panel = story.split("**Panel ")[-1]

            story += generate_story(point, last_panel, future_points + future_bullet_points, scene_characters, scene_setting, genres)

    return story


def generate_prompts(panels: str, characters: str) -> str:
    split_panels = panels.split("**Panel ")[1:]
    image_prompts = ""
    last_prompt = ""
    for pan in split_panels:
        res = image_prompt_agent.invoke({
            "messages": [HumanMessage(content=f"Panel: **{pan}\n\nCharacters: {characters}\n\nLast Prompt: {last_prompt}")]
        }
        )["messages"][-1].content + "\n"
        image_prompts += res
        last_prompt = res

    return image_prompts

@app.route("/help")
def run_prompt_gen():
    req = request.form.get("param1", "")
    print("Helping with prompt: " + req)
    return jsonify(help_agent.invoke({
        "messages": [HumanMessage(content=req)]
    })["messages"][-1].content)

@app.route("/manga", methods=['GET', 'POST'])
def create_manga():
    req = request.form.get("param1", "")
    scenes = request.form.get("scenes", "5")
    genres = request.form.get("genres", "")
    if(genres != ""):
        req += " genres: " + genres
    print("Creating manga from prompt: " + req)
    pipe, lora_model = load_synopsis_model()

    synopsis = generate_synopsis(req, pipe)

    print(synopsis)
    del lora_model
    del pipe

    characters = character_agent.invoke({
        "messages": [HumanMessage(content=synopsis)]
    })["messages"][-1].content

    outline = outline_agent.invoke({
        "messages": [HumanMessage(content=f"Synopsis: {synopsis}\nCharacters: {characters}\nScenes: {scenes}")]
    })["messages"][-1].content

    panels = generate_story_panels(outline, synopsis, characters, genres)

    image_prompts = generate_prompts(panels, characters)

    values = {"synopsis": synopsis, "characters": characters, "outline": outline, "panels": panels, "prompts": image_prompts}

    return jsonify(values)



@app.route("/synopsis", methods=['GET', 'POST'])
def run_synopsis_gen():
    req = request.form.get("param1", "")
    genres = request.form.get("genres", "");
    if(genres != ""):
        req += " genres: " + genres

    print("Creating synopsis from prompt: " + req)
    pipe, lora_model = load_synopsis_model();

    synopsis = generate_synopsis(req, pipe)

    print(synopsis)
    del lora_model
    del pipe

    return jsonify(synopsis)

@app.route("/characters", methods=['GET', 'POST'])
def run_character_agent():
    synopsis = request.form.get("param1", "")
    print("Character agent invoked with input: " + synopsis)
    
    response = character_agent.invoke({
        "messages": [HumanMessage(content=synopsis)]}
    )
    return jsonify(response["messages"][-1].content)

@app.route("/outline", methods=['GET', 'POST'])
def run_outline_agent():
    synopsis = request.form.get("param1", "")
    scenes = request.form.get("scenes", "5")
    characters = request.form.get("characters", "")

    print("Outline agent invoked with input: " + synopsis)

    
    response = outline_agent.invoke({
        "messages": [HumanMessage(content=f"Synopsis: {synopsis}\nCharacters: {characters}\nScenes: {scenes}")]}
    )
    return jsonify(response["messages"][-1].content)

@app.route("/panels", methods=['GET', 'POST'])
def run_manager_agent():
    outline = request.form.get("param1", "")
    genres = request.form.get("genres", "");
    synopsis = request.form.get("synopsis", "")
    characters = request.form.get("characters", "")

    print("Panel agent invoked with input: " + outline)

    story = generate_story_panels(outline, synopsis, characters, genres)
    return jsonify(story)

@app.route("/prompts", methods=['GET', 'POST'])
def run_prompt_agent():
    panels = request.form.get("param1", "")
    characters = request.form.get("characters", "")
    print("Prompt manager invoked with input: " + panels)

    response = generate_prompts(panels, characters)
    
    return jsonify(response)

@app.route("/image_dummy", methods=['GET', 'POST'])
def dummy_image():
    
    image = Image.open("Images/image.png")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    

    return jsonify(img_b64)

@app.route("/image", methods=['GET', 'POST'])
def run_image_agent():
    image_prompt = request.form.get("param1", "")
    width = request.form.get("width", "576")
    height = request.form.get("height", "1024")
    print("Image generator invoked with input: " + image_prompt)
    
    image = client.text_to_image(
        image_prompt,
        width=int(width),
        height=int(height),
        model="Tongyi-MAI/Z-Image-Turbo",
    )
    # device = "cuda"
    # pipe = load_image_model()

    # image = pipe(
    #     prompt=image_prompt,
    #     height=int(height),
    #     width=int(width),
    #     guidance_scale=1.0,
    #     num_inference_steps=4,
    #     generator=torch.Generator(device=device).manual_seed(0)
    # ).images[0]

    #image.save("image.png")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    

    return jsonify(img_b64)

@app.route("/bot_response_stream", methods=['GET', 'POST'])
def run_manager_agent_stream():
    print("Manager agent invoked with input: " + request.form.get("param1", ""))

    # response = manager_agent.invoke({
    #     "messages": [HumanMessage(content=request.form.get("param1", ""))]},
    #     context=InputData(
    #         synopsis=synopsis,
    #         characters=characters,
    #         outline=outline)
    # )
    # return jsonify(response["messages"][-1].content)
    
    def generate():
        for token, metadata in manager_agent.stream(
            {"messages": [HumanMessage(content=request.form.get("param1", ""))]},
            stream_mode="messages",
            context=InputData(
                synopsis=synopsis,
                characters=characters,
                outline=outline)
        ):


        # token is a message chunk with token content
        # metadata contains which node produced the token
            if token.content:  # Check if there's actual content
                yield f"data: {json.dumps({"text": token.content})}\n\n"  # Get the text of the last message chunk
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

    # return jsonify("You wrote: " + request.args.get("param1", ""))
