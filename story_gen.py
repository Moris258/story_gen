from dotenv import load_dotenv
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Pipeline
from peft import PeftModel
from langgraph.types import Command
from flask import Flask, Response, json, request, stream_with_context
from flask_cors import CORS
from flask import jsonify
import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import io
import base64
from PIL import Image
import os
import gc
from huggingface_hub import InferenceClient
from sdnq import SDNQConfig # import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
import random

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
    lora_adapter_path = "Moris258/Meta-Llama-3.1-8B-Instruct-Manga-Synopsis-v.5-LORA"
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

def load_image_model() -> DiffusionPipeline:
    dtype = torch.bfloat16

    pipe = Flux2KleinPipeline.from_pretrained("Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic", torch_dtype=dtype)

    if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
        pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
        pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

    pipe.enable_model_cpu_offload()
    return pipe

def generate_synopsis(input: str, pipe: Pipeline):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that creates manga synopses based on the given description."},
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
story_summary_container = ""
STORY_MAX_LENGTH = 10000

#load model
model = ChatOllama(model="su_robin/gemma-4-E4B-it-Q4_K_M")

SUMMARY_AGENT_PROMPT = """
    You are an agent whose job it is to summarize a story provided in the prompt.
    The summary should be no longer than 10 sentences and should capture the main plot points and themes of the story.
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
    Try to stick to the included setting when generating the scene description. Try to refer to characters by their name instead of using pronouns.
    Generate a sequence of manga panels in text form that continue the story. Each panel should have a description of the scene and any dialogue between characters. The panels should be formatted as follows:
        **Panel 1**
        *Scene Description: Description of the scene.
        *Character Name: Dialogue here.
    Try to make the descriptions and dialogue more concrete and to the point, and less vague.
    Generate at least 3 panels for each section of the story, but feel free to generate more if you think it is necessary to continue the story in a compelling way, but do not generate more than 10.
    Always start the panel numbering from 1 for each bullet point, even if the panel numbering in the overall story is different. The panel number should only indicate the order of panels for the current bullet point, not the overall story.
    Always include a scene description. If there is dialogue, include the character name followed by the dialogue. If there is no dialogue, just include the scene description.
    Return only the generated panels, do not include any extra text.
"""

CHARACTER_DETECT_AGENT = """
    You are an agent whose job it is to return information about the characters.
    You are given a list of all characters, return information about those characters that are included in mentioned characters.
    Structure the output as such for every character mentioned:
        "Full Name", "Gender (male/female)" age: "character age"
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

CHARACTER_AGENT_SYSTEM_PROMPT = """
    Pretend you are a writer creating a manga. Based on the provided synopsis in the user's prompt, create a list of characters that would be suitable for the story.
    The characters should be diverse and interesting, with unique personalities and backgrounds that fit the story's synopsis.
    Format the character description as such:
        "Full Name", "Gender (male/female)" age: "character age"
        Physical Description: "the character's physical description"
        Personality Description: "the character's personality description"
        Connection to Characters: "their connection to other characters"
    Return only information about the characters, do not generate any extra text at the start or the end.
"""
#

OUTLINE_AGENT_SYSTEM_PROMPT = """
    Pretend that you are a writer creating a general outline for a manga. Base the outline on the provided synopsis in the user's prompt.
    Generate an amount of scenes equal to the requested amount in the user's prompt if it is specified.
    The outline should include characters also provided in the user's prompt. The outline should be in a structured format, with clear sections and bullet points for each part of the story.
    The generated bullet points should be phrased as events that happen, rather than descriptions.
    Also include information about which characters are present for each scene. Only these characters should feature in this scene.
    Include information about the scene setting. Include only one specific setting, not an option between two or more.
    Generate the scene outlines one by one.
    The outline should be in a structured as such:
        **Scene number and name**
            Setting: "place where the scene is set"
            Characters: "characters in scene"
            Bullet Points:
                *"first bullet point for scene"
                *"second bullet point for scene"
                etc.
    Do not generate trailing * characters for each bullet point.
    The outline should be detailed enough to provide a clear roadmap for writing the manga, but it should not include any actual story content or dialogue.
    Focus on creating a high-level overview of the story's structure and key elements based on the provided synopsis and characters.
    Every scene should have around 3 bullet points.
    Return only the outline, do not generate any extra text.
"""

HELP_AGENT_SYSTEM_PROMPT = """
    You are a helpful agent designed to guide the user through this application.
    The purpose of this application is to generate manga panels from an initial prompt and a selection of genres.
    The user can also generate just the manga synopsis, characters, outline, panel information, image prompts or images without creating the entire manga.
    Explain to the user that the application requires certain context fields to be filled before certain generation steps.
    Explain to the user that if they wish to generate the manga step by step, they should follow the order found in the drop down menu on the webpage, the order is as follows: synopsis, characters, outline, panels, prompts, images.
    If the user asks for help, explain this to them.
    You can also answer any general questions they may have or help with any tasks. In this case you don't need to explain the application to them.
    Responds in an HTML friendly format to be displayed in a <div>. Display the message in dark mode.

# PANEL_PROMPT_AGENT_SYSTEM_PROMPT = """
#     You are a helpful agent that creates prompts to generate images through image generation software.
#     You are given a panel that contains the scene description of that panel.
#     You are also given a series of character descriptions including their physical description.
#     You are also given the last prompt generated, which you can use to maintain consistency in the generated images.
#     Based on the scene description and the character's physical description, generate a prompt for the panel that could
#     be used to generate an image. Try to match the provided scene description as closely as possible.
#     Try to use the included setting to generate background information in the prompt.
#     Structure the message as such:
#         **Panel "number of panel"**
#         Prompt: "the image prompt"
#     Do not include character names, instead replace them with their physical description. 
#     Make the physical description very detailed and based on the provided physical description.
#     Do not include any characters that aren't present in the scene description.
#     In the prompt, do not include information about character conversations.
#     Generate only one prompt per panel. Do not generate multiple prompts for the same panel.
#     Return only the prompts with no extra text.
# """

PANEL_PROMPT_AGENT_SYSTEM_PROMPT = """
    You are a helpful agent that creates prompts to generate images through image generation software.
    You are given a panel that contains the scene description of that panel.
    You are also given a series of character descriptions including their physical description and outfit information.
    You are also given the last prompt generated, which you can use to maintain consistency in the generated images.
    Based on the scene description and the character's physical description and outfit, generate a prompt for the panel that could
    be used to generate an image. Try to include every detail in the provided scene description, including character actions.
    Try to use the included setting to generate background information in the prompt.
    Structure the message as such:
        **Panel "number of panel"**
        Prompt: "the image prompt"
    Do not include character names, instead replace them with their physical description and outfit information. 
    Make the physical description very detailed and based on the provided physical description and outfit.
    Only include character descriptions in the prompt if they are mentioned in the scene description.
    In the prompt, do not include information about character conversations.
    Generate only one prompt per panel. Do not generate multiple prompts for the same panel.
    Return only the prompts with no extra text.
"""

SETTING_DETECT_AGENT_SYSTEM_PROMPT = """
    You are a helpful assistant that generates a scene setting from a manga panel description.
    You are given a panel containing information about the scene description. You should analyze that scene description and return a concise description of
    the setting that scene is taking place in. The description should not be longer than a few words.
"""

CLOTHES_GENERATOR_AGENT = """
    Pretend you are a manga writer deciding on what clothes a character should be wearing in a certain scene.
    You are given a list of characters and scene setting and a manga panel containing a scene description.
    Based on the given information, add information about each character's outfit to their character information.
    The outfit description should be generic and not specific to the provided scene.
    The outfit should be a generic description, not including names.
    Include information about the major parts of the outfit.
    Return only the character information including the newly generated outfit information.
    Do not generate any extra text.
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
image_prompt_agent = create_agent(
    model=model,
    name="image_prompt_agent",
    system_prompt=PANEL_PROMPT_AGENT_SYSTEM_PROMPT,
)

setting_detect_agent = create_agent(
    model=model,
    name="setting_detect_agent",
    system_prompt=SETTING_DETECT_AGENT_SYSTEM_PROMPT,
)

clothes_generator_agent = create_agent(
    model=model,
    name="clothes_generator_agent",
    system_prompt=CLOTHES_GENERATOR_AGENT,
)


app = Flask(__name__)
CORS(app)

def generate_story(bullet_point: str, last_panel: str, future_events: str, characters: str, setting:str, genres: str) -> str:
    """Generate a section of the story based on the provided bullet point, last sentence, story summary, and characters."""
    global story
    global story_summary_container
    if story_summary_container != "":
        story_summary = summary_agent.invoke({
            "messages": [HumanMessage(content=story_summary_container)]
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
    global story_summary_container
    story = ""
    scenes = outline.split("**Scene")[1:]
    include_last_panel = False
    for scene in scenes:
        index = scenes.index(scene)
        include_last_panel = False
        
        scene_characters_index = scene.index("Characters")
        scene_setting_index = scene.index("Setting")
        bullet_points_index = scene.index("Bullet Points")
        scene_setting = scene[scene_setting_index + len("Setting: "):scene_characters_index]
        scene_characters = scene[scene_characters_index + len("Characters: "):bullet_points_index]
        bullet_points = scene[scene_characters_index:].split("*")[1:]
        
        for point in bullet_points:
            if(point.strip() == ""):
                bullet_points.remove(point)
        
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
            print("Generating story for bullet point: " + point)
            index = bullet_points.index(point)
            future_points = "\n"
            for i in range(index + 1, len(bullet_points)):
                future_points += "* " + bullet_points[i]

            last_panel = ""
            if(include_last_panel):
                last_panel = story.split("**Panel ")[-1]

            segment = generate_story(point, last_panel, future_points + future_bullet_points, scene_characters, scene_setting, genres)
            story_summary_container += segment
            story += segment

            if(len(story_summary_container) > STORY_MAX_LENGTH):
                story_summary_container = summary_agent.invoke({
                    "messages": [HumanMessage(content=story_summary_container)]
                })["messages"][-1].content

            include_last_panel = True
    return story

def is_first_panel(panel: str) -> bool:
    if(panel[0] == "1"):
        return True
    return False

def generate_prompts(outline: str, panels: str, characters: str) -> str:
    split_panels = panels.split("**Panel ")[1:]
    image_prompts = ""
    last_prompt = ""
    setting = ""
    scene_characters = ""
    scene_index = 0
    bullet_point_index = 0
    scenes = outline.split("**Scene")[1:]
    scene_lengths = []


    for scene in scenes:
        bullet_points_index = scene.index("Bullet Points")
        bullet_points = scene[bullet_points_index:].split("*")[1:]
        
        for point in bullet_points:
            if(point.strip() == ""):
                bullet_points.remove(point)
        
        scene_lengths.append(len(bullet_points))


    for pan in split_panels:
        print("Generating image prompt for panel: " + pan)
        
        if(is_first_panel(pan)):
            if(bullet_point_index == 0):
                setting = scenes[scene_index].split("Setting:")[1].split("\n")[0]
                scene_characters = scenes[scene_index].split("Characters:")[1].split("\n")[0]
                scene_characters = character_detect_agent.invoke({
                    "messages": [HumanMessage(content=f"all characters: {characters}\n\nmentioned characters: {scene_characters}")]
                })["messages"][-1].content
                scene_characters = clothes_generator_agent.invoke({
                    "messages": [HumanMessage(content=f"Characters: {scene_characters}\n\nSetting: {setting}\n\nPanel: {pan}")]
                })["messages"][-1].content


                print("Setting: " + setting)
                print("Characters: " + scene_characters)
            
            bullet_point_index += 1
            if(bullet_point_index >= scene_lengths[scene_index]):
                scene_index += 1
                bullet_point_index = 0

        res = image_prompt_agent.invoke({
            "messages": [HumanMessage(content=f"Panel: {pan}\n\nSetting: {setting}\n\nCharacters: {scene_characters}\n\nLast Prompt: {last_prompt}")]
        }
        )["messages"][-1].content + "\n"
        image_prompts += res
        last_prompt = res




    return image_prompts

def unload_model():
    global model
    model.keep_alive = 0
    model.invoke("Generate EOL token.")
    model.keep_alive = 5

@app.route("/help", methods=['GET', 'POST'])
def run_prompt_gen():
    req = request.form.get("param1", "")
    print("Helping with prompt: " + req)
    response = help_agent.invoke({
        "messages": [HumanMessage(content=req)]
    })["messages"][-1].content
    unload_model()

    return jsonify(response)

@app.route("/synopsis", methods=['GET', 'POST'])
def run_synopsis_gen():
    req = request.form.get("param1", "")
    genres = request.form.get("genres", "");
    if(genres != ""):
        req += " genres: " + genres

    print("Creating synopsis from prompt: " + req)
    pipe, model = load_synopsis_model();

    synopsis = generate_synopsis(req, pipe)
    del model
    gc.collect()
    torch.cuda.empty_cache()   



    del pipe
    gc.collect()
    torch.cuda.empty_cache()    


    return jsonify(synopsis)

@app.route("/characters", methods=['GET', 'POST'])
def run_character_agent():
    synopsis = request.form.get("param1", "")
    print("Character agent invoked with input: " + synopsis)
    
    response = character_agent.invoke({
        "messages": [HumanMessage(content=synopsis)]}
    )
    unload_model()
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
    unload_model()
    return jsonify(response["messages"][-1].content)

@app.route("/panels", methods=['GET', 'POST'])
def run_manager_agent():
    outline = request.form.get("param1", "")
    genres = request.form.get("genres", "");
    synopsis = request.form.get("synopsis", "")
    characters = request.form.get("characters", "")

    story = generate_story_panels(outline, synopsis, characters, genres)
    unload_model()
    return jsonify(story)

@app.route("/prompts", methods=['GET', 'POST'])
def run_prompt_agent():
    panels = request.form.get("param1", "")
    characters = request.form.get("characters", "")
    outline = request.form.get("outline", "")

    response = generate_prompts(outline, panels, characters)
    unload_model()
    
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
    
    device = "cuda"
    pipe = load_image_model()

    image = pipe(
        prompt=image_prompt,
        height=int(height),
        width=int(width),
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=torch.Generator(device=device).manual_seed(random.randint(0, 10000))
    ).images[0]
    
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    

    return jsonify(img_b64)

if __name__ == "__main__":
    app.run("127.0.0.1", 4500, debug=True)
