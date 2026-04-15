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

def generate_synopsis(input: str, pipe: Pipeline):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that creates manga synopses based on the given description. The generated synopsis should be at least 500 characters."},
        {"role": "user", "content": input},
    ]

    outputs = pipe(
        messages,
    )

    return outputs[0]["generated_text"][-1]["content"][10:]



#load model
model = ChatOllama(model="llama3.1")

synopsis = """
The manga centers on a girl named Mio who is lonely because she has no friends. One day, she finds a boy at her school named Kuroe who is in the same boat as her. Kuroe can see and communicate with spirits, and he tells Mio that she can also see them. He offers to help her deal with them, but then tells her he has to leave school due to bullies. Mio wants to help Kuroe, but he tells her to stay away from the bullies. Then one day, aspirit appears to Mio and tells her that Kuroe is in danger.
"""


characters = """
    Mio Katsuragi: Personality traits: introverted, responsible, determined Connection to other characters: main protagonist, friends with Kuroe
    Kuroe Shinoda: Personality traits: quiet, observant, empathetic Connection to Mio: helps her deal with spirits, warns her about bullies
    Akira Matsumoto: Personality traits: hot-headed, aggressive, intimidating Connection to other characters: bully who targets Kuroe, rival of the main antagonist
    Sakura Tanaka: Personality traits: kind-hearted, gentle, friendly Connection to Mio: new friend at school who becomes a confidant for Mio
    Rina Yamada: Personality traits: bubbly, energetic, mischievous Connection to other characters: rival of Sakura in Mio's class, initially oblivious to the supernatural events
    Kaito Nakahara: Personality traits: charismatic, confident, manipulative Connection to Kuroe: former friend who becomes a bully due to his own struggles with spirits
    Hiroshi Inoue: Personality traits: calm, wise, mysterious Connection to other characters: spirit guide for Mio and Kuroe, provides cryptic advice
    others: various classmates, teachers, and spirits that interact with the main characters throughout the story
"""

outline = """
    Introduce protagonist Mio Katsuragi, a lonely girl who has no friends at school.
    Show Mio's daily life as an outcast, highlighting her struggles to connect with others.
    Introduce supporting character Sakura Tanaka, who becomes the first person to show kindness towards Mio.
    Introduce Kuroe Shinoda, a boy who can see and communicate with spirits.
    Show Kuroe's unique abilities as he helps Mio deal with her newfound awareness of spirits.
    Introduce antagonist Kaito Nakahara, who is struggling to control his own spirit presence.
    Set up conflict between Kuroe and Kaito, foreshadowing the danger that Kuroe will soon face.
"""

genres = "supernatural, school, drama"

MANAGER_AGENT_PROMPT = """
    Pretend you are a writer creating a manga. Based on the provided scene outline, generate the story.
    The provided outline features a scene header, indicated by a preceding "**", with bullet points for each scene.
    Bullet points are indicated by a preceding "*".
    The scene also includes information about the scene setting and characters in the scene.
    You are also provided all information about all future scenes. The future scenes information may be empty. Do not generate your own future scenes.
    Only generate story for the current scene, not future scenes.
    For every bullet point in the scene, follow these steps:
        Step 1. use the "generate_story" tool to generate a section of the story based on the current bullet point in the scene, the last panel of the story generated so far, the character and setting information and information about future events.
        Future events are all upcoming bullet points in the current scene and in all future scenes. Include bullet points from the current scene as well.
        If generating the last bullet point, provide an empty string for future events field. Character information for the current scene is included below the scene header.
        The format of "generate_story" should be: generate_story(bullet_point=bullet_point, last_panel=last_panel, future_events=future_events, characters=characters, setting=setting).

        Step 2. use the "update_story" tool to update the panels in the state with the newly generated section. Make sure to include all generated panels from the "generate_story" tool.
        The format of "update_story" should be: update_story(story_text=generated_section).

    Make only 1 tool call at a time, and wait for the response before making the next tool call.
    Repeat the process for the next bullet point in the outline until the entire outline is covered.
    After you are done generating the story sections for all bullet points in the outline, return all generated panels from the scene.
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
        Panel 1:
        Scene description here.
        Character name: Dialogue here.
        etc.
    Generate at least 3 panels for each section of the story, but feel free to generate more if you think it is necessary to continue the story in a compelling way.
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
    synopsis: str = synopsis
    characters: str = characters
    outline: str = outline
    genres: str = genres

class StoryState(AgentState):
    text: str


@tool
def summarize_story(runtime: ToolRuntime) -> str:
    """Summarize the story saved in the state generated so far."""
    try:
        text = runtime.state["text"]
    except KeyError:
        text = ""
    response = summary_agent.invoke({
        "messages": [HumanMessage(content=text)]
    })
    return response["messages"][-1].content

@tool
def update_story(story_text: str, runtime: ToolRuntime) -> Command:
    """Update the story text in the state with the newly generated section."""
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
    try:
        return runtime.state["text"]
    except KeyError:
        return ""

@tool
def generate_story(bullet_point: str, last_panel: str, future_events: str, characters: str, setting:str, runtime: ToolRuntime) -> str:
    """Generate a section of the story based on the provided bullet point, last sentence, story summary, and characters."""
    characters = character_detect_agent.invoke({
        "messages": [HumanMessage(content="all characters: " + runtime.context.characters + "\nmentioned characters: " + characters)]
    })["messages"][-1].content
    try:
        story_summary = summary_agent.invoke({
            "messages": [HumanMessage(content=runtime.state["text"])]
        })["messages"][-1].content
    except KeyError:
        story_summary = ""

    genres = runtime.context.genres
    prompt = f"Bullet Point: {bullet_point}\nLast Panel: {last_panel}\nFuture Events: {future_events}\nStory Summary: {story_summary}\nCharacters: {characters}\nGenres: {genres}\nSetting: {setting}"
    response = story_agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })
    return response["messages"][-1].content

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
    The outline should include characters also provided in the user's prompt. The outline should be in a structured format, with clear sections and bullet points for each part of the story.
    Also include information about which characters are present for each scene. Include information about the scene setting. Include only one specific setting, not an option between two or more.
    Generate the scene outlines one by one.
    The outline should be in a structured as such:
        **"Scene number and name"**
            Setting: "place where the scene is set"
            Characters: "characters in scene"
            *"first bullet point for scene"
            *"second bullet point for scene"
            etc.
    The outline should be detailed enough to provide a clear roadmap for writing the manga, but it should not include any actual story content or dialogue.
    Focus on creating a high-level overview of the story's structure and key elements based on the provided synopsis and characters.
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
    You are given a series of panels that contain the scene description of that panel.
    You are also given a series of character descriptions including their physical description.
    Based on the scene description and the character's physical description, generate prompts for each panel that could
    be used to generate an image.
    Structure the message as such:
        **Panel "number of panel"**
        Prompt: "the image prompt"
    If you encounter a character not included in the prompt, generate an appearance for them and use that same appearance going forward.
    Do not include character names, instead replace them with their physical description.
    Avoid references to previous or future panels, focus only on the current panel.
    Return only the prompts with no extra text.
"""


panels = """
Panel 1:
Scene description: A shot of Mio walking down the empty hallway of her school, looking lost in thought. Her backpack is slumped over one shoulder, and she seems to be avoiding eye contact with anyone.
Mio (thought bubble): I hate coming here every day. Everyone's so mean, and they don't care that I'm just sitting there, trying not to get noticed.

Panel 2:
Scene description: Mio sits alone at a table in the school cafeteria, eating her lunch while staring down at her plate.
Mio (thought bubble): Why can't anyone talk to me? Don't they know I'm not so bad?

Panel 3:
Scene description: The cafeteria falls silent as students start to gather around their lockers. Mio's eyes follow them, feeling like an outcast.
Mio (sighs): Just another day...
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

manager_agent = create_agent(
    model = model,
    name="manager_agent",
    system_prompt=MANAGER_AGENT_PROMPT,
    tools=[generate_story, update_story, get_story],
    context_schema=InputData,
    state_schema=StoryState,
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

#TODO: make sure that character passing to generate_story is working, alternatively just do it manually.



app = Flask(__name__)
CORS(app)

def generate_story(outline: str, synopsis: str, characters: str, genres: str) -> str:
    story = ""
    scenes = outline.split("**Scene")
    print(scenes)
    for scene in scenes:
        index = scenes.index(scene)
        
        if(index == 0):
            continue
        future_scenes = ""
        for i in range(index + 1, len(scenes)):
            future_scenes += "**Scene" + scenes[i] + "\n"

        story += manager_agent.invoke({
            "messages": [HumanMessage(content=f"Current scene: **Scene{scene}\n\nFuture scenes: {future_scenes}")]
        },
            context=InputData(
                synopsis=synopsis,
                characters=characters,
                outline=outline,
                genres=genres
            )
        )["messages"][-1].content

    return story



@app.route("/help")
def run_prompt_gen():
    req = request.args.get("param1", "")
    print("Helping with prompt: " + req)
    return jsonify(help_agent.invoke({
        "messages": [HumanMessage(content=req)]
    })["messages"][-1].content)

@app.route("/manga")
def create_manga():
    req = request.args.get("param1", "")
    genres = request.args.get("genres", "");
    if(genres != ""):
        req += " genres: " + genres
    print("Creating manga from prompt: " + req)
    pipe, lora_model = load_synopsis_model();

    synopsis = generate_synopsis(req, pipe)

    print(synopsis)
    del lora_model
    del pipe

    characters = character_agent.invoke({
        "messages": [HumanMessage(content=synopsis)]
    })["messages"][-1].content

    outline = outline_agent.invoke({
        "messages": [HumanMessage(content=f"Synopsis: {synopsis}\nCharacters: {characters}")]
    })["messages"][-1].content

    panels = generate_story(outline, synopsis, characters, genres)

    image_prompts = image_prompt_agent.invoke({
        "messages": [HumanMessage(content=f"Panels: {panels}\nCharacters: {characters}")]
    }
    )["messages"][-1].content


    return jsonify(image_prompts)



@app.route("/synopsis")
def run_synopsis_gen():
    req = request.args.get("param1", "")
    genres = request.args.get("genres", "");
    if(genres != ""):
        req += " genres: " + genres

    print("Creating synopsis from prompt: " + req)
    pipe, lora_model = load_synopsis_model();

    synopsis = generate_synopsis(req, pipe)

    print(synopsis)
    del lora_model
    del pipe

    return jsonify(synopsis)

@app.route("/characters")
def run_character_agent():
    synopsis = request.args.get("param1", "")
    print("Character agent invoked with input: " + synopsis)
    
    response = character_agent.invoke({
        "messages": [HumanMessage(content=synopsis)]}
    )
    return jsonify(response["messages"][-1].content)

@app.route("/outline")
def run_outline_agent():
    synopsis = request.args.get("param1", "")
    characters = request.args.get("characters", "")

    print("Outline agent invoked with input: " + synopsis)

    
    response = outline_agent.invoke({
        "messages": [HumanMessage(content=f"Characters: {characters}\nSynopsis: {synopsis}")]}
    )
    return jsonify(response["messages"][-1].content)

@app.route("/panels")
def run_manager_agent():
    outline = request.args.get("param1", "")
    genres = request.args.get("genres", "");
    synopsis = request.args.get("synopsis", "")
    characters = request.args.get("characters", "")

    print("Manager agent invoked with input: " + outline)

    story = generate_story(outline, synopsis, characters, genres)
    return jsonify(story)

@app.route("/prompts")
def run_prompt_agent():
    panels = request.args.get("param1", "")
    characters = request.args.get("characters", "")
    print("Prompt manager invoked with input: " + panels)

    response = image_prompt_agent.invoke({
        "messages": [HumanMessage(content=f"Panels: {panels}\nCharacters: {characters}")]
    }
    )["messages"][-1].content
    
    return jsonify(response)


@app.route("/bot_response_stream")
def run_manager_agent_stream():
    print("Manager agent invoked with input: " + request.args.get("param1", ""))

    # response = manager_agent.invoke({
    #     "messages": [HumanMessage(content=request.args.get("param1", ""))]},
    #     context=InputData(
    #         synopsis=synopsis,
    #         characters=characters,
    #         outline=outline)
    # )
    # return jsonify(response["messages"][-1].content)
    
    def generate():
        for token, metadata in manager_agent.stream(
            {"messages": [HumanMessage(content=request.args.get("param1", ""))]},
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
