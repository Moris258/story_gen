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
    lora_adapter_path = "Moris258/Meta-Llama-3.1-8B-Instruct-Manga-Synopsis3-LORA"
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

MANAGER_AGENT_PROMPT = """
    Pretend you are a writer creating a manga. Based on the provided outline, generate the story.
    For every bullet point in the outline, follow these steps:
    1. use the "generate_story" tool to generate a section of the story based on the current bullet point in the outline, the last sentence of the story generated so far, and all following bullet points in the outline. If generating the last bullet point, provide an empty string for future events field. The format of "generate_story" should be: generate_story(bullet_point=bullet_point, last_sentence=last_sentence, future_events=future_events).
    2. use the "update_story" tool to update the story text in the state with the newly generated section. The format of "update_story" should be: update_story(story_text=generated_section).
    Make only 1 tool call at a time, and wait for the response before making the next tool call.
    Repeat the process for the next bullet point in the outline until the entire outline is covered.
    After you are done generating the story sections for all bullet points in the outline, return the complete story text that has been generated by using the "get_story" tool.
    Return only the story text, do not include any extra text.
"""

SUMMARY_AGENT_PROMPT = """
    You are an agent whose job it is to summarize a story provided in the prompt.
    The summary should be no longer than 5 sentences and should capture the main plot points and themes of the story.
    The summary should be concise and should not include any unnecessary details.
    Return only the summary, do not include any extra text.
    If no story text is provided, return an empty string.
"""

STORY_AGENT_PROMPT = """
    Pretend you are a writer making panels for a manga. Continue the story from the last sentence, following the outline title and keeping in mind the overall story summary.
    You are also provided with a list of future events. Make sure that the generated story does not clash with any of the future events and that the story can flow naturally into the next event.
    Only generate story content that is relevant to the current bullet point in the outline. Do not include any content that is not relevant to the current bullet point, even if it is relevant to the overall story.
    Include the characters provided. You don't have to include all characters, only the most relevant ones to the current section of the story.
    Generate a sequence of manga panels in text form that continue the story. Each panel should have a description of the scene and any dialogue between characters. The panels should be formatted as follows:
Panel 1:
Scene description here.
Character 1: Dialogue here.
etc.
    Generate at least 3 panels for each section of the story, but feel free to generate more if you think it is necessary to continue the story in a compelling way.
    Return only the generated panels, do not include any extra text.
"""

CHARACTER_DETECT_AGENT = """
    You are an agent whose job is to identify the characters that are present in a given section of the story.
    The characters you should be looking for will be provided in the prompt.
    Be quite strict with yourself and only identify characters that are explicitly mentioned in the story section. Do not make assumptions or guesses about characters that may be present based on the context of the story.
    Return the characters detected as a string, including all the information about the character provided.
    Return only the character information, do not include any extra text.
"""

@dataclass
class InputData:
    synopsis: str = synopsis
    characters: str = characters
    outline: str = outline

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
def generate_story(bullet_point: str, last_sentence: str, future_events: str, runtime: ToolRuntime) -> str:
    """Generate a section of the story based on the provided bullet point, last sentence, story summary, and characters."""
    characters = character_detect_agent.invoke({
        "messages": [HumanMessage(content=bullet_point + "\n" + runtime.context.characters)]
    })["messages"][-1].content
    try:
        story_summary = summary_agent.invoke({
            "messages": [HumanMessage(content=runtime.state["text"])]
        })["messages"][-1].content
    except KeyError:
        story_summary = ""


    prompt = f"Bullet Point: {bullet_point}\nLast Sentence: {last_sentence}\nFuture Events: {future_events}\nStory Summary: {story_summary}\nCharacters: {characters}"
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
        "Full Name"
        Physical Description: "the character's physical description"
        Personality Description: "the character's personality description"
        Connection to Characters: "their connection to other characters"
    Return the generated characters as a string.
    Return only information about the characters, do not generate any extra text at the start or the end.
"""
#

OUTLINE_AGENT_SYSTEM_PROMPT = """
    Pretend that you are a writer creating a general outline for a manga. Base the outline on the provided synopsis in the user's prompt.
    The outline should include characters also provided in the user's prompt. The outline should be in a structured format, with clear sections and bullet points for each part of the story
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
    The purpose of this application is to generate manga panels from an initial prompt.
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

Panel 1:
Scene description: The empty hallway stretches out before Mio as she walks towards her locker. She keeps her head down, avoiding eye contact with anyone.
Mio Katsuragi: (to herself) Another day...

Panel 2:
Scene description: Mio's gaze drifts towards the cafeteria, where students are gathered for lunch. She spots a group of her classmates laughing and chatting together.
Mio Katsuragi: (thinking) Why can't I be like them? They have friends, they're popular... why do I always feel so alone?

Panel 3:
Scene description: Mio's eyes well up with tears as she opens her locker door. She pulls out a book and begins to scribble notes in the margins.
Mio Katsuragi: (whispering) Just one person, just one friend... that's all I want...

Panel 4:
Scene description: Students begin to gather around their lockers, chatting with each other and sharing stories. Mio feels a pang of sadness as she realizes she'll be spending another day alone.
Mio Katsuragi: (thinking) Why can't anyone see past my face? Past the surface level?

Panel 5:
Scene description: A group of students glance in Mio's direction, their faces filled with curiosity and amusement. They whisper to each other, and one of them cracks a joke at her expense.
Mio Katsuragi: (defensively) Shut up... just leave me alone...

Panel 6:
Scene description: The scene shifts to the school courtyard, where students are playing sports or lounging on benches. Mio sits on a bench, looking down at her feet.
Sakura Tanaka: Hi! Mind if I join you? I saw you sitting here all by yourself, and I thought it might be nice to have some company.

Panel 3:
Scene description: Sakura sits down next to Mio on the bench, unfolding a sandwich from her own lunch box. She offers a warm smile as she begins to eat.
Sakura Tanaka: Great! My name is Sakura, by the way. I'm new here this year.

Panel 4:
Scene description: Mio looks down at her own lunch, then back up to Sakura with a hint of curiosity. Sakura continues to eat and chat, not seeming to notice Mio's initial hesitation.
Sakura Tanaka: So, what brings you to our school? Are you from around here?
Mio Katsuragi: (hesitantly) Y-yeah... I'm from around here.

Panel 5:
Scene description: Sakura takes a bite of her sandwich, and her eyes light up with delight. Mio can't help but be drawn in by Sakura's infectious enthusiasm.
Sakura Tanaka: Mmm, this is so good! Have you tried the sandwiches at the café near school? They're amazing!
Mio Katsuragi: (softening) N-no... I don't usually get lunch from there.

Panel 6:
Scene description: The two girls continue to talk and laugh together, their conversation flowing easily. For the first time all day, Mio feels like she's found someone who doesn't see her as an outcast.
Sakura Tanaka: So, what do you like to do for fun? I'm really into art and music – maybe we can exchange recommendations sometime?
Mio Katsuragi: (smiling slightly) Y-yeah... that sounds nice.

Panel 7:
Scene description: The bell rings, signaling the end of lunchtime. Sakura looks at Mio with a friendly smile.
Sakura Tanaka: Well, I should probably get going. But it was really great talking to you, Mio! Maybe we can sit together again tomorrow?
Mio Katsuragi: (smiling more genuinely) Y-yeah... that would be nice.

Panel 1:
Scene description: A quiet corner of the school hallway during lunchtime. Mio stands alone, looking down at her feet, while Sakura sits on the floor beside her, chatting.
Mio: I'm just glad we're not in that awful math class together...
Sakura Tanaka: (laughs) Yeah, Mr. Tanaka is crazy! Do you think he'd make us do algebra for the rest of our lives?

Panel 2:
Scene description: Sakura glances over at Mio, concerned.
Sakura Tanaka: You okay? You seem a bit... lost again.
Mio Katsuragi: (smiling more genuinely) Y-yeah... that would be nice.

Panel 3:
Scene description: A boy with messy black hair and sunglasses walks into the hallway. He's wearing a white school jacket with no emblem, and his eyes scan the area as if searching for someone.
Kuroe Shinoda: (whispering to himself) She's not alone...


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





app = Flask(__name__)
CORS(app)


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

    panels = manager_agent.invoke({
        "messages": [HumanMessage(content=outline)]},
        context=InputData(
            synopsis=synopsis,
            characters=characters,
            outline=outline,
        )
    )["messages"][-1].content

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

    print("Outline manager agent invoked with input: " + synopsis)

    
    response = outline_manager_agent.invoke({
        "messages": [HumanMessage(content=f"Characters: {characters}\nSynopsis: {synopsis}")]}
    )
    return jsonify(response["messages"][-1].content)

@app.route("/panels")
def run_manager_agent():
    outline = request.args.get("param1", "")
    synopsis = request.args.get("synopsis", "")
    characters = request.args.get("characters", "")

    print("Manager agent invoked with input: " + outline)

    response = manager_agent.invoke({
        "messages": [HumanMessage(content=outline)]},
        context=InputData(
            synopsis=synopsis,
            characters=characters,
            outline=outline)
    )
    return jsonify(response["messages"][-1].content)

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
