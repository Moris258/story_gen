from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama

load_dotenv()

# class StoryState(AgentState):
#     synopsis: str
#     characters: list[str]
#     outline: str

@dataclass
class StructuredResponse:
    output: list[str]

#load the model

# llm=HuggingFaceEndpoint(
#     repo_id = "deepseek-ai/DeepSeek-V3.2",
#     task = "text-generation",
#     provider="auto"
# )

# model = ChatHuggingFace(llm=llm, temperature=1, max_tokens=2048)
model = ChatOllama(model="llama3.1")



CHARACTER_AGENT_SYSTEM_PROMPT = """
    Pretend you are a writer creating a manga. Based on the provided synposis in the user's prompt, create a list of characters that would be suitable for the story.
    The characters should be diverse and interesting, with unique personalities and backgrounds that fit the story's synopsis.
    In the description of each character, include their name, personality traits and connection to other characters in the story.
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

MANAGER_AGENT_SYSTEM_PROMPT = """
    Pretend you are a writer creating a manga. Based on the provided synopsis by the user, you will first generate a list of characters that would be suitable for the story using the "generate_characters" tool.
    Wait for the tool to generate the characters before proceeding to the next step.
    Only after the characters have been generated, you will then create a general outline for the manga based on the provided synopsis and the generated characters.
    The character information should be the same as the output from the "generate_characters" tool. Do not modify or add any extra information about the characters.
    Return the generated story outline as a list of strings, with each string representing a section of the outline.
    The outline should be long enough to generate the designated amount of chapters by the user, and it must include an ending.
    Return only the outline, do not generate any extra text.
"""

    # To generate the characters, you will use the tool "generate_characters" which takes the synopsis as input and returns a list of characters.
    # To generate the outline, you will use the tool "generate_outline" which takes the synopsis and the generated characters as input and returns a structured outline for the manga.

character_agent = create_agent(
    model=model,
    system_prompt=CHARACTER_AGENT_SYSTEM_PROMPT
)

outline_agent = create_agent(
    model=model,
    system_prompt=OUTLINE_AGENT_SYSTEM_PROMPT
)

synopsis = """
    The manga centers on a girl named Mio who is lonely because she has no
    friends. One day, she finds a boy at her school named Kuroe who is in the
    same boat as her. Kuroe can see and communicate with spirits, and he tells
    Mio that she can also see them. He offers to help her deal with them, but
    then tells her he has to leave school due to bullies. Mio wants to help
    Kuroe, but he tells her to stay away from the bullies. Then one day, a
    spirit appears to Mio and tells her that Kuroe is in danger.
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


outline_manager_agent = create_agent(
    model=model,
    tools = [generate_characters],
    system_prompt=MANAGER_AGENT_SYSTEM_PROMPT,
)