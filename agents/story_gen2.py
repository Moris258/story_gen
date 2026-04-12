from dotenv import load_dotenv
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langgraph.types import Command

load_dotenv()

class SystemState(AgentState):
    synopsis: str
    characters: str
    outline: str

model = ChatOllama(model="llama3.1")

SYSTEM_PROMPT = """
    Pretend you are a writer creating a manga. Based on the provided synopsis generate a list of characters that would be suitable for the story. You can include characters not explicitly mentioned in the synopsis.
    Include information about the characters such as their name, personality traits and connection to other characters in the story.
    Then create a general outline for the manga based on the provided synopsis and the generated characters.
    The outline should be about 3 chapters long.
    The outline should be in a structured format, with clear sections and bullet points for each part of the story.
    Do not include information about key scenes.
    The outline should be detailed enough to provide a clear roadmap for writing the manga, but it should not include any actual story content or dialogue.
    Focus on creating a high-level overview of the story's structure and key elements based on the provided synopsis and characters.
    Save the synopsis, generated characters, and outline information in the agent's state using the "save_state_information" tool.
    When calling the "save_state_information" tool, make sure to provide the arguments in string format.
"""

@tool
def save_state_information(synopsis: str, characters: str, outline: str, runtime: ToolRuntime) -> Command:
    """A tool to save the synopsis, generated characters, and outline information in the agent's state."""
    return Command(update={
        "synopsis": synopsis,
        "characters": characters,
        "outline": outline,
        "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)]}
    )



agent = create_agent(
    model = model,
    system_prompt=SYSTEM_PROMPT,
    state_schema=SystemState,
    tools=[save_state_information]
)