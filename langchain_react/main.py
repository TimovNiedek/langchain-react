from typing import List, Union, Tuple

from dotenv import load_dotenv
from langchain.agents import tool, Tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def get_text_length(text: str) -> str:
    """Returns the length of a text by characters"""
    text = text.strip().strip("'\n\"")  # Remove quotes and newlines
    return str(len(text))


def find_tool_by_name(tools: List[Tool], name: str) -> Tool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool with name '{name}' not found in tools list")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")

    tools: List[Tool] = [get_text_length]

    # ReAct prompt (a one-shot learning, chain of thoughts prompt)
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    # OpenAI GPT-3.5-turbo model
    # stop argument is used to stop the model from generating more text after the final answer
    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", model_kwargs={"stop": ["\nObservation", "Observation"]})

    # Agent scratchpad to store intermediate steps
    # Formatted as list of (action, observation) tuples
    intermediate_steps: List[Tuple[AgentAction, str]] = []

    # ReAct agent
    input_dict = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
    }
    agent = input_dict | prompt | llm | ReActSingleInputOutputParser()

    # Run the agent (reasoning engine)
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of the text 'Hello, World!' in characters?",
            "agent_scratchpad": intermediate_steps
        }
    )

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool.func(str(tool_input))
        print(f"{observation=}")
        intermediate_steps.append((agent_step, str(observation)))

    # Execute it again
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of the text 'Hello, World!' in characters?",
            "agent_scratchpad": intermediate_steps
        }
    )
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
