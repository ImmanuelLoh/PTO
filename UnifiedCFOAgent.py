from openai import OpenAI
from langchain.agents import Tool, AgentExecutor, OpenAIToolsAgent
from retrieval_tools import build_retrieval_tools

def create_cfo_agent():
    client = OpenAI()
    tools = build_retrieval_tools()

    tool_definitions = [
        Tool(
            name="retrieve_text",
            func=tools["retrieve_text"],
            description="Use this to retrieve textual evidence from annual reports or 10-Ks."
        ),
        Tool(
            name="retrieve_table",
            func=tools["retrieve_table"],
            description="Use this to retrieve financial tables like Opex, Gross Margin, Revenue."
        ),
        Tool(
            name="retrieve_image",
            func=tools["retrieve_image"],
            description="Use this to retrieve slide OCR text from presentations."
        )
    ]

    agent = OpenAIToolsAgent.from_llm_and_tools(
        llm=client.chat.completions,
        tools=tool_definitions,
        system_prompt="""
        You are the CFO AI Agent. 
        Always plan first, then call retrieval tools only when needed.
        Prefer table retrieval for numerical data.
        Prefer text retrieval for descriptive content.
        Prefer image retrieval when slides contain relevant captions.
        Cite all sources accurately.
        """
    )

    return AgentExecutor(agent=agent, tools=tool_definitions)
