from langchain.agents import (
    initialize_agent,
    AgentType,
    Tool,
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from retrieval_tools import build_retrieval_tools


SYSTEM_PROMPT = """
You are the CFO AI Agent.

Your responsibilities:
- Retrieve evidence using the correct retrieval tool:
    • retrieve_table → structured financial tables (Opex, Revenue, Margins)
    • retrieve_text → narrative filings, footnotes, MD&A
    • retrieve_image → OCR text from presentation slides
- Think step-by-step BEFORE calling a tool.
- For numerical computations, compute explicitly.
- ALWAYS cite the sources from tool outputs.
- After using tools, summarize the results into a final CFO-level answer.

You may prune your plan:
- Skip irrelevant retrievals.
- Stop early if enough evidence has been found.
- Choose only the most relevant modality.


Return only the FINAL ANSWER as output.
"""


def create_cfo_agent():
    # Load retrieval functions
    retrieval = build_retrieval_tools()

    tools = [
        Tool(
            name="retrieve_text",
            func=retrieval["retrieve_text"],
            description="Retrieve textual evidence from annual/quarterly filings.",
        ),
        Tool(
            name="retrieve_table",
            func=retrieval["retrieve_table"],
            description="Retrieve financial tables such as Opex, Revenue, Margins.",
        ),
        Tool(
            name="retrieve_image",
            func=retrieval["retrieve_image"],
            description="Retrieve OCR slide text from presentation decks.",
        ),
    ]

    # OpenAI LLM with function calling
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    # Create the OLD-STYLE AGENT (same as your previous pipeline)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": SystemMessage(content=SYSTEM_PROMPT)},
    )

    return agent
