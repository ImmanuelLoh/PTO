from langchain.agents import initialize_agent, AgentType
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage


try:
    from AgentTools import retriever_tool, calculator_tool, available_sections_retriever 
except ImportError: 
    from TextRetrieval.AgentTools import retriever_tool, calculator_tool, available_sections_retriever 

def text_agent_executor(query: str = ""): 
    """
    Initialize and return a text-based agent executor for financial analysis.
    The agent uses a language model and various tools to process and analyze financial data.
    """ 
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0) 
    system_prompt = """
You are a financial-analysis agent.

You may receive different types of tool outputs depending on which tool was used.

When the retriever tool is used, you will receive a JSON object
{
  raw_context: [{ text, source }],
  extracted_values: [{ component, year, value, source }]
}

STRICT RULES FOR STRUCTURED DATA (applies ONLY when extracted_values is present):
1. NEVER extract numbers from raw_context.
2. NEVER infer, approximate, or guess missing values.
3. ONLY use numeric values provided in extracted_values.
4. ONLY perform calculations using calculator_tool.
5. For multi-year or multi-component answers, ALWAYS produce tables.
6. For every number used, ALWAYS cite its `source`.
7. If a required number is missing from extracted_values, state exactly what is missing and DO NOT compute partial results.
8. Use raw_context ONLY for quoting text or explaining wording — NEVER for numeric extraction.
9. At the end of your answer, output:
SOURCES: [list of unique sources here]

At the end of your answer, ALWAYS output:
SOURCES: {{sources_used}}

"""

    # 4️⃣ tools
    tools = [retriever_tool, calculator_tool, available_sections_retriever]

    # 5️⃣ initialize agent (already returns an AgentExecutor)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs={"system_message": SystemMessage(content=system_prompt)}
    )

    agent.max_iterations = None 

    # 6️⃣ run agent 
    if query == "": 
        query = "Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values."

    response = agent.invoke({"input": query,"query": query }, handle_parsing_errors=True) 

   
    print ("Final response from agent:") 
    print (response["output"]) 


if __name__ == "__main__": 
    # text_agent_executor("" \
    # "What is the operating expense for the last 3 fiscal years, year-on-year comparison.") 

    text_agent_executor("what is the Gross Margin for the last 5 quarters, with values.") 
