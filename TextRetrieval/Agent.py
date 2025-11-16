from langchain.agents import initialize_agent, AgentType
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

When the retriever tool is used, you will receive a JSON object with the fields:
- raw_context: list of unstructured text excerpts
- extracted_values: list of structured financial entries with {component, year, value, source}

STRICT RULES FOR STRUCTURED DATA (applies ONLY when extracted_values is present):
1. NEVER extract numbers from raw_context.
2. NEVER infer, approximate, or guess missing values.
3. ONLY use numeric values provided in extracted_values.
4. ONLY perform calculations using calculator_tool.
5. ALWAYS produce tables for multi-year or multi-component results.
6. ALWAYS cite the source for every number using the source field.
7. If a required number is missing from extracted_values, state exactly what is missing and DO NOT compute partial results.
8. Use raw_context ONLY for quoting text or explaining wording — NEVER for numeric extraction.

WHEN STRUCTURED DATA IS NOT PRESENT (e.g., calculator_tool responses, other tools):
- Behave normally.
- Do NOT assume the structured schema is available.
- Do NOT attempt to enforce structured-data rules if extracted_values is not provided.

GENERAL:
- Identify which financial components and years the user is asking for.
- Use structured data when available.
- Be precise and follow all rules strictly when extracted_values exists.


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
    text_agent_executor("" \
    "What is the operating expense for the last 3 fiscal years, year-on-year comparison.") 
