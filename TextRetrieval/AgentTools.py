
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.chains import LLMMathChain

from IndexSearch import init_indexes, search_query 
from AgentHelpers import safe_json_loads, choose_sections_for_query, expand_query_for_retrieval 
from logger import format_context 


@tool("retriever", return_direct=False)
def retriever_tool(query : str ) -> str: 
    """
    Retrieve relevant text snippets from financial filings.

    Accepts:
    - A plain query string (e.g., "Show total operating expenses for 2023").
    - Or a JSON string with fields:
        query: financial question
        sections: list of section names such as ["income_statement", "balance_sheet"]

    If no sections are provided, the tool will automatically choose the most relevant ones.
    """

    available_sections = init_indexes().keys() 

    print (f"[RETRIEVER_TOOL] agent provided query: {query}")
    print (f"[RETRIEVER_TOOL] query type: {type(query)}") 
    

    parsed = safe_json_loads(query)

    # 1️⃣ Try to parse JSON input if agent passes structured info
    if parsed:
        print(f"[INFO] Parsed agent input: {parsed}")
        query = parsed.get("query", query)
        sections = parsed.get("sections", None)
        print(f"[INFO] Agent provided query: {query}")
        print(f"[INFO] Agent provided sections: {sections}")
    else:
        print("[INFO] No structured sections provided, will auto-select.")
        sections = None
    
    # create an llm instance 
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0) 

    # 1️⃣ Expand query internally
    expanded_query = expand_query_for_retrieval(query, llm)

    # 2️⃣ If agent didn't pass sections, auto-select
    if not sections:
        sections = choose_sections_for_query(
            llm, expanded_query, available_sections)
        print (f"[INFO] Auto selected sections: {sections}") 
        if not sections:
            sections = available_sections  # fallback to all sections 

    # 3️⃣ Perform retrieval
    results = search_query(
        expanded_query=expanded_query, 
        sections=sections, 
        k=10,
        query=query 
    ) 
    return format_context(results) 

@tool("available_sections_retriever", return_direct=False) 
def available_sections_retriever(query: str = "") -> list:
    """
    List all available sections in the indexed financial filings.

    Returns a JSON array of section names, e.g.:
    ["income_statement", "balance_sheet", "cash_flow", "mdna", "risk_factors"].

    Useful for understanding which sections can be queried via the retriever tool.
    """

    return list(init_indexes().keys()) 

@tool("calculator", return_direct=False)
def calculator_tool(expression: str) -> str:
    """Safely evaluate a mathematical expression, e.g. (165 - 150) / 150 * 100."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error evaluating: {e}"
