from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage

from AgentTools import retriever_tool, calculator_tool, available_sections_retriever 

if __name__ == "__main__": 

    # 2️⃣ model
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # 3️⃣ system message
    system_prompt = """
    You are a financial analyst assistant that can use tools.

    When given retrieved data:
    1. Identify all relevant components (e.g., for Operating Expenses: R&D, Sales & Marketing, G&A)
    2. Extract the values for the requested years
    3. Calculate using standard financial formulas
    4. Cite each component source

    If data is incomplete, state what's missing.
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
    query = "Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values."

    response = agent.invoke({"input": query,"query": query }, handle_parsing_errors=True) 

    print ("Final response from agent:") 
    print (response["output"]) 
