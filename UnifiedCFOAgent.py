from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import (
    initialize_agent,
    AgentType,
    Tool,
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage


import numpy as np
import json
from TextRetrieval.Embedding import embed_text_query
from CacheOptimization.query_cache import search_semantic_cache, store_in_semantic_cache
from retrieval_tools import build_retrieval_tools
import asyncio
import nest_asyncio

nest_asyncio.apply()

# SYSTEM_PROMPT = """
# You are the CFO AI Agent.

# Your responsibilities:
# - ALWAYS RUN retrieve_text for every single question to get context.
# - Retrieve evidence using the correct retrieval tool:
#     • retrieve_table → structured financial tables (Opex, Revenue, Margins)
#     • retrieve_text → narrative filings, footnotes, MD&A
#     • retrieve_image → OCR text from presentation slides
# - Think step-by-step BEFORE calling a tool.
# - For numerical computations, compute explicitly.
# - ALWAYS cite the sources from tool outputs.
# - After using tools, summarize the results into a final CFO-level answer.

# You may prune your plan:
# - Skip irrelevant retrievals.
# - Stop early if enough evidence has been found.
# - Choose only the most relevant modality.


# Return only the FINAL ANSWER as output.
# """

SYSTEM_PROMPT = """ 
You are the CFO AI Agent.

Your workflow always follows three phases:

PHASE 1 — PLANNING  
• Begin every query by generating a short internal plan.  
• Consider ALL THREE retrieval tools in the initial plan:
      - retrieve_table  → financial statements and structured numbers
      - retrieve_text   → MD&A, footnotes, narrative explanations
      - retrieve_image  → OCR of slides, charts, and investor materials
• Decide what each tool *might* contribute.

PHASE 2 — PLAN PRUNING  
• After forming the plan, prune it by removing retrievals that are clearly irrelevant.  
• You may prune ONLY when you can justify (in your reasoning, NOT in the final answer) that a tool cannot contribute meaningful evidence.  
• When uncertain, keep the tool in the plan.  
• Pruning must be deliberate, not automatic or skipped.

PHASE 3 — EXECUTION  
• Run every retrieval tool that remains after pruning.
• If "parallel_context" is provided, you may use it instead of calling retrieval tools again.
• Cite all sources from tool outputs.  
• Use retrieved values only—never hallucinate.  
• Perform explicit numerical computations (YoY, margins, deltas).  
• Merge the evidence into a concise CFO-level final answer.

Additional rules:
• Always try to retrieve tables unless it is 100% narrative-only.  
• Always try to retrieve text unless the query is purely numeric.  
• Always try to retrieve images unless slides are guaranteed irrelevant.  
• Final output is ONLY the CFO-level answer.
"""


# Asyncio runner
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop → normal python
        return asyncio.run(coro)

    # Running inside Jupyter/IPython (loop already running)
    task = asyncio.ensure_future(coro)
    loop.run_until_complete(task)
    return task.result()


# Async wrappers (for parallel execution)
async def async_retrieve_text(retrieve_func, query):
    return await asyncio.to_thread(retrieve_func, query)


async def async_retrieve_table(retrieve_func, query):
    return await asyncio.to_thread(retrieve_func, query)


async def async_retrieve_image(retrieve_func, query):
    return await asyncio.to_thread(retrieve_func, query)


# Parallel runner
async def run_parallel_retrievals(retrieval, query):
    print("\n[PARALLEL] Starting parallel retrieval...")
    tasks = [
        async_retrieve_text(retrieval["retrieve_text"], query),
        async_retrieve_table(retrieval["retrieve_table"], query),
        async_retrieve_image(retrieval["retrieve_image"], query),
    ]
    results = await asyncio.gather(*tasks)
    print("[PARALLEL] Completed parallel retrieval.\n")

    return {
        "parallel_text": results[0],
        "parallel_table": results[1],
        "parallel_image": results[2],
    }


class SourceCapturingCallback(BaseCallbackHandler):
    """Callback to capture tool outputs and extract sources"""

    def __init__(self):
        self.sources = []
        self.image_paths = []
        self.tool_outputs = []

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes execution"""
        self.tool_outputs.append(output)

        # Try to parse as JSON and extract sources
        try:
            if isinstance(output, str):
                output_dict = json.loads(output)
                if isinstance(output_dict, dict):
                    if "sources" in output_dict:
                        self.sources.extend(output_dict["sources"])
                    if "image_paths" in output_dict:  # ADD THIS
                        self.image_paths.extend(output_dict["image_paths"])
        except json.JSONDecodeError:
            # Not JSON, skip
            pass
        except Exception as e:
            print(f"[CALLBACK] Error extracting sources: {e}")


def create_cfo_agent(parallel_context=None):
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
        model="gpt-4.1-mini",
        temperature=0,
    )

    system_messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Inject parallel context as system message
    if parallel_context is not None:
        system_messages.append(
            SystemMessage(content=f"PARALLEL_CONTEXT:\n{json.dumps(parallel_context)}")
        )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": system_messages},
    )

    return agent


def query_cfo_agent(
    question: str, cache_threshold: float = 0.85, use_cache: bool = True
):
    """
    Main entry point for querying the CFO agent with semantic caching.

    Args:
        question (str): The user's question to the CFO agent.
        cache_threshold (float): Similarity threshold for using cached responses.
        use_cache (bool): Whether to use semantic caching.
    """
    # Check cache first
    if use_cache:
        query_embedded = np.array(embed_text_query(question), dtype="float32")
        cache_result = search_semantic_cache(query_embedded, threshold=cache_threshold)
        if cache_result["hit"]:
            print("Cache HIT:")
            print(f"  Similarity: {cache_result['similarity']:.4f}")
            print(f"  Cached Query: {cache_result['cache_query']}")
            print()
            return {
                "query": question,
                "answer": cache_result["response"],
                "metadata": cache_result["metadata"],
                "cache_hit": True,
            }
        else:
            print("Cache MISS")
            print()

    # Create callback to capture sources
    source_callback = SourceCapturingCallback()

    # If no cache hit, query the agent

    ######################## Non-parallel version ########################################
    # agent = create_cfo_agent()
    # response = agent.invoke(
    #     {"input": question},
    #     config={"callbacks": [source_callback]}
    # )

    # Run parallel retrievals invoking the agent
    retrieval = build_retrieval_tools()
    # parallel_context = asyncio.run(run_parallel_retrievals(retrieval, question))
    parallel_context = run_async(run_parallel_retrievals(retrieval, question))

    # Build agent with parallel context embedded in system prompt
    agent = create_cfo_agent(parallel_context=parallel_context)

    # Only pass the actual question to the agent
    response = agent.invoke(
        {"input": question}, config={"callbacks": [source_callback]}
    )

    # Extract final answer and sources
    answer = response["output"]
    sources = list(set(source_callback.sources))
    image_paths = list(set(source_callback.image_paths))

    metadata_sources = {"sources": sources, "image_paths": image_paths}

    # Store in cache
    if use_cache:
        query_embedded = np.array(embed_text_query(question), dtype="float32")

        store_in_semantic_cache(
            query=question,
            embedding=query_embedded,
            results=answer,
            metadata=metadata_sources,
        )
    return {
        "query": question,
        "answer": answer,
        "metadata": metadata_sources,
        "cache_hit": False,
    }
