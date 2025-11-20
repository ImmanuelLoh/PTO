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
                    if 'sources' in output_dict:
                        self.sources.extend(output_dict['sources'])
                    if 'image_paths' in output_dict:  # ADD THIS
                        self.image_paths.extend(output_dict['image_paths'])
        except json.JSONDecodeError:
            # Not JSON, skip
            pass
        except Exception as e:
            print(f"[CALLBACK] Error extracting sources: {e}")

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

def query_cfo_agent(question:str, cache_threshold: float = 0.85, use_cache: bool = True):
    """
    Main entry point for querying the CFO agent with semantic caching.

    Args:
        question (str): The user's question to the CFO agent.
        cache_threshold (float): Similarity threshold for using cached responses.
        use_cache (bool): Whether to use semantic caching.
    """
    # Check cache first
    if use_cache:
        query_embedded = np.array(embed_text_query(question), dtype='float32')
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
                    "cache_hit": True
                }
        else:
            print("Cache MISS")
            print()

    # Create callback to capture sources
    source_callback = SourceCapturingCallback()

    # If no cache hit, query the agent
    agent = create_cfo_agent()
    response = agent.invoke(
        {"input": question},
        config={"callbacks": [source_callback]}
    )

    # Extract final answer and sources
    answer = response['output']
    sources = list(set(source_callback.sources))
    image_paths = list(set(source_callback.image_paths))

    metadata_sources = {
        "sources": sources,
        "image_paths": image_paths
    }

    # Store in cache
    if use_cache:
        query_embedded = np.array(embed_text_query(question), dtype='float32')

        store_in_semantic_cache(
            query=question,
            embedding=query_embedded,
            results=answer,
            metadata=metadata_sources
        )
    return {
        "query": question,
        "answer": answer,
        "metadata": metadata_sources,
        "cache_hit": False
    }
    
