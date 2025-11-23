import os
import json
import faiss
import numpy as np

from TextRetrieval.Embedding import embed_text_query
from TextRetrieval.config import DATA_DIR
from TableRetrieval.table_agentic_rag import TableAgenticRAG
from ImageRetrieval.ImageRetrieval import ImageRetriever
from TextRetrieval.TextRetrieval import TextSectionRetriever
from sentence_transformers import CrossEncoder

# Load cross-encoder model for re-ranking
# cross-encoder/ms-marco-MiniLM-L-6-v2 (fast)
# cross-encoder/ms-marco-MiniLM-L-12-v2 (better)
# BAAI/bge-reranker-base (very strong)
# BAAI/bge-reranker-large (slower, best accuracy)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def build_retrieval_tools():
    # Initialise all 3 retrievers

    # TEXT: load all per-section FAISS indices under 00-data/sections
    text_retriever = TextSectionRetriever()

    # TABLE
    table_retriever = TableAgenticRAG(
        faiss_index_path="00-data/base/faiss_table_index",
        metadata_json_path="00-data/base/faiss_table_metadata.json",
    )

    # IMAGE (slides)
    image_retriever = ImageRetriever(
        "00-data/base/faiss_image_index", "00-data/base/faiss_image_metadata.json"
    )

    # Wrap tools for agent
    ############### Text retrieval (without reranking) SWITCH BETWEEN THIS AND RERANKING FOR TESTING IF YOU WANT ###################################
    # def retrieve_text(query: str):
    #     result = text_retriever.search(query, k=5)
    #     sources = []
    #     for section_result in result:
    #         for item in section_result.get("ranking", []):
    #             source = item["metadata"].get("document", "unknown")
    #             if source not in sources:
    #                 sources.append(source)

    #     output = {
    #         "results": result,
    #         "sources": list(set(sources))
    #     }
    #     return json.dumps(output)

    # Text retrieval (with reranking)
    def retrieve_text(query: str):
        # Dynamic-K FAISS search
        initial_results = text_retriever.search(query, k=10)

        # Flatten all section results
        flattened = []
        for section in initial_results:
            for item in section.get("ranking", []):
                flattened.append(item)

        if len(flattened) == 0:
            return json.dumps({"results": [], "sources": []})

        # Build pairs for reranker
        pairs = [(query, item["text"]) for item in flattened]

        # Cross-Encoder scoring
        scores = reranker.predict(pairs)

        # Add scores back into objects
        for i, item in enumerate(flattened):
            item["rerank_score"] = float(scores[i])

        # Sort by rerank score
        flattened_sorted = sorted(
            flattened, key=lambda x: x["rerank_score"], reverse=True
        )

        # Top 5 after reranking
        top5 = flattened_sorted[:5]
        print("[RERANK] Top 5 reranked chunks selected.")
        for idx, chunk in enumerate(top5):
            print(
                f"   [TOP {idx+1}] score={chunk['rerank_score']}, doc={chunk['metadata'].get('document')}"
            )

        # Extract unique sources
        sources = list({doc["metadata"].get("document", "unknown") for doc in top5})

        # Return JSON-safe output
        output = {
            "results": convert_to_serializable(top5),
            "sources": sources,
        }
        
        print("[RETRIEVE_TEXT] Completed text retrieval with reranking.\n")
        return json.dumps(output)

    def retrieve_table(query: str):
        # return table_retriever.query(query, verbose=False)
        result = table_retriever.query(query, verbose=False)
        
        # Ensure result is a dict with sources
        if not isinstance(result, dict):
            result = {"answer": str(result), "sources": []}

        # Make sure sources key exists
        if "sources" not in result:
            result["sources"] = []

        print("[RETRIEVE_TABLE] Completed table retrieval.\n")
        # Return as JSON string so LangChain preserves it
        return json.dumps(result)

    def retrieve_image(query: str):
        # return image_retriever.search(query, k=5)
        # Format as dict with sources
        result = image_retriever.search(query, k=5)

        sources = []
        image_paths = []
        for item in result:
            metadata = item.get("metadata", {})
            source = metadata.get("source_report", "unknown")
            image_path = metadata.get("image_path", None)
            if source and source not in sources:
                sources.append(source)
            if image_path and image_path not in image_paths:
                image_paths.append(image_path)

        output = {
            "results": convert_to_serializable(result),
            "sources": sources,
            "image_paths": image_paths,
        }

        print(f"[RETRIEVE_IMAGE] Retrieved {len(result)} images.\n")
        # Return as JSON string so LangChain preserves it
        return json.dumps(output)

    return {
        "retrieve_text": retrieve_text,
        "retrieve_table": retrieve_table,
        "retrieve_image": retrieve_image,
    }
