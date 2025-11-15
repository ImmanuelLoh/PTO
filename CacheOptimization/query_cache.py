import time
from typing import Any, Dict, List
import numpy as np

SEMANTIC_CACHE: List[Dict[str, Any]] = []

def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Computes cosine similarity between two embeddings.
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
def store_in_semantic_cache(query, embedding, results, metadata):
    """
    Stores the query, its embedding, and associated results in the semantic cache.
    """
    start_time = time.time()
    
    SEMANTIC_CACHE.append({
        'query': query,
        'embedding': embedding,
        'response': results,
        'metadata': metadata,
        'timestamp': time.time()
    })
    
    end_time = time.time()
    print(f"Cache storage completed in {end_time - start_time:.4f} seconds.")

def search_semantic_cache(query_embedding, threshold=0.8):
    """
    Searches the semantic cache for relevant entries based on the query embedding.
    Returns cached results if similarity exceeds the threshold.
    """
    best_score = -1
    best_result = None

    for entry in SEMANTIC_CACHE:
        cached_embedding = entry['embedding']
        similarity = compute_similarity(query_embedding, cached_embedding)

        if similarity > best_score:
            best_score = similarity
            best_result = entry

    if best_score >= threshold:
        return {
            "hit": True,
            "similarity": best_score,
            "response": best_result['response'], 
            "metadata": best_result['metadata'], 
            "cache_query": best_result["query"]

        }
    return {"hit": False}
    