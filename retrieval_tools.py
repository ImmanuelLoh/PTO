import os
import json
import faiss
import numpy as np

from TextRetrieval.Embedding import embed_text_query
from TextRetrieval.config import DATA_DIR
from TableRetrieval.table_agentic_rag import TableAgenticRAG
from ImageRetrieval.ImageRetrieval import ImageRetriever

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

class TextSectionRetriever:
    """
    Loads all per-section FAISS indices built by TextFaissBuilder
    and provides a simple .search(query, k) API.
    """

    def __init__(self, sections_root: str | None = None):
        if sections_root is None:
            sections_root = os.path.join(DATA_DIR, "sections")

        self.sections_root = sections_root
        self.section_indices = {}  # sec -> (faiss_index, chunks_list)

        if not os.path.isdir(self.sections_root):
            raise FileNotFoundError(
                f"Text sections directory not found: {self.sections_root}. "
                f"Did you run create_chunks() and built_indices()?"
            )

        for sec in os.listdir(self.sections_root):
            sec_dir = os.path.join(self.sections_root, sec)
            if not os.path.isdir(sec_dir):
                continue

            idx_path = os.path.join(sec_dir, f"faiss_index_{sec}.idx")
            chunk_path = os.path.join(sec_dir, f"chunk_{sec}.json")

            if not (os.path.isfile(idx_path) and os.path.isfile(chunk_path)):
                continue

            index = faiss.read_index(idx_path)
            with open(chunk_path, "r") as f:
                chunks = json.load(f)

            self.section_indices[sec] = (index, chunks)

        if not self.section_indices:
            raise RuntimeError(
                f"No section indices loaded from {self.sections_root}. "
                f"Check that built_indices() finished successfully."
            )

        print(
            f"[TextSectionRetriever] Loaded {len(self.section_indices)} section indices "
            f"from {self.sections_root}"
        )

    def search(self, query: str, k: int = 5):
        """
        Search across all sections and return the top-k closest chunks.
        """
        # embed query
        q_emb_list = embed_text_query(query)  # list of floats
        q_emb = np.array(q_emb_list, dtype="float32")[None, :]  # shape (1, dim)

        candidates = []

        # search each section index
        for sec, (index, chunks) in self.section_indices.items():
            # ask for up to k from each section, then merge
            distances, indices = index.search(q_emb, k)
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(chunks):
                    continue
                chunk = chunks[idx]
                candidates.append(
                    {
                        "section": sec,
                        "distance": float(dist),
                        "text": chunk.get("text", ""),
                        "metadata": chunk.get("metadata", {}),
                    }
                )

        # sort by distance and keep global top-k
        candidates.sort(key=lambda x: x["distance"])
        return candidates[:k]


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
    def retrieve_text(query: str):
        # return text_retriever.search(query, k=5)

        result = text_retriever.search(query, k=5)
        sources = []
        for item in result:
            source = item["metadata"].get("document", "unknown")
            if source not in sources:
                sources.append(source)
        output = {
            "results": result,
            "sources": list(set(sources))
        }
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
            "image_paths": image_paths
        }
    
        # Return as JSON string so LangChain preserves it
        return json.dumps(output)

    return {
        "retrieve_text": retrieve_text,
        "retrieve_table": retrieve_table,
        "retrieve_image": retrieve_image,
    }
