
import os
try:
    from  config  import DATA_DIR
except ImportError: 
    from TextRetrieval.config import DATA_DIR 

try: 
    from AgentHelpers import choose_sections_for_query, expand_query_for_retrieval, determine_k
except ImportError: 
    from TextRetrieval.AgentHelpers import choose_sections_for_query, expand_query_for_retrieval , determine_k


try:
    from IndexSearch import init_indexes , search_query
except ImportError: 
    from TextRetrieval.IndexSearch import init_indexes, search_query 


from langchain_openai import ChatOpenAI 

class TextSectionRetriever:
    """
    Loads all per-section FAISS indices built by TextFaissBuilder
    and provides a simple .search(query, k) API.
    This is for the Unified CFO Agent to retrieve text snippets. 
    """

    def __init__(self, sections_root: str | None = None):
        if sections_root is None:
            sections_root = os.path.join(DATA_DIR, "sections")

        self.sections_root = sections_root
        self.section_indices = {}  # sec -> (faiss_index, chunks_list)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4.1")

        if not os.path.isdir(self.sections_root):
            raise FileNotFoundError(
                f"Text sections directory not found: {self.sections_root}. "
                f"Did you run create_chunks() and built_indices()?"
            )
        self.section_indices = init_indexes() 

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

        available_sections = list(self.section_indices.keys())
        expanded = expand_query_for_retrieval(query=query, llm=self.llm) 
        chosen_sections = choose_sections_for_query(
            llm=self.llm, query=expanded, available_sections=available_sections) 
        print(f"[TextSectionRetriever] Searching sections: {chosen_sections}") 

        k = determine_k(query=query) 
        print (f"[TextSectionRetriever] Using k={k} for query: '{query}'") 

        results = search_query(
            expanded_query=expanded,
            sections=chosen_sections,
            k=k,
            query=query 
        )

        return results 
    