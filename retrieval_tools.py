from TextRetrieval.TextFaissBuilder import text_faiss_retriever
from TableRetrieval.table_agentic_rag import TableAgenticRAG
from ImageRetrieval.ImageRetrieval import ImageRetriever


def build_retrieval_tools():
    # Initialise all 3 retrievers
    text_retriever = text_faiss_retriever()
    table_retriever = TableAgenticRAG(
        faiss_index_path="00-data/base/faiss_table_index",
        metadata_json_path="00-data/base/faiss_table_metadata.json",
    )
    image_retriever = ImageRetriever(
        "00-data/base/faiss_image_index", "00-data/base/faiss_image_metadata.json"
    )

    # Wrap tools for agent
    def retrieve_text(query: str):
        return text_retriever.search(query, k=5)

    def retrieve_table(query: str):
        return table_retriever.query(query, verbose=False)

    def retrieve_image(query: str):
        return image_retriever.search(query, k=5)

    return {
        "retrieve_text": retrieve_text,
        "retrieve_table": retrieve_table,
        "retrieve_image": retrieve_image,
    }
