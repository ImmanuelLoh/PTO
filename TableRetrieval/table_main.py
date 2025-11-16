from table_agentic_rag import RetrievedChunk, TableAgenticRAG
from table_ingestion import stage1_extract_and_save, stage2_create_embeddings

DATA_DIR = "00-data"
EXTRACTED_JSON = f"{DATA_DIR}/extracted_tables.json"
FAISS_INDEX = f"{DATA_DIR}/faiss_table_index"
METADATA_JSON = f"{DATA_DIR}/faiss_table_metadata.json"

# =============================================================
# STEP 1 — Run ingestion (only run once per dataset)
# =============================================================
# Uncomment the first time you ingest PDFs
# print("Extracting tables...")
# stage1_extract_and_save(DATA_DIR, EXTRACTED_JSON)
# print("Embedding + FAISS...")
# stage2_create_embeddings(EXTRACTED_JSON, FAISS_INDEX, METADATA_JSON)

# =============================================================
# STEP 2 — Initialize FAISS Agent
# =============================================================
print("Loading FAISS-based Agentic CFO...")
agent = TableAgenticRAG(faiss_index_path=FAISS_INDEX, metadata_json_path=METADATA_JSON)

# =============================================================
# STEP 3 — Example Query
# =============================================================
# query = "Calculate the operating margin trend for the past 3 years."
# query = "Report the Gross Margin over the last 5 quarters, with values."
query = "Show Operating Expenses for the last 3 fiscal years, year-on-year comparison."


print("\nRunning agent query...\n")
result = agent.query(query, verbose=True)


print("\n====================== ANSWER ======================")
print(result["answer"])
print("===================================================\n")


print("Sources:", result["sources"])