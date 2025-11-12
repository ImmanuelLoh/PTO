import os 
import json 
from config import DATA_DIR 

def generate_test_log_path_name(base_path: str): 
    # create the directory if not exist 
    os.makedirs(base_path, exist_ok=True) 
    existing_files = [f for f in os.listdir(base_path) if f.startswith("test_") and f.endswith(".json")] 
    existing_indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()] 
    next_index = max(existing_indices) + 1 if existing_indices else 1 

    return f"{base_path}/test_{next_index}.json"


def format_context(results) -> str :
    """
    Format retrieval results (from documents or slides) into a readable text context.
    Automatically detects the source type from metadata.
    """
        
    parts = []
    for section_data in results:
        section = section_data.get("section", "unknown") 
        parts.append(f"## Section: {section}\n") 
        for r in section_data.get("ranking", []):
            meta = r["metadata"]
            text = r["text"].strip()

            # Detect the type of source (document vs slide)
            if "document" in meta:
                doc = meta.get("document", "unknown")
                page = meta.get("page_number", "?")
                section = meta.get("page_section", "")
                header = f"[{doc}, page {page}] {section}".strip()
            else:
                header = "[unknown source]"

            parts.append(f"{header}\n{text} ")

    return "\n\n---\n\n".join(parts)


def log_search_results(query: str, 
                       expanded_query: str, 
                       results: list[dict]): 
    # create dir if not exist 
    os.makedirs(f"{DATA_DIR}/logs/sections/", exist_ok=True) 
    # save it locally 
    dir = f"{DATA_DIR}/logs/sections/" 
    file_name = generate_test_log_path_name(f"{dir}") 

    # add the query then save the results as json 
    with open(file_name, "w") as f: 
        json.dump({
            "query": query, 
            "expanded_query": expanded_query, 
            "results": results 
        }, f, indent=4)  