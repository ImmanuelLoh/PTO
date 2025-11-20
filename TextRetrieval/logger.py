import asyncio
import os 
import json 
try:
    from config import DATA_DIR 
except ImportError: 
    from TextRetrieval.config import DATA_DIR 

try: 
    from AgentHelpers import extract_financial_values_async
except ImportError: 
    from TextRetrieval.AgentHelpers import  extract_financial_values_async 


def generate_test_log_path_name(base_path: str): 
    # create the directory if not exist 
    os.makedirs(base_path, exist_ok=True) 
    existing_files = [f for f in os.listdir(base_path) if f.startswith("test_") and f.endswith(".json")] 
    existing_indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()] 
    next_index = max(existing_indices) + 1 if existing_indices else 1 

    return f"{base_path}/test_{next_index}.json"


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

    
def append_json_entry(path, entry):
    # If file doesn't exist, create it with an empty list
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([entry], f, indent=2)
        return

    # If file exists, load → append → save
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    data.append(entry)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


async def format_context_async(results):
    output = {
        "raw_context": [],
        "extracted_values": [],
    }

    tasks = []

    for section_data in results:
        print(f"[INFO] Formatting section: {section_data.get('section','unknown')}")

        for r in section_data.get("ranking", []):
            meta = r["metadata"]
            text = r["text"].strip()

            doc = meta.get("document", "unknown")
            page = meta.get("page_number", "?")
            chunk_idx = meta.get("chunk_index", None)
            section = meta.get("page_section", "")

            source = f"{doc}, page {page} chunk {chunk_idx} {section}".strip()

            # Store raw text
            output["raw_context"].append({
                "text": text,
                "source": source
            })

            print(f"[DEBUG] scheduling extraction for: {source}")
            tasks.append(extract_financial_values_async(text, source))

    # Run all scheduled extractions concurrently
    results_list = await asyncio.gather(*tasks, return_exceptions=False)

    # Collect extracted rows
    for rows in results_list:
        output["extracted_values"].extend(rows)

    # Collect unique sources used
    # unique_sources = {entry["source"] for entry in output["extracted_values"]} 
    # output["sources_used"] = list(unique_sources)


    append_json_entry(f"{DATA_DIR}/logs/text_context_debug.json", output)

    #return output 
    return json.dumps(output, indent=2)
