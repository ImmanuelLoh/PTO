from config import DATA_DIR 
import os
import faiss 
import json 
from Embedding import embed_text_query 
import numpy as np 
from logger import log_search_results

"""
This is after you build the FAISS indices per section. 
you can now load them and search across them for a given query. 

"""

indices = {}  # global variable to hold loaded indices 


def init_indexes() -> dict[str, dict]:
    """
    indices = {
    "section_name": {
        "index": faiss.IndexFlatL2,
        "chunks": [
            {
                "id":id,
                "text": text, 
                "metadata": { 
                    "document": doc_name,
                    "page_number": page_num,
                    "page_section": section_name
                    } 
            }, 
            {chunk2}, ... 
            ],
    "section_name2": {
        "index": faiss.IndexFlatL2,
        "chunks": [{}, {}], ...
        }, 
    ... 
    }
    """ 
    global indices 
    if indices: 
        print ("[INFO] FAISS indices already initialized, reusing existing indices.") 
        return indices  # already loaded     
    print ("[INFO] Initializing FAISS indices from disk...") 
    documents_base_dir = f"{DATA_DIR}/sections" 
    sections_path = [os.path.join(documents_base_dir, f) for f in os.listdir(documents_base_dir) if os.path.isdir(os.path.join(documents_base_dir, f)) ]
    print ("Sections found:", sections_path) 

    for sec_path in sections_path: 
        section = os.path.basename(sec_path) 
        print (f"Loading section: {section}") 

        # load faiss 
        idx = faiss.read_index(f"{sec_path}/faiss_index_{section}.idx") 
        indices[section] = {
            "index": idx,
            "chunks": json.load(open(f"{sec_path}/chunk_{section}.json")) 
        }
    
    return indices 


def search_query( 
        expanded_query: str, 
        sections: list, 
        k:int =10, 
        query: str = "" 
        ) -> list[dict]:

    """
    Search across section-specific FAISS indexes for a given (already expanded) query.
    Result = [
         {
            "section": "section_name",
            "ranking": [
                {
                    "rank": 1,
                    "score": 0.1234,
                    "text": "matched text chunk ...",
                    "metadata": { ... }
                },
                ... ]
        }, ... ]
    """

    print(f"[INFO] Searching {len(sections)} sections for: '{expanded_query}'")
    indices = init_indexes()

    query_embedding = embed_text_query(expanded_query) 
    results = []


    for sec in sections: 
        if sec in indices: 
            idx = indices[sec]["index"]
            D, I = idx.search(np.array([query_embedding]), k=min(k, idx.ntotal))

            results.append(
                {
                    "section": sec, 
                    "ranking" : [
                        {
                            "rank": rank + 1 , 
                            "score": float(D[0][rank]), 
                            "text": indices[sec]["chunks"][identified_chunk_idx]["text"], 
                            "metadata": indices[sec]["chunks"][identified_chunk_idx]["metadata"] 
                        }   for rank, identified_chunk_idx in enumerate(I[0])           
                    ]
                }
            )
        else:
            print(f"[WARN] Section '{sec}' not found in indexes.") 

    print (f"[INFO] Completed search across sections.")
    print (f"results preview : {results[:1]} ... ")
    log_search_results( 
        query=query,
        expanded_query=expanded_query, 
        results=results
    ) 
    return results


if __name__ == "__main__": 
    indices = init_indexes() 
    print (f"Loaded indices for {len(indices)} sections: {list(indices.keys())}") 
    test_results = search_query("What is the company's revenue growth?", ["balance_sheet", "income_statement"], k=5, indices=indices) 
    for item in test_results: 
        print (f"Section: {item['section']}") 
        for r in item['ranking']: 
            print (f" Rank: {r['rank']}, Score: {r['score']}, Metadata: {r['metadata']}") 