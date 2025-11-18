
import json 
from collections import defaultdict

import numpy as np 
import faiss 
import os

try:
    from Embedding import embed_text_passage 
except ImportError: 
    from TextRetrieval.Embedding import embed_text_passage 

try:
    from config import DATA_DIR , CHUNK_SIZE , RETRIEVED_DATA_DIR
except ImportError: 
    from TextRetrieval.config import DATA_DIR , CHUNK_SIZE , RETRIEVED_DATA_DIR



"""
Where you chunk the text from the json file (from textExtractor) into smaller pieces for embedding and indexing.
Each chunk will have an ID, section label, and text content. 

Indices will be built per section using FAISS. 
"""

def create_chunks(): 
    '''
    Create text chunks from the documents in DATA_DIR/test.json , by chunk size (number of words). 
    chunks = [
        {"id": "doc1_page_1_chunk_0", "section": "Management's Discussion", "text": "...."},
        {"id": "doc1_page_1_chunk_1", "section": "Management's Discussion", "text": "...."},
        ...
        ]
    '''
    
    # load the json file
    with open(RETRIEVED_DATA_DIR, "r") as f:
        doc = json.load(f)

    # create chunks
    chunks : list[dict] = []

    for fileDoc, docContent in doc.items():
        for page_num, content in docContent.items():
            page_section = content.get("page_section", "unknown") 
            page_text = content.get("text", "") 
            words = page_text.split() 
            for i in range(0, len(words), CHUNK_SIZE):
                chunk_text = " ".join(words[i:i + CHUNK_SIZE]) 
                chunks.append({
                    "id": f"{fileDoc}_page_{page_num}_chunk_{i // CHUNK_SIZE}",
                    "text": chunk_text,
                    "metadata": {
                        "document": fileDoc,
                        "page_number": page_num, 
                        "chunk_index": i // CHUNK_SIZE, 
                        "page_section": page_section 
                        }
                })
    print (f"Created {len(chunks)} chunks from the documents.") 
    return chunks
 

def built_indices(chunks: list[dict]): 
    '''
    Build FAISS indices from the chunks. (per section)
    sections = {
    "section_name": [chunk1, chunk2, ...], 
    "section_name2": [chunk1, chunk2, ...], 
    ...
    }

    '''
    
    sections_path = f"{DATA_DIR}/sections" 
    sections = defaultdict(list) 

    # loop through the chunks and group by section 
    for c in chunks: 
        sec = c["metadata"].get("page_section", "unknown")
        sections[sec].append(c)

    # build FAISS index per section 
    for idx, (sec, sec_chunks) in enumerate(sections.items()): 
        print(f"Building index for section: {sec} with {len(sec_chunks)} chunks.")
        text: list[str] = [c["text"] for c in sec_chunks] 
        embeddings: np.ndarray = embed_text_passage(text)

        print (f"Embeddings shape for section {sec}: {embeddings.shape}") 
        dim = embeddings.shape[1] 
        index = faiss.IndexFlatL2(dim) 
        index.add(embeddings) 

        # make the dir 
        os.makedirs(f"{sections_path}/{sec}", exist_ok=True)

        # store the idx
        faiss.write_index(index, f"{sections_path}/{sec}/faiss_index_{sec}.idx")

        # store the mapping (chunk id to text) 
        with open(f"{sections_path}/{sec}/chunk_{sec}.json","w") as f:
            json.dump(sec_chunks, f, indent=4) 

        # store the embeddings 
        np.save(f"{sections_path}/{sec}/embeddings_{sec}.npy", embeddings) 

    print (f"Built {idx+1} FAISS indices for sections.")

if __name__ == "__main__": 
    chunks = create_chunks() 
    built_indices(chunks) 