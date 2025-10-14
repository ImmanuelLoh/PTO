import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


# load the json file 
with open("Data/jpmorgan10k-hybrid.json") as f:
    doc = json.load(f)

chunks = [] 

for page_num, content in doc.items(): 
    text = content["text"]
    tables = content.get("tables", []) 

    if text: 
        chunks.append({
            "id" : f"page-{page_num}-text", 
            "text": text,
            "metadata": { "page_number" : page_num, "chunk_type": "prose" }
        }) 

    if tables: 
        for t_index, table in enumerate(tables):
                table_text = "\n".join([ "\t".join(row) for row in table ])
                chunks.append({
                    "id" : f"page-{page_num}-table-{t_index}", 
                    "text": table_text,
                    "metadata": { "page_number" : page_num, "chunk_type": "table", "table_index": t_index }
                }) 



# LOAD intfloat/e5-base-v2

# load E5-base-v2
e5 = SentenceTransformer("intfloat/e5-base-v2")

text = [ chunk["text"] for chunk in chunks ] 

# embed the text chunks 
embeddings = e5.encode( text, 
                        convert_to_numpy=True, 
                        normalize_embeddings=True,
                        show_progress_bar=True
                        )

print (f"Embeddings shape: {embeddings.shape}")   

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print (f"FAISS index contains {index.ntotal} vectors.") 

# how many shares did jpmorg purchase ?
while True:
    user_input = input("Enter your query (or 'exit' to quit): ")

    if user_input.lower() == 'exit':
        break

    query = user_input
    query_embedding = e5.encode( query, 
                                convert_to_numpy=True, 
                                normalize_embeddings=True
                                ) 
    D , I = index.search( np.array([query_embedding]), k=5) 
    print (f"Distances: {D}") 
    print (f"Indices: {I}") 

    print ("Top 5 results:") 
    for i, idx in enumerate(I[0]):
        print ("-----------------------------------------------------") 
        print (f"Rank {i+1}, Score: {D[0][i]:.4f}")
        print (chunks[idx]["text"][:500]) 
        # print ("on page ", chunks[idx]["metadata"]["page_number"])
        # print ("Chunk type: ", chunks[idx]["metadata"]["chunk_type"])
        print ("metadata: ", chunks[idx]["metadata"]) 
        print ("-----------------------------------------------------") 