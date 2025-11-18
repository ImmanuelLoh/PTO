import json
import faiss
import numpy as np
from openai import OpenAI

def create_image_embeddings(docs, model="text-embedding-3-small"):
    """
    Embed each slide document.
    """
    client = OpenAI()
    batch_size = 50

    contents = [d.page_content for d in docs]
    embeddings = []

    for i in range(0, len(contents), batch_size):
        batch = contents[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        embeddings.extend([e.embedding for e in resp.data])

    return np.array(embeddings).astype("float32")


def store_image_faiss(embeddings, index_path):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"[FAISS] Saved index → {index_path}")
    return index


def save_image_metadata(docs, output_json):
    mapping = {}

    for idx, doc in enumerate(docs):
        mapping[idx] = doc.metadata

    with open(output_json, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"[FAISS] Saved metadata → {output_json}")
