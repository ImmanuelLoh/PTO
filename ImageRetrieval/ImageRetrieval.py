import faiss
import json
import numpy as np
from openai import OpenAI


class ImageRetriever:
    def __init__(self, index_path, metadata_json):
        self.index = faiss.read_index(index_path)
        self.metadata = json.load(open(metadata_json, "r"))
        self.client = OpenAI()

    def embed_query(self, query, model="text-embedding-3-small"):
        resp = self.client.embeddings.create(model=model, input=[query])
        return np.array(resp.data[0].embedding).astype("float32")

    def search(self, query, k=5):
        q_emb = self.embed_query(query)
        q_emb = np.expand_dims(q_emb, axis=0)

        distances, indices = self.index.search(q_emb, k)
        results = []

        for idx, dist in zip(indices[0], distances[0]):
            meta = self.metadata[str(idx)]
            results.append({"slide_id": idx, "distance": float(dist), "metadata": meta})

        return results
