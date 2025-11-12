from openai import OpenAI
import numpy as np 


client = OpenAI()

# choose your model
EMBED_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"

def embed_text_query(s):
    # E5 expects a prefix; same here for consistency
    text = f"query: {s.strip().lower()}"
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding

def embed_text_passage(chunks : list[str]) -> np.ndarray: 
    # Normalize and add E5-like prefix to all passages
    inputs = [f"{chunk.strip().lower()}" for chunk in chunks]
    response = client.embeddings.create(model=EMBED_MODEL, input=inputs)
    # convert it to numpy array
    embeddings = np.array([d.embedding for d in response.data])
    # return a list of numpy-like vectors
    return embeddings
