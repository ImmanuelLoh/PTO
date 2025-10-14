import json

import  numpy as np 

# # read the npy file 

# embeddings = np.load("Google/sections/cover_page/embeddings_cover_page.npy")  # shape (num_chunks, embedding_dim) 

# print (embeddings.shape)  # e.g. (100, 1536) 


data = json.load(open("Google/sections/cover_page/chunks_cover_page.json"))

print (len(data))  # should match num_chunks in embeddings 