from sentence_transformers import SentenceTransformer 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2") 

# s = "help" 
# print(model.encode(s))

texts = [
    "I love programming",
    "coding is fun",
    "I hate bugs",
    "the sky is blue"
]

emb = model.encode(texts)
# print(emb.shape)
# sin = cosine_similarity(emb[0],emb) 
# print(sin)

# SEARCHING... 
query = "I am not good solving the errors." 
query_emb = model.encode(query) 

scores = cosine_similarity([query_emb],emb)[0] 

best_index = scores.argmax() 
print(texts[best_index])
print(best_index)