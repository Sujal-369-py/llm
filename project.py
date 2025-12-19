from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity
import json 
import numpy as np 

with open("web scrap/quotes.json","r",encoding="utf-8") as f: 
    data = json.load(f) 


#Generate embedding.
model = SentenceTransformer("all-MiniLM-L6-v2")
def generate_embedding(text): 
    return model.encode(text).tolist()

#add column and do embedding
# for row in data: 
#     row["embedding"] = generate_embedding(row["quote"]) 

# with open("web scrap/quotes.json","w",encoding="utf-8") as f: 
#     json.dump(data,f,indent=4)


user_query = input("Enter your quote : ") 
 

#collect all embedding 
embedding = np.array([row["embedding"] for row in data])
emb_query = np.array(generate_embedding(user_query)).reshape(1,-1)

scores = cosine_similarity(emb_query,embedding)[0] 

top_index = scores.argsort()[::-1][:3] 

for i in top_index: 
    print(data[i]["quote"])
    print("-->",data[i]["author"]) 
    print()


