# import tiktoken 

# enc = tiktoken.get_encoding("cl100k_base")

# text = "hello man" 
# token = enc.encode(text)
# print(len(token))  

# txt = [890] 
# res = enc.decode(txt) 
# print(res)


########################################################################
#          MINI PROJECT          # 
import tiktoken 
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2") 

def token_analyzer(text): 
    enc = tiktoken.get_encoding("cl100k_base")
    res = enc.encode(text) 
    print("Token: ") 
    for token in res: 
        print(f"{token} -> {enc.decode([token])}") 
    print(f"Total: {len(res)} tokens.")

def create_embedding(text): 
    emb = model.encode(text) 
    return emb

def vector_search(que_emb,doc_emb): 
    sims = cosine_similarity([que_emb],doc_emb)[0] 
    return sims

def main():

    with open("test.txt", "r", encoding="utf-8") as f:
        data = f.read()

    chunks = data.split("\n\n")
    doc_emb = model.encode(chunks)

    query = "what is search for spritulity?"
    q_emb = model.encode(query)

    scores = vector_search(q_emb, doc_emb)
    best = scores.argmax()

    print("Best chunk:")
    print(chunks[best])

main()
