# !pip install -q gensim faiss-cpu

from gensim.models import Word2Vec
import numpy as np
import faiss

sentences = [
    ["machine","learning","is","powerful"],
    ["deep","learning","uses","neural","networks"],
    ["machine","learning","uses","data"],
    ["artificial","intelligence","is","useful"],
    ["deep","learning","is","part","of","AI"]
]

model = Word2Vec(sentences,vector_size=50,window=3,
                 min_count=1,workers=1,seed=42)

words = list(model.wv.index_to_key)
vectors = np.array([model.wv[w] for w in words],dtype="float32")

index = faiss.IndexFlatL2(50)
index.add(vectors)

query = "learning"
q = np.array([model.wv[query]],dtype="float32")

distances,indices = index.search(q,4)

print("Query:",query)

for i in indices[0]:
    if words[i] != query:
        print(words[i])