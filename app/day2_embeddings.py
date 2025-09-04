from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model (small + CPU friendly)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example docs
docs = ["AI is transforming QA automation.",
        "MLOps helps scale machine learning.",
        "RAG combines search with LLMs."]

# Compute embeddings
embs = model.encode(docs).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embs.shape[1])
index.add(embs)

# Search with a query
query = "How do I scale ML pipelines?"
q_emb = model.encode([query]).astype("float32")

D, I = index.search(q_emb, k=2)
print("Query:", query)
for idx in I[0]:
    print("Nearest doc:", docs[idx])
