import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load Artifacts (Executes once when this file is imported)
print("   [Tool] Loading RAG Knowledge Base...")
try:
    # Load Index
    index = faiss.read_index("airline_db.index")
    
    # Load Text Chunks
    with open("airline_texts.pkl", "rb") as f:
        feature_texts = pickle.load(f)
        
    # Load Model (Must be same as vector_embedding.py)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("   [Tool] Knowledge Base Loaded.")
    
except Exception as e:
    print(f"CRITICAL ERROR: Could not load RAG artifacts. {e}")
    # Create empty placeholders to prevent crash on import, 
    # but search will fail if called.
    index = None
    feature_texts = []
    embedder = None

def search_knowledge_base(query: str, k: int = 3):
    """
    Embeds the query and searches the FAISS index.
    Returns a list of the top k text chunks.
    """
    if not index or not embedder:
        return ["Error: Database not loaded."]

    # 1. Embed Query
    query_vec = embedder.encode([query], convert_to_numpy=True)
    
    # 2. Search FAISS
    distances, indices = index.search(query_vec, k)
    
    # 3. Retrieve Documents
    results = []
    for idx in indices[0]:
        if idx < len(feature_texts):
            results.append(feature_texts[idx])
            
    return results