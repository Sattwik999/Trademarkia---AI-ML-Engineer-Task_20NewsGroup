"""
One-time setup script:
  1. Embeds all documents and populates the ChromaDB collection.
  2. Trains and saves UMAP + GMM models locally.

Usage:
    python setup_db.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.mixture import GaussianMixture

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH       = "data/clean_newsgroups_dataset.csv"
CHROMA_PATH     = "chroma_db"
COLLECTION_NAME = "newsgroups_docs"
MODEL_DIR       = "models"
TEXT_COL        = "processed_text"
N_CLUSTERS      = 20
UMAP_COMPONENTS = 10
RANDOM_STATE    = 42
BATCH_SIZE      = 256
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
df[TEXT_COL] = df[TEXT_COL].fillna("")
texts = df[TEXT_COL].tolist()
print(f"  {len(texts)} documents loaded.")

print("Embedding documents …")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = encoder.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
embeddings = np.array(embeddings, dtype=np.float32)

# ── Populate ChromaDB ────────────────────────────────────────────────────────
print("Populating ChromaDB collection …")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Get or create — preserves the UUID so running servers don't get stale references
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# Clear existing documents before re-ingesting
existing_ids = collection.get(include=[])['ids']
if existing_ids:
    for i in range(0, len(existing_ids), chunk):
        collection.delete(ids=existing_ids[i:i + chunk])
    print(f"  Cleared {len(existing_ids)} existing documents.")

chunk = 5000
for start in range(0, len(texts), chunk):
    end = min(start + chunk, len(texts))
    collection.add(
        ids=[str(i) for i in range(start, end)],
        documents=texts[start:end],
        embeddings=embeddings[start:end].tolist(),
        metadatas=[{"category": df["category"].iloc[i]} for i in range(start, end)],
    )
    print(f"  Added {end}/{len(texts)} documents …")

print(f"  Collection count: {collection.count()}")

# ── Train UMAP ───────────────────────────────────────────────────────────────
print("Fitting UMAP …")
umap_model = UMAP(
    n_components=UMAP_COMPONENTS,
    random_state=RANDOM_STATE,
    n_neighbors=15,
    min_dist=0.1,
)
reduced = umap_model.fit_transform(embeddings)

# ── Train GMM ────────────────────────────────────────────────────────────────
print("Fitting GMM …")
gmm_model = GaussianMixture(
    n_components=N_CLUSTERS,
    covariance_type="full",
    random_state=RANDOM_STATE,
    max_iter=200,
)
gmm_model.fit(reduced)

# ── Save models ───────────────────────────────────────────────────────────────
joblib.dump(umap_model, os.path.join(MODEL_DIR, "umap_model.pkl"))
joblib.dump(gmm_model,  os.path.join(MODEL_DIR, "gmm_model.pkl"))
print("Models saved to models/umap_model.pkl and models/gmm_model.pkl")

print("\nSetup complete. You can now run: uvicorn app.main:app --reload")
