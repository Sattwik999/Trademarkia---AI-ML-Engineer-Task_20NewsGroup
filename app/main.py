from fastapi import FastAPI
from fastapi.responses import JSONResponse
import traceback
import chromadb

from app.embedding_utils import embed_query
from app.cluster_utils import get_cluster
from app.semantic_cache import SemanticCache


app = FastAPI()

cache = SemanticCache(threshold=0.9)


client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection("newsgroups_docs")

@app.post("/query")
def query(payload: dict):
    try:
        query_text = payload["query"]

        query_vector = embed_query(query_text)

        cluster_id, probs = get_cluster(query_vector)

        cached = cache.lookup(cluster_id, query_vector)

        if cached:

            return {
                "query": query_text,
                "cache_hit": True,
                "matched_query": cached["matched_query"],
                "similarity_score": cached["similarity_score"],
                "result": cached["result"],
                "dominant_cluster": cluster_id
            }

        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=5
        )

        result_text = results["documents"][0]

        cache.store(cluster_id, query_text, query_vector, result_text)

        return {
            "query": query_text,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": 0,
            "result": result_text,
            "dominant_cluster": cluster_id
        }
    except Exception:
        return JSONResponse(status_code=500, content={"detail": traceback.format_exc()})
@app.get("/cache/stats")
def cache_stats():

    return cache.stats()

@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "cache cleared"}