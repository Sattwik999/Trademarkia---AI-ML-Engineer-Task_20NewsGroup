from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback
import chromadb

from app.embedding_utils import embed_query
from app.cluster_utils import get_cluster
from app.semantic_cache import SemanticCache


app = FastAPI(
    title="20 Newsgroups Semantic Search API",
    description="Semantic search over the 20 Newsgroups dataset with cluster-aware caching.",
    version="1.0.0",
)

cache = SemanticCache(threshold=0.9)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("newsgroups_docs")


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: float
    result: str
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class CacheClearResponse(BaseModel):
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    try:
        query_text = payload.query

        query_vector = embed_query(query_text)
        cluster_id, _ = get_cluster(query_vector)
        cached = cache.lookup(cluster_id, query_vector)

        if cached:
            return QueryResponse(
                query=query_text,
                cache_hit=True,
                matched_query=cached["matched_query"],
                similarity_score=round(float(cached["similarity_score"]), 4),
                result=cached["result"],
                dominant_cluster=cluster_id,
            )

        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=5,
        )
        docs = results["documents"][0]
        result_text = "\n\n---\n\n".join(docs)

        cache.store(cluster_id, query_text, query_vector, result_text)

        return QueryResponse(
            query=query_text,
            cache_hit=False,
            matched_query=None,
            similarity_score=0.0,
            result=result_text,
            dominant_cluster=cluster_id,
        )

    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats():
    return cache.stats()


@app.delete("/cache", response_model=CacheClearResponse)
def clear_cache():
    cache.clear()
    return CacheClearResponse(message="Cache cleared successfully.")