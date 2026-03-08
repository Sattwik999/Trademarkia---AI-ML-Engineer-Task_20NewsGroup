import requests
import json
import time

BASE = "http://127.0.0.1:8000"

def post_query(text):
    r = requests.post(f"{BASE}/query", json={"query": text})
    return r.json()

def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# Give server a moment to be ready
time.sleep(2)

# ── 1. Cache MISS ─────────────────────────────────────────────
separator("POST /query  ->  cache MISS")
d = post_query("existence of god religion beliefs christian bible discussion")
print(json.dumps({
    "query":            d["query"],
    "cache_hit":        d["cache_hit"],
    "matched_query":    d["matched_query"],
    "similarity_score": d["similarity_score"],
    "result":           [{"text": r["text"][:80]+"...", "category": r["category"]} for r in d["result"]],
    "dominant_cluster": d["dominant_cluster"],
}, indent=2))

# ── 2. Cache HIT (exact) ──────────────────────────────────────
separator("POST /query  ->  cache HIT (exact same query)")
d = post_query("existence of god religion beliefs christian bible discussion")
print(json.dumps({
    "query":            d["query"],
    "cache_hit":        d["cache_hit"],
    "matched_query":    d["matched_query"],
    "similarity_score": d["similarity_score"],
    "result":           "[ ... same cached result ... ]",
    "dominant_cluster": d["dominant_cluster"],
}, indent=2))

# ── 3. Cache HIT (semantic) ───────────────────────────────────
separator("POST /query  ->  cache HIT (semantically similar)")
d = post_query("does god exist christian faith bible")
print(json.dumps({
    "query":            d["query"],
    "cache_hit":        d["cache_hit"],
    "matched_query":    d["matched_query"],
    "similarity_score": d["similarity_score"],
    "result":           "[ ... cached result ... ]",
    "dominant_cluster": d["dominant_cluster"],
}, indent=2))

# ── 4. Cache MISS (different topic) ───────────────────────────
separator("POST /query  ->  cache MISS (different topic)")
d = post_query("NASA space shuttle orbit launch")
print(json.dumps({
    "query":            d["query"],
    "cache_hit":        d["cache_hit"],
    "matched_query":    d["matched_query"],
    "similarity_score": d["similarity_score"],
    "result":           "[ ... new results ... ]",
    "dominant_cluster": d["dominant_cluster"],
}, indent=2))

# ── 5. Cache stats ────────────────────────────────────────────
separator("GET /cache/stats")
r = requests.get(f"{BASE}/cache/stats")
print(json.dumps(r.json(), indent=2))
