import requests
import json

def q(text):
    r = requests.post("http://127.0.0.1:8000/query", json={"query": text})
    d = r.json()
    print(f"Query: {text}")
    print(f"  cache_hit={d['cache_hit']}  similarity={d['similarity_score']}  cluster={d['dominant_cluster']}")
    print()

q("space shuttle nasa")       # miss
q("space shuttle nasa")       # exact hit
q("nasa shuttle mission")     # semantic hit (similar meaning)
q("python programming tips")  # miss (different topic)

print("--- Cache Stats ---")
r = requests.get("http://127.0.0.1:8000/cache/stats")
print(json.dumps(r.json(), indent=2))
