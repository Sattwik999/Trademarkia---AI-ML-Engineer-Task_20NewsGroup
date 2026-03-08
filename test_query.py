import requests

r = requests.post(
    "http://127.0.0.1:8000/query",
    json={"query": "existence of god religion beliefs christian bible discussion"}
)
d = r.json()
print(f"cache_hit: {d['cache_hit']}")
print(f"dominant_cluster: {d['dominant_cluster']}")
print()
for i, item in enumerate(d["result"], 1):
    print(f"[{i}] category: {item['category']}")
    print(f"     text: {item['text'][:150]}...")
    print()
