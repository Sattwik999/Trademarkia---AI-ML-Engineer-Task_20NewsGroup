import numpy as np


class SemanticCache:

    def __init__(self, threshold=0.9):

        self.cache = {}

        self.threshold = threshold

        self.hit_count = 0
        self.miss_count = 0


    def cosine_similarity(self, a, b):

        return np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )


    def lookup(self, cluster_id, query_vector):

        if cluster_id not in self.cache:
            return None

        bucket = self.cache[cluster_id]

        similarities = []

        for vec in bucket["vectors"]:

            sim = self.cosine_similarity(query_vector, vec)

            similarities.append(sim)

        if len(similarities) == 0:
            return None

        best_idx = int(np.argmax(similarities))

        best_score = similarities[best_idx]

        if best_score >= self.threshold:

            self.hit_count += 1

            return {
                "matched_query": bucket["queries"][best_idx],
                "result": bucket["results"][best_idx],
                "similarity_score": float(best_score)
            }

        return None


    def store(self, cluster_id, query, vector, result):

        if cluster_id not in self.cache:

            self.cache[cluster_id] = {
                "queries": [],
                "vectors": [],
                "results": []
            }

        self.cache[cluster_id]["queries"].append(query)
        self.cache[cluster_id]["vectors"].append(vector)
        self.cache[cluster_id]["results"].append(result)

        self.miss_count += 1


    def stats(self):

        total_entries = sum(
            len(self.cache[c]["queries"])
            for c in self.cache
        )

        total = self.hit_count + self.miss_count

        hit_rate = self.hit_count / total if total > 0 else 0

        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }


    def clear(self):

        self.cache = {}

        self.hit_count = 0
        self.miss_count = 0