from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(query):

    vector = model.encode([query])[0]

    return vector