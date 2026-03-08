import joblib
import numpy as np

umap_model = joblib.load("models/umap_model.pkl")
gmm_model = joblib.load("models/gmm_model.pkl")


def get_cluster(query_vector):

    reduced = umap_model.transform([query_vector])

    probs = gmm_model.predict_proba(reduced)[0]

    dominant_cluster = int(np.argmax(probs))

    return dominant_cluster, probs