# src/retrieval.py
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from config.settings import EMBEDDING_MODEL, QDRANT_PATH, COLLECTION_NAME

# Load once, reuse across calls
embedder = SentenceTransformer(EMBEDDING_MODEL)
client   = QdrantClient(path=QDRANT_PATH)

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    vec  = embedder.encode(query).tolist()
     # using .search() instead of .query_points()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"content": h.payload["content"], "score": round(h.score, 4)}
        for h in hits.points
    ]