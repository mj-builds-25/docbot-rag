# src/ingestion.py
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config.settings import (
    EMBEDDING_MODEL, QDRANT_PATH, COLLECTION_NAME, VECTOR_DIM, CHUNK_SIZE, DOC_URL
)

def load_document(url: str) -> str:
    print(f"Loading document from URL...")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.text

def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[dict]:
    lines = [l.strip().lstrip("#").strip() for l in text.splitlines() if l.strip()]
    words = " ".join(lines).split()
    return [
        {"chunk_index": i, "content": " ".join(words[i : i + size])}
        for i in range(0, len(words), size)
    ]

def build_index():
    # Load
    text   = load_document(DOC_URL)
    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")

    # Embed
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts    = [c["content"] for c in chunks]
    vectors  = embedder.encode(texts, show_progress_bar=True)

    # Store
    client = QdrantClient(path=QDRANT_PATH)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=i, vector=v.tolist(), payload={"content": c["content"]})
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    print(f"✅ Indexed {len(points)} chunks successfully!")

if __name__ == "__main__":
    build_index()