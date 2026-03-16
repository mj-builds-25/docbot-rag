# src/generation.py
from groq import Groq
from config.settings import GROQ_MODEL
from src.retrieval import retrieve

groq_client = Groq()   # auto-reads GROQ_API_KEY from .env

SYSTEM_PROMPT = """You are a helpful HR assistant for AtliQ AI.
Answer ONLY using the provided context below.
If the answer is not in the context, say: "I don't have that information."
Always mention which source you're referencing."""

def build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        [f"[Source {i+1}]\n{c['content']}" for i, c in enumerate(chunks)]
    )

def generate_answer(query: str, top_k: int = 5) -> str:
    chunks  = retrieve(query, top_k=top_k)
    context = build_context(chunks)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content