"""
RAG answer generator — uses Groq (llama-3.3-70b) with retrieved context.
Returns structured answer + citations. Supports Markdown tables and formatting.
"""

import logging
import os
from typing import List, Dict, Any

from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert research assistant answering questions
strictly based on the provided document excerpts.

Rules:
- Answer only from the given context. Do NOT hallucinate.
- Be concise but complete.
- Always cite your sources using [PDF: filename, Page X] format inline.
- If the context doesn't contain the answer, say so clearly.
- Use Markdown tables when comparing multiple items or presenting structured data.
- Use **bold** for key terms and important points.
- Use bullet points or numbered lists for step-by-step content.
- Use `code blocks` for any code, commands, or technical syntax.
- Structure longer answers with short paragraphs."""

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment.")
        _client = Groq(api_key=api_key)
    return _client


def build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Excerpt {i}] Source: {c['filename']}, Page {c['page_number']}\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Returns dict with:
      - answer (str, Markdown formatted)
      - sources (list of {filename, page_number})
      - context_used (list of chunk texts)
    """
    if not chunks:
        return {
            "answer": "I could not find relevant information in the documents.",
            "sources": [],
            "context_used": [],
        }

    context = build_context(chunks)
    user_message = f"""Context from documents:
{context}

Question: {query}

Answer (use Markdown formatting, cite sources inline):"""

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )

    answer = response.choices[0].message.content.strip()

    # Deduplicated sources
    seen = set()
    sources = []
    for c in chunks:
        key = (c["filename"], c["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append({"filename": c["filename"], "page_number": c["page_number"]})

    return {
        "answer": answer,
        "sources": sources,
        "context_used": [c["text"] for c in chunks],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }
