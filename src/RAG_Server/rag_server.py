import os
import asyncio
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

# --- Your backends / factories ---
from Embedders import HFEmbeddingBackend
from VectorDB_Factory import create_vectordb
from LLM_Factory import create_llm

# ------------------------------------------------------------
# Config (adapt to your environment)
# ------------------------------------------------------------

# Config Constants
KB_NAME = os.getenv("KB_NAME")

# ------------------------------------------------------------
# Instantiate global components
# ------------------------------------------------------------

# Initialize embedding_backend
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "C:/Models"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DISABLE_TRANSFORMERS_AVX_CHECK"] = "1"
embedding_backend = HFEmbeddingBackend("C:/Models/multilingual-e5-large/")



# Initialize LLM
LLM_BACKEND = "azure"
LLM_NAME = "o4-mini"
LLM_DEPLOYMENT = "o4-mini"
LLM_API_VERSION = "2024-12-01-preview"
API_KEY = os.getenv("AZURE_AI_PROJECT_API_KEY")
END_POINT = "https://ragtime-openai.openai.azure.com/"
llm = create_llm(
    backend=LLM_BACKEND,
    endpoint=END_POINT,
    api_key=API_KEY,
    deployment=LLM_DEPLOYMENT,
    model_name=LLM_NAME,
    api_version=LLM_API_VERSION
)


# Initialize vector DB backend (Chroma or others)
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME="Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"
VECTOR_DB_LANG = "en"  # language of your indexed corpus
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)

# ------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    max_chunks: int = 20          # per expanded query
    expansion_k: int = 3          # number of expanded queries
    top_k_rerank: int = 8         # final chunks passed to LLM


class QueryResponse(BaseModel):
    answer: str
    language: str
    used_queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]


class StatusResponse(BaseModel):
    status: str
    llm_ok: bool


# ------------------------------------------------------------
# Language utilities (simple stubs – swap for your own)
# ------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Replace with langdetect / fastText / LLM-based detection.
    """
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        # Fallback: assume English
        return "en"


async def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Simple LLM-based translation using the same backend.
    Replace with a dedicated translation model if you prefer.
    """
    if source_lang == target_lang:
        return text

    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}. "
        "Keep meaning and tone, no extra commentary.\n\n"
        f"TEXT:\n{text}"
    )
    return await llm.acomplete(prompt)


# ------------------------------------------------------------
# Query expansion
# ------------------------------------------------------------

async def expand_query(base_query: str, target_lang: str, k: int) -> List[str]:
    """
    LLM-based paraphrase expansion in the Vector DB language.
    Returns k queries (including the original).
    """
    if k <= 1:
        return [base_query]

    prompt = (
        f"You will generate {k-1} alternative queries in {target_lang} "
        "that are semantically close to the original.\n"
        "Return ONLY the queries, one per line, no numbering.\n\n"
        f"Original query:\n{base_query}"
    )

    completion = await llm.acomplete(prompt)
    lines = [l.strip() for l in completion.splitlines() if l.strip()]

    expansions = [base_query]
    for line in lines:
        if len(expansions) >= k:
            break
        expansions.append(line)

    # Ensure at least the original
    if not expansions:
        expansions = [base_query]

    return expansions


# ------------------------------------------------------------
# Retrieval + re-ranking
# ------------------------------------------------------------

async def retrieve_for_single_query(
    query: str,
    max_chunks: int,
) -> List[Dict[str, Any]]:
    """
    Embed a single query and search in Chroma.
    """
    embedding = embedding_backend.embed([query])[0]
    # ChromaBackend.search(embedding, top_n=30) → list of dicts
    results = vectordb.search(embedding, top_n=max_chunks)

    # Normalize structure (already close to what we want)
    normalized = []
    for r in results:
        normalized.append(
            {
                "query": query,
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "score": r.get("score", 0.0),
                "metadata": r.get("metadata", {}),
            }
        )
    return normalized


async def retrieve_in_parallel(
    queries: List[str],
    max_chunks: int,
) -> List[Dict[str, Any]]:
    """
    Parallel retrieval from local Vector DB using asyncio.gather.
    """
    tasks = [
        retrieve_for_single_query(q, max_chunks=max_chunks)
        for q in queries
    ]
    results_per_query = await asyncio.gather(*tasks)

    all_results: List[Dict[str, Any]] = []
    for batch in results_per_query:
        all_results.extend(batch)
    return all_results


def rerank_chunks(
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Simple score-based re-ranking.
    Replace with cross-encoder / LLM reranker if needed.
    """
    sorted_chunks = sorted(
        chunks,
        key=lambda c: c.get("score", 0.0),
        reverse=True,
    )
    return sorted_chunks[:top_k]


# ------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------

def build_context_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    context_texts = [c["text"] for c in chunks]
    context_block = "\n\n---\n\n".join(context_texts)

    prompt = (
        "You are a retrieval-augmented assistant.\n"
        "Use ONLY the context below to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{query}\n\n"
        "ANSWER:"
    )
    return prompt


# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------

app = FastAPI(title="RAG Server")


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """
    Status / health endpoint.
    """
    # Very lightweight LLM check (optional)
    try:
        _ = await llm.acomplete("Reply with 'OK'.")
        llm_ok = True
    except Exception:
        llm_ok = False

    return StatusResponse(status="ok", llm_ok=llm_ok)


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    # 1) Detect input language
    input_lang = detect_language(request.query)

    # 2) Translate query to Vector DB language if needed
    query_for_db = await translate(
        request.query,
        source_lang=input_lang,
        target_lang=VECTOR_DB_LANG,
    )

    # 3) Query expansion (in Vector DB language)
    expanded_queries = await expand_query(
        base_query=query_for_db,
        target_lang=VECTOR_DB_LANG,
        k=request.expansion_k,
    )

    # 4) Parallel retrieval from local Vector DB
    retrieved_chunks = await retrieve_in_parallel(
        queries=expanded_queries,
        max_chunks=request.max_chunks,
    )

    # 5) Re-ranking
    reranked_chunks = rerank_chunks(
        chunks=retrieved_chunks,
        top_k=request.top_k_rerank,
    )

    # 6) Build prompt and call LLM in Vector DB language
    prompt = build_context_prompt(query_for_db, reranked_chunks)
    llm_answer_in_db_lang = await llm.acomplete(prompt)

    # 7) Translate answer back to original language if needed
    final_answer = await translate(
        llm_answer_in_db_lang,
        source_lang=VECTOR_DB_LANG,
        target_lang=input_lang,
    )

    return QueryResponse(
        answer=final_answer,
        language=input_lang,
        used_queries=expanded_queries,
        retrieved_chunks=reranked_chunks,
    )


# ------------------------------------------------------------
# Run with: uvicorn rag_server:app --host 0.0.0.0 --port 8000
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag_server:app", host="0.0.0.0", port=8000, reload=True)
