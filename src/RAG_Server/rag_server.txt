import os
import asyncio
import time
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# --- Your backends / factories ---
from Embedders import HFEmbeddingBackend
from VectorDB_Factory import create_vectordb
from LLM_Factory import create_llm

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

KB_NAME = os.getenv("KB_NAME")

# ------------------------------------------------------------
# Instantiate global components
# ------------------------------------------------------------

# Embedding backend (multilingual)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "C:/Models"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DISABLE_TRANSFORMERS_AVX_CHECK"] = "1"
embedding_backend = HFEmbeddingBackend("C:/Models/multilingual-e5-large/")

# LLM
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
    api_version=LLM_API_VERSION,
)

# Vector DB backend (Chroma or others)
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME = "Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"

vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH,
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
# Language utilities
# ------------------------------------------------------------

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # deterministic detection


def detect_language(text: str) -> str:
    # Short queries → assume English (you can tweak this later)
    if len(text) < 20:
        return "en"
    try:
        return detect(text)
    except Exception:
        return "en"


async def translate_to_german(text: str) -> str:
    """
    Translate arbitrary user text into German for retrieval against a German KB.
    Returns ONLY the translation.
    """
    prompt = (
        "Translate the following text into German. "
        "Return ONLY the translation, with no explanations or quotes.\n\n"
        f"{text}"
    )
    translation = await llm.acomplete(prompt)
    return translation.strip()


# ------------------------------------------------------------
# Query expansion (language-agnostic, but fed with DB-language query)
# ------------------------------------------------------------

async def expand_query(base_query: str, k: int) -> List[str]:
    """
    LLM-based paraphrase expansion.
    Returns k queries (including the original).
    """
    if k <= 1:
        return [base_query]

    prompt = (
        f"You will generate {k-1} alternative queries "
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
    Embed a single query and search in the Vector DB.
    Multilingual embeddings allow cross-lingual retrieval,
    but we now prefer DB-language queries (German) for better alignment.
    """
    print("  [retrieve_for_single_query] QUERY:", query)

    # Time embedding
    t0 = time.perf_counter()
    embedding = embedding_backend.embed([query])[0]
    t1 = time.perf_counter()

    # Time vector search
    results = vectordb.search(embedding, top_n=max_chunks)
    t2 = time.perf_counter()

    print(f"    Embedding time: {t1 - t0:.4f}s")
    print(f"    VectorDB search time: {t2 - t1:.4f}s")

    normalized: List[Dict[str, Any]] = []
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
    print("  [retrieve_in_parallel] num_queries:", len(queries))
    t0 = time.perf_counter()

    tasks = [
        retrieve_for_single_query(q, max_chunks=max_chunks)
        for q in queries
    ]
    results_per_query = await asyncio.gather(*tasks)

    all_results: List[Dict[str, Any]] = []
    for batch in results_per_query:
        all_results.extend(batch)

    t1 = time.perf_counter()
    print(f"  Total retrieval (all queries): {t1 - t0:.4f}s")
    return all_results


def rerank_chunks(
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Simple score-based re-ranking.
    Replace with cross-encoder / LLM reranker if needed.
    """
    t0 = time.perf_counter()
    sorted_chunks = sorted(
        chunks,
        key=lambda c: c.get("score", 0.0),
        reverse=True,
    )
    top = sorted_chunks[:top_k]
    t1 = time.perf_counter()
    print(f"  Reranking time: {t1 - t0:.4f}s (chunks in: {len(chunks)}, out: {len(top)})")
    return top


# ------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------

def build_context_prompt(query: str, chunks: List[Dict[str, Any]], user_lang: str) -> str:
    context_texts = [c["text"] for c in chunks]
    context_block = "\n\n---\n\n".join(context_texts)

    prompt = (
        "You are a retrieval-augmented assistant.\n"
        "Use ONLY the context below to answer the question.\n"
        "If the retrieved context is empty or below a similarity threshold, ask the user for clarification instead of answering.\n"
        "If the query contains a domain-specific term that does not appear in the knowledge base, ask the user whether they meant one of the known terms."
        "If the query contains a term semantically close to known domain terms but not identical, ask the user to confirm which one they meant."
        "If after all of the above the answer is still not in the context, say you don't know.\n"
        f"IMPORTANT: The answer MUST be written in the same language as the user's question ({user_lang}).\n"
        "Do NOT translate the question. Do NOT answer in English unless the user asked in English.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION ({user_lang}):\n{query}\n\n"
        f"ANSWER (in {user_lang}):"
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
    try:
        t0 = time.perf_counter()
        _ = await llm.acomplete("Reply with 'OK'.")
        t1 = time.perf_counter()
        print(f"[status] LLM health check time: {t1 - t0:.4f}s")
        llm_ok = True
    except Exception:
        llm_ok = False

    return StatusResponse(status="ok", llm_ok=llm_ok)


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    print("\n================= NEW REQUEST =================")
    print(f"Query: {request.query!r}")
    t_start = time.perf_counter()

    # 1) Detect input language
    t0 = time.perf_counter()
    input_lang = detect_language(request.query)
    t1 = time.perf_counter()
    print(f"[step] Language detection: {t1 - t0:.4f}s → {input_lang}")

    # 2) Prepare query for DB retrieval
    #    KB content is German → translate non-German queries to German for retrieval.
    query_for_db = request.query
    if input_lang != "de":
        print("[info] Translating query to German for retrieval against German KB...")
        query_for_db = await translate_to_german(request.query)
        print("       Translated query for DB:", query_for_db)

    # 3) Query expansion (in DB language: German)
    t2 = time.perf_counter()
    expanded_queries = await expand_query(
        base_query=query_for_db,
        k=request.expansion_k,
    )
    t3 = time.perf_counter()
    print(f"[step] Query expansion: {t3 - t2:.4f}s (k={request.expansion_k})")
    print("  Expanded queries (DB language):")
    for q in expanded_queries:
        print("   -", q)

    # 4) Parallel retrieval from local Vector DB
    t4 = time.perf_counter()
    retrieved_chunks = await retrieve_in_parallel(
        queries=expanded_queries,
        max_chunks=request.max_chunks,
    )
    t5 = time.perf_counter()
    print(f"[step] Retrieval (all queries): {t5 - t4:.4f}s (chunks: {len(retrieved_chunks)})")

    # 5) Re-ranking
    t6 = time.perf_counter()
    reranked_chunks = rerank_chunks(
        chunks=retrieved_chunks,
        top_k=request.top_k_rerank,
    )
    t7 = time.perf_counter()
    print(f"[step] Reranking: {t7 - t6:.4f}s")

    # 6) Build prompt and call LLM
    #    QUESTION stays in original user language; CONTEXT is German.
    t8 = time.perf_counter()
    prompt = build_context_prompt(request.query, reranked_chunks, input_lang)
    t9 = time.perf_counter()
    print(f"[step] Prompt construction: {t9 - t8:.4f}s")

    t10 = time.perf_counter()
    llm_answer = await llm.acomplete(prompt)
    t11 = time.perf_counter()
    print(f"[step] LLM answer generation: {t11 - t10:.4f}s")

    final_answer = llm_answer

    t_end = time.perf_counter()
    print("------------------------------------------------")
    print(f"Total time: {t_end - t_start:.4f}s")
    print("Breakdown:")
    print(f"  Language detection:   {t1 - t0:.4f}s")
    print(f"  Query expansion:      {t3 - t2:.4f}s")
    print(f"  Retrieval:            {t5 - t4:.4f}s")
    print(f"  Reranking:            {t7 - t6:.4f}s")
    print(f"  Prompt construction:  {t9 - t8:.4f}s")
    print(f"  LLM answer:           {t11 - t10:.4f}s")
    print("===============================================\n")

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
