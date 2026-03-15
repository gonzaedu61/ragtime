import os
import asyncio
import time
from typing import List, Dict, Any
import json
import statistics
import math

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
CLUSTERS_COLLECTION_NAME = "Clusters"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"

CLUSTER_TOP_K = 40

vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH,
)


vdb_clusters = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=CLUSTERS_COLLECTION_NAME,
    persist_dir=VDB_PATH,
)


GERMAN_ROOTS = {
    "stamm", "lohn", "daten", "kosten", "stelle", "stellen",
    "personal", "art", "arten", "zeit", "plan", "auftrag",
    "material", "artikel", "kunde", "adresse", "nummer",
    "regel", "regelung", "kalkulation", "preis", "gruppe",
}


def load_all_cluster_metadata():
    """
    Loads all cluster metadata from the Clusters VDB.
    Each cluster record contains:
      - cluster_id
      - representative text (record["Text"])
      - cluster_chunks
      - leaf_chunks
      - keywords
      - summary
    """

    all_clusters = []

    # Iterate over all cluster records in the cluster VDB

    clusters = vdb_clusters.get()
    for i in range(len(clusters["ids"])):

        meta = clusters["metadatas"][i]
        text = clusters["documents"][i]

        record_json = meta.get("record_json")

        if not record_json:
            continue

        data = json.loads(record_json)

        cluster_info = {
            "id": data.get("cluster_id"),
            "representative_text": text,
            "cluster_chunks": data.get("cluster_chunks", []),
            "leaf_chunks": data.get("leaf_chunks", []),
            "keywords": data.get("keywords", []),
            "summary": data.get("summary", ""),
            "raw_metadata": meta,
        }

        all_clusters.append(cluster_info)

    print(f"Loaded {len(all_clusters)} clusters.")
    return all_clusters

ALL_CLUSTERS = load_all_cluster_metadata()


# ------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    max_chunks: int = 40          # per expanded query
    expansion_k: int = 3          # number of expanded queries
    top_k_rerank: int = 10         # final chunks passed to LLM


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
# Terms extraction
# ------------------------------------------------------------
def extract_terms(q: str) -> List[str]:
    q = q.replace("_", " ")
    tokens = q.lower().split()

    terms = set(tokens)

    # Add concatenated forms
    terms.add("".join(tokens))

    # Add compound splits
    for tok in tokens:
        if tok.isalpha():
            for part in split_german_compound(tok):
                terms.add(part)

    result = list(terms)

    print("--> TERMS:", result)

    return result

# ------------------------------------------------------------
# German split
# ------------------------------------------------------------
def split_german_compound(word: str) -> List[str]:
    word = word.lower()
    parts = []

    i = 0
    while i < len(word):
        found = False
        # try longest possible root first
        for j in range(len(word), i, -1):
            segment = word[i:j]
            if segment in GERMAN_ROOTS:
                parts.append(segment)
                i = j
                found = True
                break
        if not found:
            # fallback: consume one character
            parts.append(word[i])
            i += 1

    # filter out single letters unless necessary
    parts = [p for p in parts if len(p) > 1]
    return parts

# ------------------------------------------------------------
# Lexical clusters search
# ------------------------------------------------------------
from rapidfuzz import fuzz
def fuzzy_match(term, text, threshold=80):
    return fuzz.partial_ratio(term, text) >= threshold

def lexical_cluster_search(query: str, max_hits: int = 5) -> List[Dict[str, Any]]:
    q = query.lower()
    hits = []

    for cluster in ALL_CLUSTERS:
        text = cluster["representative_text"].lower()
        if any(fuzzy_match(term, text) for term in extract_terms(q)):
            r = {
                "id": cluster["id"],
                "text": cluster["representative_text"],
                "score": 1,
                "metadata": cluster["raw_metadata"]
            }
            hits.append(r)
            if len(hits) >= max_hits:
                break

    return hits

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

    """
    prompt = (
        f"You will generate {k-1} alternative queries that are semantically close "
        "to the original, but not simple paraphrases.\n"
        "Create variations that:\n"
        "• explore meaning, context, purpose, or usage of the term\n"
        "• rephrase the question as 'what does this refer to', 'how is it used', "
        "'what role does it play', or 'in what context does it appear'\n"
        "• include possible synonyms, related forms, or domain-specific variants "
        "(e.g., German compound forms, underscore variants, abbreviations)\n"
        "• anchor the query in the context of the given text\n"
        "• avoid dictionary-style 'What is X' phrasing\n"
        "• aim to retrieve explanatory or descriptive passages rather than "
        "glossary definitions\n"
        "• prefer formulations that match how technical documentation describes "
        "objects, master data, parameters, or workflows\n\n"
        "Return ONLY the rewritten queries, one per line, no numbering.\n\n"
        f"Original query:\n{base_query}"
    )
    """

    prompt = (
        f"You will generate {k-1} alternative queries "
        "that are semantically close to the original.\n"
        "If the question is dictionary-style (i.e. 'What is X') create variations as 'what does X refer to'\n"
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


def merge_clusters(semantic, lexical):

    seen = set()
    merged = []

    for c in semantic + lexical:
        cid = c["id"]
        if cid not in seen:
            merged.append(c)
            seen.add(cid)

    return merged



async def hierarchical_retrieve_for_single_query(
    query: str,
    max_chunks: int,
) -> List[Dict[str, Any]]:
    """
    Two-stage hierarchical retrieval:
      1) Retrieve top clusters
      2) Expand cluster_chunks ∪ leaf_chunks
      3) Retrieve chunks restricted to those IDs
    """
    #print("B:", query)

    # ---------------------------------------------------------
    # 1) Embed query
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    embedding = embedding_backend.embed([query])[0]
    t1 = time.perf_counter()
    print(f"    Embedding time: {t1 - t0:.4f}s")

    # ---------------------------------------------------------
    # 2) Stage 1: Hybrid Cluster Retrieval
    # ---------------------------------------------------------
    t2 = time.perf_counter()

    # 2a. semantic cluster retrieval
    semantic_clusters = vdb_clusters.search(
        embedding,
        top_n=CLUSTER_TOP_K,
    )

    t3a = time.perf_counter()
    print(f"    Semantic Clusters search time: {t3a - t2:.4f}s")
    print(f"    Retrieved Semantic clusters: {len(semantic_clusters)}")

    # 2b. lexical cluster retrieval (NEW)
    lexical_clusters = lexical_cluster_search(query, max_hits=CLUSTER_TOP_K)


    t3b = time.perf_counter()
    print(f"    Lexical Clusters search time: {t3b - t3a:.4f}s")
    print(f"    Retrieved Lexical clusters: {len(lexical_clusters)}")


    # 2c. merge + dedupe
    cluster_results = merge_clusters(semantic_clusters, lexical_clusters)

    t3c = time.perf_counter()
    print(f"    Hybrid Clusters search time: {t3c - t3b:.4f}s")
    print(f"    Retrieved Hybrid clusters: {len(cluster_results)}")


    """
    t2 = time.perf_counter()
    cluster_results = vdb_clusters.search(
        embedding,
        top_n=CLUSTER_TOP_K,   # your chosen constant
    )
    t3 = time.perf_counter()
    print(f"    Cluster search time: {t3 - t2:.4f}s")
    print(f"    Retrieved clusters: {len(cluster_results)}")
    """

    # ---------------------------------------------------------
    # 3) Expand cluster_chunks ∪ leaf_chunks
    # ---------------------------------------------------------
    candidate_chunk_ids = set()

    for c in cluster_results:
        meta = c["metadata"]
        data = json.loads(meta['record_json'])
        cluster_chunks = data.get("cluster_chunks", [])
        leaf_chunks = data.get("leaf_chunks", [])
        candidate_chunk_ids.update(cluster_chunks)
        candidate_chunk_ids.update(leaf_chunks)

    print(f"    Candidate chunk IDs: {len(candidate_chunk_ids)}")

    # ---------------------------------------------------------
    # 4) Stage 2: Chunk retrieval (restricted search)
    # ---------------------------------------------------------
    t4 = time.perf_counter()
    chunk_results = vectordb.search(
        embedding,
        top_n=max_chunks,
        filter_ids=candidate_chunk_ids,   # <— the key difference
    )
    t5 = time.perf_counter()
    print(f"    Chunk search time: {t5 - t4:.4f}s")
    print(f"    Retrieved chunks: {len(chunk_results)}")

    # ---------------------------------------------------------
    # 5) Normalize output (same shape as before)
    # ---------------------------------------------------------
    normalized: List[Dict[str, Any]] = []
    for r in chunk_results:
        normalized.append(
            {
                "query": query,
                "chunk_id": r["id"],
                "text": r["text"],
                "score": r.get("score", 0.0),
                "metadata": r.get("metadata", {}),
                # Optional: attach cluster info for reranking
                # "cluster_score": ...,
                # "cluster_depth": ...,
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

    # -----------------------------
    # DEDUPLICATE ACROSS QUERIES
    # -----------------------------
    seen = set()
    all_results = []

    for batch in results_per_query:
        for item in batch:
            cid = item["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            all_results.append(item)

    t1 = time.perf_counter()
    print(f"  Total retrieval (all queries): {t1 - t0:.4f}s")
    return all_results




async def hierarchical_retrieve_in_parallel(
    queries: List[str],
    max_chunks: int,
) -> List[Dict[str, Any]]:
    print("  [hierarchical_retrieve_in_parallel] num_queries:", len(queries))
    t0 = time.perf_counter()

    print("A:", queries, max_chunks)

    tasks = [
        hierarchical_retrieve_for_single_query(q, max_chunks=max_chunks)
        for q in queries
    ]

    results_per_query = await asyncio.gather(*tasks)

    # ------------------------------------------------------------
    # DEDUPLICATE ACROSS ALL QUERIES
    # ------------------------------------------------------------
    seen = set()
    all_results = []

    for batch in results_per_query:
        for item in batch:
            cid = item["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            all_results.append(item)

    t1 = time.perf_counter()
    print(f"  Total hierarchical retrieval (all queries): {t1 - t0:.4f}s")
    print(f"  Total hierarchical retrieval chunks: {len(all_results)}")

    return all_results




def rerank_chunks(
    chunks: List[Dict[str, Any]],
    max_total_tokens: int = 2500,
    base_threshold: float = 0.6,
    dynamic_alpha: float = 0.2,
    length_boost: float = 0.30,
) -> List[Dict[str, Any]]:
    """
    Enhanced reranking:
      1) Keep chunks ABOVE a semantic threshold
      2) Stop when total token count reaches max_total_tokens
      3) Dynamic thresholding (mean + alpha * std)
      4) Length-aware scoring (log token bonus)
    """

    t0 = time.perf_counter()

    # --- Extract scores ---
    scores = [c.get("score", 0.0) for c in chunks]
    if not scores:
        return []

    # --- Dynamic thresholding ---
    mean_s = statistics.mean(scores)
    std_s = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    dynamic_threshold = mean_s + dynamic_alpha * std_s

    # Final threshold = max(base_threshold, dynamic_threshold)
    threshold = max(base_threshold, dynamic_threshold)

    # --- Filter by semantic relevance ---
    filtered = [c for c in chunks if c.get("score", 0.0) >= threshold]

    if not filtered:
        # fallback: keep top 1 if everything is below threshold
        filtered = [max(chunks, key=lambda c: c.get("score", 0.0))]

    # --- Length-aware scoring ---
    def final_score(c):
        sim = c.get("score", 0.0)
        tok = max(1, c.get("token_count", 1))
        return sim + length_boost * math.log(1 + tok)

    # --- Sort by final score ---
    sorted_chunks = sorted(filtered, key=final_score, reverse=True)

    # --- Accumulate until token budget ---
    selected = []
    total_tokens = 0

    for c in sorted_chunks:
        tok = c.get("token_count", 0)
        if total_tokens + tok > max_total_tokens:
            break
        selected.append(c)
        total_tokens += tok

    t1 = time.perf_counter()
    print(
        f"  Reranking time: {t1 - t0:.4f}s "
        f"(chunks in: {len(chunks)}, filtered: {len(filtered)}, out: {len(selected)})"
    )

    return selected


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
    retrieved_chunks = await hierarchical_retrieve_in_parallel(
        queries=expanded_queries,
        max_chunks=request.max_chunks,
    )
    t5 = time.perf_counter()
    print(f"[step] Retrieval (all queries): {t5 - t4:.4f}s (chunks: {len(retrieved_chunks)})")

    # 5) Re-ranking
    t6 = time.perf_counter()
    reranked_chunks = rerank_chunks(
        chunks=retrieved_chunks
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
