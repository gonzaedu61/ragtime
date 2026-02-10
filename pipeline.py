import os
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

#from openai import AsyncOpenAI
#from azure.ai.formrecognizer.aio import DocumentAnalysisClient
#from azure.core.credentials import AzureKeyCredential

from pdf_chunker import extract_and_chunk_pdf


#from openai import AzureOpenAI
from openai import AsyncAzureOpenAI


# ---------------------------------------------------------
# 0. Load environment variables
# ---------------------------------------------------------
load_dotenv()

FOUNDATION_ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT")  # ends with .models.ai.azure.com
FOUNDATION_API_KEY = os.getenv("AZURE_AI_PROJECT_API_KEY")

#DI_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
#DI_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")

PDF_FOLDER = "DATA/PDFs"


# ---------------------------------------------------------
# 1. Initialize async clients
# ---------------------------------------------------------

'''
llm = AsyncOpenAI(
    api_key=FOUNDATION_API_KEY,
    base_url=FOUNDATION_ENDPOINT,
)
'''



endpoint = "https://ragtime-openai.openai.azure.com/"
model_name = "o4-mini"
deployment = "o4-mini"

#subscription_key = (get from .env)
api_version = "2024-12-01-preview"

client = AsyncAzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)








'''
document_client = DocumentAnalysisClient(
    endpoint=DI_ENDPOINT,
    credential=AzureKeyCredential(DI_KEY)
)
'''


# ---------------------------------------------------------
# 2. Load PDFs
# ---------------------------------------------------------
def load_pdfs(folder):
    for file in Path(folder).glob("*.pdf"):
        yield file.name, file.read_bytes()


# ---------------------------------------------------------
# 3. Extract text using Document Intelligence
# ---------------------------------------------------------
async def extract_text(pdf_bytes):
    try:
        poller = await document_client.begin_analyze_document(
            model_id="prebuilt-read",
            document=pdf_bytes
        )
        result = await poller.result()

        if not result.pages:
            return ""

        return "\n".join(
            line.content
            for page in result.pages
            for line in page.lines
        )

    except Exception as e:
        return f"[ERROR extracting text: {e}]"


# ---------------------------------------------------------
# 4. Run a prompt against the LLM
# ---------------------------------------------------------
async def run_prompt(system_prompt, text, metadata):
    try:
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
                {"role": "user", "content": metadata},
            ],
            max_completion_tokens=2000,
            model=deployment

        )
        return response.choices[0].message.content

    except Exception as e:
        return f"[ERROR calling LLM: {e}]"


# ---------------------------------------------------------
# 5. Load prompts A/B/C/D
# ---------------------------------------------------------
def load_prompt(path):
    return Path(path).read_text(encoding="utf-8")

PROMPT_A = load_prompt("prompts/prompt_A.txt")
PROMPT_B = load_prompt("prompts/prompt_B.txt")
PROMPT_C = load_prompt("prompts/prompt_C.txt")
PROMPT_D = load_prompt("prompts/prompt_D.txt")


# ---------------------------------------------------------
# 6. Process a single document chunk (parallel A/B/C/D)
# ---------------------------------------------------------
async def process_document(chunk):

    metadata = "This is the metadata\n"
    metadata += "id: " + str(chunk["chunk_id"]) + "\n"
    metadata += "index: " + str(chunk["chunk_number"]) + "\n"
    metadata += "document: " + str(chunk["document"]) + "\n"
    metadata += "page: " + str(chunk["page"]) + "\n"

    text = "This is the text:\n " + chunk["text"]

    A, B, C, D = await asyncio.gather(
        run_prompt(PROMPT_A, text, metadata),
        run_prompt(PROMPT_B, text, metadata),
        run_prompt(PROMPT_C, text, metadata),
        run_prompt(PROMPT_D, text, metadata)
    )

    return {

        "id": chunk["chunk_id"],
        "index": chunk["chunk_number"],
        "document": chunk["document"],
        "page": chunk["page"],
        "text": chunk["text"],
        "A": A,
        "B": B,
        "C": C,
        "D": D
    }


# ---------------------------------------------------------
# 7. Aggregate results
# ---------------------------------------------------------
def aggregate_results(results):
    return {
        "A": [r["A"] for r in results],
        "B": [r["B"] for r in results],
        "C": [r["C"] for r in results],
        "D": [r["D"] for r in results],
    }


# ---------------------------------------------------------
# 8. Unified model generation
# ---------------------------------------------------------
UNIFIED_PROMPT = """
Using the following A/B/C/D outputs, generate a unified process description...
"""

async def generate_unified_model(aggregated):
    text = json.dumps(aggregated, indent=2)
    return await run_prompt(UNIFIED_PROMPT, text)


# ---------------------------------------------------------
# 9. Main pipeline
# ---------------------------------------------------------

import json

def safe_json_loads(value):
    if not value or not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


async def main():
    results = []

    for filename, pdf_bytes in load_pdfs(PDF_FOLDER):
        print(f"Processing {filename}...")

        print('chunking ...')


        chunks = extract_and_chunk_pdf(pdf_bytes, document_name=filename)
        doc_results = []

        print('analyzing ...')

        for chunk in chunks: 
            print('Chunk: ', chunk["chunk_id"])
            result = await process_document(chunk)

            result['A'] = safe_json_loads(result['A'])
            result['B'] = safe_json_loads(result['B'])
            result['C'] = safe_json_loads(result['C'])
            result['D'] = safe_json_loads(result['D'])

            with open("DATA/A_B_C_D/chunk-" + chunk["chunk_id"] + ".json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            doc_results.append(result)

    sys.exit()

    aggregated = aggregate_results(results)
    unified_model = await generate_unified_model(aggregated)

    print("\n=== Unified Process Model ===\n")
    print(unified_model)


# ---------------------------------------------------------
# 10. Run
# ---------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
