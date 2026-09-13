import os
import json
import re
import time
import boto3
import chromadb
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from src.ingest import get_embedding
from src.paths import CHROMA_DIR

load_dotenv()

CHAT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
COLLECTION_NAME = "asrs_narratives"
RETRYABLE_ERRORS = {
    "ThrottlingException",
    "ModelNotReadyException",
    "ServiceUnavailableException",
}
DEFAULT_K = 5
FILTERED_K = 10

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(COLLECTION_NAME)

# Airport code -> number of documents tagged with it. Built once at import.
# Also guarantees tokens like "ATC", "FAA" or "CRJ" are never treated as airports.
AIRPORT_COUNTS: dict[str, int] = {}
for m in collection.get(include=["metadatas"])["metadatas"]:
    code = m.get("airport")
    if code and code != "UNKNOWN":
        AIRPORT_COUNTS[code] = AIRPORT_COUNTS.get(code, 0) + 1


def detect_airport(question: str) -> str | None:
    for token in re.findall(r"\b[A-Z]{3}\b", question):
        if token in AIRPORT_COUNTS:
            return token
    return None


def retrieve(question: str) -> tuple[list[str], str | None]:
    query_embedding = get_embedding(question)
    airport = detect_airport(question)

    if airport:
        where = {"airport": airport}
        k = min(FILTERED_K, AIRPORT_COUNTS[airport])
    else:
        where = None
        k = DEFAULT_K

    results = collection.query(
        query_embeddings=[query_embedding], n_results=k, where=where
    )
    return results["documents"][0], airport


def ask_claude(question: str, context_chunks: list[str], max_retries: int = 3) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are an aviation safety assistant. Answer the question using ONLY the context below, which is drawn from real ASRS aviation safety incident reports. If the context doesn't contain a clear answer, say so explicitly.

Context:
{context}

Question: {question}

Answer:"""

    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                modelId=CHAT_MODEL_ID,
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1000,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                ),
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code not in RETRYABLE_ERRORS or attempt == max_retries - 1:
                raise
            wait = 2**attempt
            print(f"Attempt {attempt + 1} failed with {code}; retrying in {wait}s")
            time.sleep(wait)


def query(question: str) -> dict:
    chunks, airport = retrieve(question)
    answer = ask_claude(question, chunks)
    return {"answer": answer, "chunks": chunks, "airport": airport}


if __name__ == "__main__":
    while True:
        q = input("\nQuestion (or 'quit'): ")
        if q.lower() in ("quit", "q"):
            break
        result = query(q)
        if result["airport"]:
            print(
                f"\n[filtered to airport: {result['airport']}, {len(result['chunks'])} chunks]"
            )
        print("\n" + result["answer"])
        print("\n--- Retrieved chunks ---")
        for i, chunk in enumerate(result["chunks"], 1):
            print(f"\n[{i}] {chunk[:200]}...")
