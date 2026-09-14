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
from src.bts_stats import on_time_summary, detect_carrier, CARRIER_NAMES

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

# Keywords that indicate the question is about delay/cancellation/on-time
# performance (BTS data) rather than a safety narrative (ASRS data).
STATS_TOPIC_WORDS = ("on-time", "on time", "ontime", "cancellation")
STATS_METRIC_WORDS = ("rate", "percentage", "percent", "performance", "%")

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


def is_stats_question(question: str) -> bool:
    q = question.lower()
    has_topic = any(t in q for t in STATS_TOPIC_WORDS)
    has_metric = any(m in q for m in STATS_METRIC_WORDS)
    return has_topic and has_metric


def answer_from_stats(question: str) -> dict:
    airport = detect_airport(question)
    carrier = detect_carrier(question)
    stats = on_time_summary(airport=airport, carrier=carrier)

    if stats["n_flights"] == 0:
        scope = f"{CARRIER_NAMES.get(carrier, carrier)} " if carrier else ""
        scope += f"at {airport} " if airport else ""
        answer = (
            f"The BTS on-time performance data on file (June 2025) has no {scope}"
            f"flight records, so this can't be answered from what's currently loaded."
        )
    else:
        who = stats["carrier_name"] or "All carriers"
        where = f" at {stats['airport']}" if stats["airport"] else " system-wide"
        answer = (
            f"{who}{where}, {stats['period']} ({stats['n_flights']:,} flights):\n"
            f"- On-time arrival rate: {stats['on_time_rate_pct']}%\n"
            f"- Average arrival delay: {stats['avg_arr_delay_min']} min\n"
            f"- Average departure delay: {stats['avg_dep_delay_min']} min\n"
            f"- Cancellation rate: {stats['cancellation_rate_pct']}%\n\n"
            f"Source: BTS Reporting Carrier On-Time Performance, June 2025 only. "
            f"On-time is defined as arriving within 15 minutes of schedule."
        )

    return {"answer": answer, "chunks": [], "airport": airport, "source": "bts_stats"}


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

Report only what the reports describe happening. Do not classify an event against a formal term (e.g. "incursion," "violation," "near miss") unless the source text itself uses that term — describe the event and let the reader draw that conclusion.

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
    if is_stats_question(question):
        return answer_from_stats(question)

    chunks, airport = retrieve(question)
    answer = ask_claude(question, chunks)
    return {
        "answer": answer,
        "chunks": chunks,
        "airport": airport,
        "source": "asrs_rag",
    }


if __name__ == "__main__":
    while True:
        q = input("\nQuestion (or 'quit'): ")
        if q.lower() in ("quit", "q"):
            break
        result = query(q)
        print(f"\n[source: {result['source']}]", end="")
        if result["airport"]:
            print(f" [airport: {result['airport']}]", end="")
        print("\n\n" + result["answer"])
        if result["chunks"]:
            print("\n--- Retrieved chunks ---")
            for i, chunk in enumerate(result["chunks"], 1):
                print(f"\n[{i}] {chunk[:200]}...")
