import os
import json
import time
import boto3
import chromadb
from dotenv import load_dotenv
from src.ingest import get_embedding
from src.paths import CHROMA_DIR

load_dotenv()

CHAT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection("asrs_narratives")


def retrieve(question: str, k: int = 5) -> list[str]:
    query_embedding = get_embedding(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["documents"][0]


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
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                ),
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise


def query(question: str) -> dict:
    chunks = retrieve(question)
    answer = ask_claude(question, chunks)
    return {"answer": answer, "chunks": chunks}


if __name__ == "__main__":
    while True:
        q = input("\nQuestion (or 'quit'): ")
        if q.lower() == "quit" or q.lower() == "q":
            break
        result = query(q)
        print("\n" + result["answer"])
        print("\n--- Retrieved chunks ---")
        for i, chunk in enumerate(result["chunks"], 1):
            print(f"\n[{i}] {chunk[:200]}...")
