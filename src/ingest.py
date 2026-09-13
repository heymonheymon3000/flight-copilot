import os
import json
import boto3
import pandas as pd
import chromadb
from dotenv import load_dotenv
from src.paths import DATA_RAW, CHROMA_DIR

load_dotenv()

CSV_PATH = DATA_RAW / "asrs_air_carrier_2025_2026.csv"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)


def get_embedding(text: str) -> list[float]:
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text[:8000]}),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def main():
    # header=[0, 1] because this CSV has two header rows (category + field name)
    df = pd.read_csv(CSV_PATH, header=[0, 1])

    narrative_col = ("Report 1", "Narrative")
    synopsis_col = ("Report 1", "Synopsis")
    acn_col = (" ", "ACN")  # unique report ID, useful as metadata

    # Use Narrative when present; fall back to Synopsis if Narrative is missing.
    # header=[0, 1] makes df.columns a MultiIndex, so assigning df["text"] actually
    # creates the column ("text", "") rather than "text" — reference it consistently.
    text_col = ("text", "")
    df[text_col] = df[narrative_col].fillna(df[synopsis_col])
    df = df.dropna(subset=[text_col])

    print(f"Loaded {len(df)} narratives.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("asrs_narratives")

    ids, docs, embeddings, metadatas = [], [], [], []

    for idx, row in df.iterrows():
        text = str(row[text_col])
        embedding = get_embedding(text)

        ids.append(str(row[acn_col]))
        docs.append(text)
        embeddings.append(embedding)
        metadatas.append({"acn": str(row[acn_col])})

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(df)}")

    collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    print(f"Done. Stored {len(ids)} narratives in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
