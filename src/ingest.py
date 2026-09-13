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
COLLECTION_NAME = "asrs_narratives"
BATCH_SIZE = 500

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


def load_narratives() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, header=[0, 1], low_memory=False)

    narrative_col = ("Report 1", "Narrative")
    synopsis_col = ("Report 1", "Synopsis")
    acn_col = (" ", "ACN")
    locale_col = ("Place", "Locale Reference")

    text_col = ("text", "")
    df[text_col] = df[narrative_col].fillna(df[synopsis_col])
    df = df.dropna(subset=[text_col])

    airport_col = ("airport", "")
    locale = df[locale_col].fillna("")
    is_airport = locale.str.endswith(".Airport") & ~locale.str.startswith("ZZZ")
    df[airport_col] = locale.where(is_airport, "UNKNOWN").str.replace(
        ".Airport", "", regex=False
    )

    out = pd.DataFrame(
        {
            "acn": df[acn_col].astype(str).values,
            "text": df[text_col].astype(str).values,
            "airport": df[airport_col].astype(str).values,
        }
    )
    return out.reset_index(drop=True)


def main():
    df = load_narratives()
    print(f"Loaded {len(df)} narratives.")
    print(f"Rows with a real airport code: {(df['airport'] != 'UNKNOWN').sum()}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids, docs, embeddings, metadatas = [], [], [], []

    for i, row in df.iterrows():
        ids.append(row["acn"])
        docs.append(row["text"])
        embeddings.append(get_embedding(row["text"]))
        metadatas.append({"acn": row["acn"], "airport": row["airport"]})

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(df)}")

        if len(ids) == BATCH_SIZE:
            collection.add(
                ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas
            )
            ids, docs, embeddings, metadatas = [], [], [], []

    if ids:
        collection.add(
            ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas
        )

    print(f"Done. Stored {collection.count()} narratives in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
