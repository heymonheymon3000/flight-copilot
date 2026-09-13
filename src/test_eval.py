import csv
import json

from src.paths import EVALS
from src.query import query

CHUNK_SEP = "\n\n===CHUNK===\n\n"


def main():
    questions = json.loads((EVALS / "questions.json").read_text(encoding="utf-8"))

    with open(EVALS / "eval_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "question",
                "should_answer",
                "airport_filter",
                "answer",
                "num_chunks_retrieved",
                "retrieved_chunks",
                "notes",
            ]
        )
        for q in questions:
            result = query(q["question"])
            answer = result["answer"]
            chunks = result["chunks"]
            airport = result["airport"] or ""
            print(
                f"\n[{q['id']}] airport_filter={airport or '-'}\nQ: {q['question']}\nA: {answer}\n{'-'*60}"
            )
            writer.writerow(
                [
                    q["id"],
                    q["question"],
                    q["should_answer"],
                    airport,
                    answer,
                    len(chunks),
                    CHUNK_SEP.join(chunks),
                    "",
                ]
            )


if __name__ == "__main__":
    main()
