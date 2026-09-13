import argparse
import csv
import sys
import textwrap

from src.paths import EVALS

csv.field_size_limit(sys.maxsize)
CHUNK_SEP = "\n\n===CHUNK===\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    with open(EVALS / "eval_log.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grades = []
    for r in rows[: args.n]:
        print(
            f"\n{'='*80}\n[{r['id']}] should_answer={r['should_answer']}\nQ: {r['question']}\n{'-'*80}"
        )
        print("ANSWER:\n" + textwrap.fill(r["answer"], 100))
        for j, c in enumerate(r["retrieved_chunks"].split(CHUNK_SEP), 1):
            print(f"\n--- chunk {j} ---\n" + textwrap.fill(c[:1500], 100))
        grounded = input("\nGrounded? [y/n/partial] > ").strip().lower()
        note = input("Note > ").strip()
        grades.append(
            {
                "id": r["id"],
                "should_answer": r["should_answer"],
                "grounded": grounded,
                "note": note,
            }
        )

    with open(EVALS / "eval_grades.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "should_answer", "grounded", "note"])
        w.writeheader()
        w.writerows(grades)
    print("\nWrote evals/eval_grades.csv")


if __name__ == "__main__":
    main()
