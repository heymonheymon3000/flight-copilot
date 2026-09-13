import argparse
import csv
import sys
import textwrap

from src.paths import EVALS

csv.field_size_limit(sys.maxsize)
CHUNK_SEP = "\n\n===CHUNK===\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="grade the first N rows")
    ap.add_argument(
        "--ids", nargs="+", help="grade only these ids, e.g. --ids q011 q017"
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="print full chunk text instead of first 1500 chars",
    )
    args = ap.parse_args()

    with open(EVALS / "eval_log.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.ids:
        wanted = set(args.ids)
        rows = [r for r in rows if r["id"] in wanted]
    else:
        rows = rows[: args.n]

    # Load existing grades so a partial re-grade doesn't wipe the rest.
    grades_path = EVALS / "eval_grades.csv"
    existing = {}
    if grades_path.exists():
        with open(grades_path, newline="", encoding="utf-8") as f:
            existing = {r["id"]: r for r in csv.DictReader(f)}

    for r in rows:
        print(
            f"\n{'='*80}\n[{r['id']}] should_answer={r['should_answer']} airport_filter={r['airport_filter'] or '-'}"
        )
        print(f"Q: {r['question']}\n{'-'*80}")
        print("ANSWER:\n" + textwrap.fill(r["answer"], 100))
        for j, c in enumerate(r["retrieved_chunks"].split(CHUNK_SEP), 1):
            body = c if args.full else c[:1500]
            print(f"\n--- chunk {j} ---\n" + textwrap.fill(body, 100))
        grounded = input("\nGrounded? [y/n/partial] > ").strip().lower()
        note = input("Note > ").strip()
        existing[r["id"]] = {
            "id": r["id"],
            "should_answer": r["should_answer"],
            "airport_filter": r["airport_filter"],
            "grounded": grounded,
            "note": note,
        }

    with open(grades_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["id", "should_answer", "airport_filter", "grounded", "note"]
        )
        w.writeheader()
        w.writerows(existing[k] for k in sorted(existing))
    print(f"\nWrote {grades_path} ({len(existing)} rows)")


if __name__ == "__main__":
    main()
