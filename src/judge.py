import csv
import json
import time
from collections import Counter

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from src.paths import EVALS

load_dotenv()

CHAT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
RETRYABLE_ERRORS = {
    "ThrottlingException",
    "ModelNotReadyException",
    "ServiceUnavailableException",
}
CHUNK_SEP = "\n\n===CHUNK===\n\n"

# The judge is non-deterministic run to run (confirmed even at temperature=0 --
# Bedrock/Claude inference isn't bit-for-bit reproducible), so a single call's
# grade can flip on borderline cases. We take a majority vote over N independent
# calls instead of trusting one sample.
N_VOTES = 3
JUDGE_TEMPERATURE = 0.7

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Few-shot examples drawn directly from hand-graded rows, so the judge
# calibrates to this project's bar for "partial" rather than a generic one.
FEWSHOT = """Example 1 (grounded=partial):
Question: What weather-related conditions have contributed to reported incidents?
Answer excerpt: "...causing loss of weather radar capability..."
Issue: the answer attributes weather-radar loss to a lightning-strike incident, but that
detail actually belongs to a different, separate chunk describing a different flight.
Verdict: partial -- claim is true of the corpus overall but misattributed to the wrong incident.

Example 2 (grounded=partial):
Question: Have any reports described conflicts between requested altitude changes and
traffic separation?
Answer excerpt: "...ATC then instructed them to descend to 5,000ft, creating a situation
where the altitude change (descent) was complicated by the traffic conflict."
Issue: the source chunk shows the crew was already descending in response to a TCAS
Resolution Advisory before ATC gave the 5,000ft instruction -- ATC's instruction formalized
an already-underway descent, it didn't complicate anything. The answer reverses the
causality between the RA-driven descent and ATC's instruction.
Verdict: partial -- facts are individually accurate, but the sequence/causality between
them is misstated.

Example 3 (grounded=y):
Question: Have any reports mentioned maintenance status or deferred maintenance items as
a factor?
Answer: four maintenance-deferral cases, each with specific details (a fire-loop
ambiguity, a metal-chip detector history, a hydraulic pump deferral, a divert after
recurring messages).
Issue: none -- every detail traces to a specific chunk with matching specifics.
Verdict: y -- fully grounded.

Example 4 (grounded=y):
Question: What events led to go-arounds in these reports?
Answer excerpt: "...the crew was higher than expected on approach, leading to an EGPWS
warning (though this approach was continued to landing rather than going around)."
Issue: none -- the answer already states, in its own text, that this case was not an
actual go-around. Do not re-flag a nuance the answer has already correctly caveated.
Verdict: y -- fully grounded.
"""

RAG_RUBRIC = (
    """You are grading whether an AI-generated answer is fully supported by the source
chunks it was given, using the same standard a careful human fact-checker would apply.

Grade "y" if every factual claim in the answer is supported by the chunks, including
correct refusals when the chunks genuinely don't answer the question.

Grade "partial" if the answer is mostly accurate but has at least one issue: a claim not
present in any chunk, a detail misattributed to the wrong chunk, an inference stated as
fact, or a sequence/causality claim that reverses or misstates how events in the chunk
actually relate to each other.

Grade "n" if the answer contains a claim contradicted by or absent from all chunks, or
fails to refuse when the chunks don't support an answer at all.

Before flagging a misattribution, causality, or omitted-nuance issue, quote the exact
phrase from the chunk and the exact phrase from the answer that conflict. The "answer"
quote must be copied verbatim from the text under the "Answer:" heading below -- never
from the "Source chunks:" section. If the words you want to quote as the answer's claim
only appear in the source chunks and not in the answer text itself, you are quoting the
wrong section -- do not flag it. If you cannot quote a directly conflicting phrase that is
verbatim present in the answer, do not flag it -- do not infer a conflict that isn't
stated. Also check whether the answer already contains a caveat or hedge addressing your
concern before flagging it as missing.

"""
    + FEWSHOT
    + """

Now grade this case. Respond with ONLY a JSON object: {"grade": "y"|"partial"|"n", "reason": "one sentence"}

Question: {question}

Answer:
{answer}

Source chunks:
{chunks}
"""
)

STATS_RUBRIC = """You are grading whether an AI-generated answer correctly reports computed
statistics, given the ground-truth numbers those statistics were computed from.

Grade "y" if every number and claim in the answer matches the ground truth, and the answer
is transparent about the data's scope (e.g. doesn't claim a full year when the data is one
month).

Grade "partial" if the numbers are right but a claim about scope, definition, or coverage
is misleading.

Grade "n" if any number in the answer doesn't match the ground truth, or a claim
contradicts it.

Before flagging a scope/coverage claim as misleading, quote the exact words the answer
uses to describe its scope. If the answer already states the correct time period, region,
or other scope qualifier in its own text (e.g. "June 2025 only"), that is transparent --
grade it "y", not "partial". A question asking broadly about "2025" being answered with
only the month of data actually available is NOT itself a scope mismatch to penalize --
that is the system correctly answering with the best data it has. Only grade "partial" for
scope if the answer's own wording could make a reader believe it covers more than it does
(e.g. states a figure without any period qualifier at all).

Example (grade=y): Question asks for the 2025 on-time rate. Answer opens "Delta at ATL,
June 2025 (42,354 flights): On-time arrival rate: 70.82%..." and ends "Source: BTS
Reporting Carrier On-Time Performance, June 2025 only." The scope is stated twice, plainly,
before and after the numbers -- a reader cannot come away thinking this is a full-year
figure. Grade "y", not "partial", even though the question asked about "2025" broadly and
only one month was available.

Respond with ONLY a JSON object: {"grade": "y" | "partial" | "n", "reason": "one sentence"}

Question: {question}

Answer:
{answer}

Ground truth data the answer should be consistent with:
{chunks}
"""


def call_judge(prompt: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                modelId=CHAT_MODEL_ID,
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 300,
                        "temperature": JUDGE_TEMPERATURE,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                ),
            )
            result = json.loads(response["body"].read())
            text = result["content"][0]["text"].strip()
            text = (
                text.removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            return json.loads(text)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code not in RETRYABLE_ERRORS or attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt == max_retries - 1:
                return {"grade": "error", "reason": f"judge output unparseable: {e}"}
            time.sleep(1)


def grade_row(row: dict, n_votes: int = N_VOTES) -> dict:
    if row["source"] == "bts_stats":
        rubric = STATS_RUBRIC
        chunks_text = (
            "(no chunks -- see answer itself, which states the computed figures)"
        )
    else:
        rubric = RAG_RUBRIC
        chunks_text = row["retrieved_chunks"].replace(CHUNK_SEP, "\n\n---\n\n")

    prompt = (
        rubric.replace("{question}", row["question"])
        .replace("{answer}", row["answer"])
        .replace("{chunks}", chunks_text)
    )

    votes = [call_judge(prompt) for _ in range(n_votes)]
    grades = [v["grade"] for v in votes if v["grade"] != "error"]
    if not grades:
        return {"grade": "error", "reason": "all votes unparseable", "votes": votes}

    counts = Counter(grades)
    top_grade, top_count = counts.most_common(1)[0]
    tied = [g for g, c in counts.items() if c == top_count]
    if len(tied) > 1:
        # No clear majority (e.g. 1/1/1 split) -- default to "partial" as the
        # honest middle ground rather than arbitrarily picking a winner.
        top_grade = "partial" if "partial" in tied else tied[0]

    # Use the reason from a vote that matches the chosen grade.
    reason = next(
        (v["reason"] for v in votes if v["grade"] == top_grade and "reason" in v),
        votes[0].get("reason", ""),
    )
    vote_summary = ",".join(f"{g}:{c}" for g, c in counts.most_common())
    return {"grade": top_grade, "reason": reason, "votes": vote_summary}


def main():
    with open(EVALS / "eval_log.csv", newline="", encoding="utf-8") as f:
        log_rows = list(csv.DictReader(f))

    judge_grades = {}
    for row in log_rows:
        result = grade_row(row)
        judge_grades[row["id"]] = result
        votes = result.get("votes", "")
        print(f"[{row['id']}] judge={result['grade']}  ({votes})  {result.get('reason', '')}")

    with open(EVALS / "judge_grades.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "judge_grade", "judge_reason", "votes"])
        w.writeheader()
        for qid, r in judge_grades.items():
            w.writerow(
                {
                    "id": qid,
                    "judge_grade": r["grade"],
                    "judge_reason": r.get("reason", ""),
                    "votes": r.get("votes", ""),
                }
            )
    print("\nWrote evals/judge_grades.csv")

    # Compare against hand grades if they exist.
    grades_path = EVALS / "eval_grades.csv"
    if grades_path.exists():
        with open(grades_path, newline="", encoding="utf-8") as f:
            hand_grades = {r["id"]: r["grounded"] for r in csv.DictReader(f)}

        agree, total = 0, 0
        print("\nid    hand      judge     match")
        for qid in sorted(judge_grades):
            hand = hand_grades.get(qid, "?")
            judge = judge_grades[qid]["grade"]
            match = "yes" if hand == judge else "NO"
            if hand != "?":
                total += 1
                agree += hand == judge
            print(f"{qid:<6}{hand:<10}{judge:<10}{match}")

        if total:
            print(f"\nAgreement: {agree}/{total} ({100 * agree / total:.0f}%)")


if __name__ == "__main__":
    main()
