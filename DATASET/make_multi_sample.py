"""Sample questions that have several explanations with different scores.

Reads DATASET/processed/eli5c_answers.parquet (run prepare_eli5c.py first) and writes
DATASET/eli5c_multi_1k.csv: one row per explanation, grouped by question.

Selection:
  - an answer counts as an explanation if it is 30-400 words, not a counter-question,
    not mostly a link, and not in the "Repost" bucket (flags from prepare_eli5c.py)
  - per question keep the (up to) 5 highest-scored explanations
  - keep questions with >= 3 such explanations, all scores different,
    and the best scored at least 3x the lowest
  - sample N questions, spread as evenly as possible over the categories

Usage:
  python DATASET/make_multi_sample.py
  python DATASET/make_multi_sample.py --n 500 --min-expl 4
"""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def balanced_allocation(available: pd.Series, n: int) -> pd.Series:
    """Split n over categories as evenly as possible; small categories give all they have."""
    alloc = pd.Series(0, index=available.index)
    remaining = n
    open_cats = list(available.index)
    while remaining > 0 and open_cats:
        share = max(remaining // len(open_cats), 1)
        for cat in sorted(open_cats, key=lambda c: available[c]):
            take = min(share, available[cat] - alloc[cat], remaining)
            alloc[cat] += take
            remaining -= take
        open_cats = [c for c in open_cats if alloc[c] < available[c]]
    return alloc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1000, help="number of questions (default 1000)")
    p.add_argument("--min-expl", type=int, default=3, help="min explanations per question (default 3)")
    p.add_argument("--max-expl", type=int, default=5, help="max explanations kept per question (default 5)")
    p.add_argument("--min-ratio", type=float, default=3.0, help="best score / lowest score (default 3)")
    p.add_argument("--min-words", type=int, default=30)
    p.add_argument("--max-words", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(HERE / "eli5c_multi_1k.csv"))
    args = p.parse_args()

    df = pd.read_parquet(HERE / "processed" / "eli5c_answers.parquet")
    expl = df[
        df["n_words"].between(args.min_words, args.max_words)
        & ~df["is_question_reply"] & ~df["is_link_reply"] & ~df["is_repost"]
    ]
    # Answers are already sorted best-first within a question.
    expl = expl.sort_values(["q_id", "score"], ascending=[True, False])
    expl = expl.groupby("q_id").head(args.max_expl)

    stats = expl.groupby("q_id").agg(
        n=("a_id", "size"), n_scores=("score", "nunique"),
        top=("score", "max"), low=("score", "min"), category=("category", "first"),
    )
    eligible = stats[
        (stats["n"] >= args.min_expl)
        & (stats["n_scores"] == stats["n"])
        & (stats["top"] >= args.min_ratio * stats["low"])
    ]

    alloc = balanced_allocation(eligible["category"].value_counts(), args.n)
    picked = pd.concat([
        eligible[eligible["category"] == cat].sample(k, random_state=args.seed)
        for cat, k in alloc.items() if k > 0
    ]).index

    out = expl[expl["q_id"].isin(picked)].copy()
    out["answer_rank"] = out.groupby("q_id").cumcount() + 1
    out["n_explanations"] = out.groupby("q_id")["a_id"].transform("size")
    out["relative_score"] = (out["score"] / out.groupby("q_id")["score"].transform("max")).round(3)
    out["score_group"] = "middle"
    out.loc[out["answer_rank"] == 1, "score_group"] = "highest"
    out.loc[out["answer_rank"] == out["n_explanations"], "score_group"] = "lowest"

    out = out.rename(columns={
        "q_id": "question_id", "title": "question_title", "selftext": "question_text",
        "a_id": "answer_id", "score": "answer_score", "n_words": "answer_words", "text": "answer_text",
    })[[
        "question_id", "category", "question_title", "question_text", "n_explanations",
        "answer_rank", "score_group", "answer_score", "relative_score",
        "answer_words", "answer_text", "answer_id",
    ]].sort_values(["category", "question_id", "answer_rank"])

    # utf-8-sig so Excel detects the encoding (emoji, accents) correctly.
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"eligible questions: {len(eligible)}")
    print(f"picked: {out['question_id'].nunique()} questions, {len(out)} explanations -> {args.out}")
    print("\nquestions per category:")
    print(out.drop_duplicates("question_id")["category"].value_counts().to_string())
    print("\nexplanations per question:", out.drop_duplicates("question_id")["n_explanations"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
