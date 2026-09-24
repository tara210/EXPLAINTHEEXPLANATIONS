"""Download and filter the ELI5-Category dataset (Gao et al., 2021).

Produces two files in DATASET/processed/:

  eli5c_answers.parquet       every answer, one row each, with quality flags
                              (nothing dropped, so filters can be changed later)
  eli5c_explanations.parquet  the filtered subset: the best answer(s) per question
                              that look like actual explanations
  eli5c_explanations_sample.csv  the first 200 filtered rows, for eyeballing

Source: https://huggingface.co/datasets/rexarski/eli5_category (the HF entry is only a
loading script; the data itself is hosted by the authors at jingshensn2.github.io/eli5c).

Notes on the raw data:
  - Splits are divided by topic: train = 9 categories, validation-1 = Culture,
    validation-2 = Repost, test = Engineering. We merge all four and keep `category`.
  - Answers are already pre-filtered by the authors (score >= 3, no [deleted]) and
    sorted best-first within each question.
  - Paragraph breaks were stripped; links are replaced by placeholders URL_0, URL_1, ...

Usage:
  python DATASET/prepare_eli5c.py
  python DATASET/prepare_eli5c.py --min-score 20 --per-category 500
"""

import argparse
import gzip
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://jingshensn2.github.io/eli5c/datasets/"
SPLITS = ["train", "validation-1", "validation-2", "test"]

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
OUT_DIR = HERE / "processed"

URL_TOKEN = re.compile(r"\s*URL_\d+\s*")
MD_LINK = re.compile(r"\[([^\]]*)\]\(\s*URL_\d+[^)]*\)")
# A trailing "Edit: ..." / "EDIT 2: ..." note. Short ones are thanks or typo fixes;
# long ones usually continue the explanation, so only short ones get stripped.
TRAILING_EDIT = re.compile(r"\s*\b(?:edit|update)\s*\d*\s*:(?:(?!\b(?:edit|update)\s*\d*\s*:).)*$", re.IGNORECASE)
MAX_EDIT_WORDS = 25
MARKDOWN = re.compile(r"(\*\*|__|~~|^#+\s*|^&gt;\s*)", re.MULTILINE)


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = RAW_DIR / f"eli5-category-{split}.json.gz"
        if target.exists() and not force:
            continue
        print(f"downloading {target.name} ...")
        urllib.request.urlretrieve(BASE_URL + target.name, target)


def load_answers() -> pd.DataFrame:
    """Flatten all splits into one row per answer."""
    rows = []
    for split in SPLITS:
        with gzip.open(RAW_DIR / f"eli5-category-{split}.json.gz", "rt", encoding="utf-8") as f:
            questions = json.load(f)
        for q in questions:
            a = q["answers"]
            for rank, (a_id, text, score) in enumerate(zip(a["a_id"], a["text"], a["score"])):
                rows.append({
                    "q_id": q["q_id"],
                    "split": split,
                    "category": q["category"],
                    "title": q["title"],
                    "selftext": q["selftext"],
                    "n_answers": len(a["text"]),
                    "a_id": a_id,
                    "rank": rank,  # 0 = highest-scored answer to this question
                    "score": score,
                    "text_raw": text,
                })
    return pd.DataFrame(rows)


def clean(text: str) -> str:
    """Normalise an answer for linguistic analysis (keeps wording, drops noise)."""
    # Strip short trailing edit notes, repeatedly ("... EDIT: typo EDIT 2: thanks!").
    while (m := TRAILING_EDIT.search(text)) and m.start() > 0 and len(m.group().split()) <= MAX_EDIT_WORDS:
        text = text[: m.start()]
    text = MD_LINK.sub(r"\1", text)
    text = URL_TOKEN.sub(" <URL> ", text)
    text = MARKDOWN.sub("", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", text).strip()


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text_raw"].map(clean)
    df["n_words"] = df["text"].str.split().str.len()
    df["n_urls"] = df["text_raw"].str.count(r"URL_\d+")
    df["had_edit"] = df["text_raw"].str.contains(r"\b(?:edit|update)\s*\d*\s*:", case=False, regex=True)
    # Short counter-questions ("Why does mine say 87?") rather than explanations.
    df["is_question_reply"] = (df["n_words"] < 30) & df["text"].str.rstrip().str.endswith("?")
    # Mostly a pointer elsewhere ("Asked before: URL_0", "This should help: URL_0").
    df["is_link_reply"] = (df["n_urls"] > 0) & (df["n_words"] < 20)
    df["is_repost"] = df["category"].eq("Repost")
    return df


def select_explanations(df: pd.DataFrame, args) -> pd.DataFrame:
    keep = (
        (df["score"] >= args.min_score)
        & df["n_words"].between(args.min_words, args.max_words)
        & (df["rank"] < args.top_k)
        & ~df["is_question_reply"]
        & ~df["is_link_reply"]
        & ~df["is_repost"]
    )
    out = df[keep]
    if args.per_category:
        out = (out.groupby("category", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), args.per_category), random_state=args.seed)))
    return out.sort_values(["category", "score"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-score", type=int, default=10, help="minimum Reddit score (default 10)")
    p.add_argument("--min-words", type=int, default=40, help="minimum answer length in words (default 40)")
    p.add_argument("--max-words", type=int, default=400, help="maximum answer length in words (default 400)")
    p.add_argument("--top-k", type=int, default=1, help="keep only the k best answers per question (default 1)")
    p.add_argument("--per-category", type=int, default=0, help="sample at most N per category (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-download", action="store_true")
    args = p.parse_args()

    download(args.force_download)
    df = add_flags(load_answers())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "eli5c_answers.parquet", index=False)

    expl = select_explanations(df, args)
    expl.to_parquet(OUT_DIR / "eli5c_explanations.parquet", index=False)
    cols = ["q_id", "category", "score", "n_words", "title", "text"]
    expl[cols].head(200).to_csv(OUT_DIR / "eli5c_explanations_sample.csv", index=False, encoding="utf-8-sig")

    print(f"\nall answers:  {len(df):>7} ({df['q_id'].nunique()} questions)")
    print(f"explanations: {len(expl):>7} ({expl['q_id'].nunique()} questions)")
    print("\nper category:")
    print(expl["category"].value_counts().to_string())
    print(f"\nwords: median {expl['n_words'].median():.0f}, mean {expl['n_words'].mean():.0f}")
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
