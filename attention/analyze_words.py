"""Which kinds of words get attention, and does that differ between highest and lowest answers?

Reads an extract_attention.py run (tokens-*.parquet), tags every answer word with a
part of speech and a word class (word_classes.py), and writes to <run>/analysis/:

  words.parquet           one row per word: class, POS, position, attention per layer band
  by_class.csv            attention per word class, with 95% CIs
  by_pos.csv              attention per part of speech
  by_class_band.csv       attention per word class and layer band
  highest_vs_lowest.csv   paired by question: highest- vs lowest-scored answer

Attention measure: log2 of the attention a word receives (relative to an even spread,
averaged over its tokens and all layers), minus the mean of its text (so texts are
comparable) and minus the mean at its relative position in the text (early words
are attended differently from late ones). 0 = typical, +1 = twice the typical attention.
--measure picks the column family: recv (attention), vrecv (value-weighted attention,
--value-norm), grel (attention x gradient, extract_attn_grad.py).

Confidence intervals: bootstrap over questions (the words of one question are not
independent), 1000 resamples.

Usage:
  python attention/analyze_words.py
  python attention/analyze_words.py --run attention/output/sample200/Qwen3-1.7B-Base
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from word_classes import CLASS_ORDER, tag_words  # noqa: E402

MIN_PAIRED = 10  # questions needed before a paired confidence interval is reported


def band_ranges(n_layers: int) -> dict:
    """Four equal layer bands, labelled 1-based (28 layers: L01-07, ..., L22-28)."""
    edges = np.linspace(0, n_layers, 5).round().astype(int)
    return {f"L{a + 1:02d}-{b:02d}": range(a, b) for a, b in zip(edges[:-1], edges[1:])}


def load_words(run: Path, data: Path, measure: str) -> pd.DataFrame:
    tokens = pd.concat(pd.read_parquet(p) for p in sorted(run.glob("tokens-*.parquet")))
    layer_cols = sorted(c for c in tokens.columns if c.startswith(f"{measure}_L"))
    if not layer_cols:
        raise SystemExit(f"no {measure}_L.. columns in {run}; run the extraction with the matching option")
    bands = band_ranges(len(layer_cols))
    tokens = tokens[tokens["word_index"] >= 0]

    per_word = tokens.groupby(["answer_id", "word_index"])[layer_cols].mean()
    out = pd.DataFrame(index=per_word.index)
    out["recv"] = per_word.mean(axis=1)
    for band, layers in bands.items():
        cols = [c for c in layer_cols if int(c[-2:]) in layers]
        out[f"recv_{band}"] = per_word[cols].mean(axis=1)
    out = out.reset_index()

    texts = pd.read_csv(data, encoding="utf-8-sig")
    texts = texts[texts["answer_id"].isin(out["answer_id"].unique())]
    tagged = []
    for row in texts.itertuples(index=False):
        for w in tag_words(row.answer_text):
            tagged.append({"answer_id": row.answer_id, "question_id": row.question_id,
                           "category": row.category, "score_group": row.score_group,
                           "n_words_text": row.answer_words, **w})
    words = pd.DataFrame(tagged).merge(out, on=["answer_id", "word_index"], how="inner")
    return words.dropna(subset=["recv"])  # the last word has no later tokens


def band_columns(words: pd.DataFrame) -> list[str]:
    return [c for c in words.columns if c.startswith("recv_L")]


def add_measure(words: pd.DataFrame) -> pd.DataFrame:
    words["rel_pos"] = words["word_index"] / words.groupby("answer_id")["word_index"].transform("max")
    words["pos_bin"] = (words["rel_pos"] * 10).clip(upper=9.99).astype(int)
    for col in ["recv"] + band_columns(words):
        y = np.log2(words[col].clip(lower=1e-6))
        y = y - y.groupby(words["answer_id"]).transform("mean")
        y = y - y.groupby(words["pos_bin"]).transform("mean")
        words["att" if col == "recv" else "att_" + col[5:]] = y
    return words


def boot_ci(df: pd.DataFrame, value: str, n_boot: int, rng) -> tuple:
    """Mean of `value` with a 95% CI from resampling questions."""
    g = df.groupby("question_id")[value].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    means = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return s.sum() / c.sum(), *np.percentile(means, [2.5, 97.5])


def summarize(words: pd.DataFrame, key: str, value: str, n_boot: int, rng, min_n: int = 1) -> pd.DataFrame:
    rows = []
    for k, sub in words.groupby(key):
        if len(sub) < min_n:
            continue
        m, lo, hi = boot_ci(sub, value, n_boot, rng)
        rows.append({key: k, "n_words": len(sub), "n_texts": sub["answer_id"].nunique(),
                     "share_of_words_%": 100 * len(sub) / len(words),
                     "attention": m, "ci_low": lo, "ci_high": hi, "x_typical": 2 ** m})
    return pd.DataFrame(rows).set_index(key)


def paired_highest_lowest(words: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    """Per question and class: highest minus lowest answer, in attention and in frequency."""
    ends = words[words["score_group"].isin(["highest", "lowest"])]
    n_words = ends.groupby(["question_id", "score_group"])["answer_id"].size()
    rows = []
    for cls in CLASS_ORDER:
        sub = ends[ends["cls"] == cls]
        att = sub.groupby(["question_id", "score_group"])["att"].mean().unstack()
        att = att.dropna(subset=[c for c in ["highest", "lowest"] if c in att]) if len(att) else att
        freq = (ends.assign(hit=ends["cls"] == cls).groupby(["question_id", "score_group"])["hit"].sum()
                / n_words * 100).unstack().dropna()
        row = {"cls": cls}
        for name, table in [("attention", att), ("per_100_words", freq)]:
            if {"highest", "lowest"} <= set(table.columns) and len(table):
                d = (table["highest"] - table["lowest"]).to_numpy()
                boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
                # A bootstrap over a handful of questions gives misleadingly narrow intervals.
                ok = len(d) >= MIN_PAIRED
                row.update({f"{name}_diff": d.mean(),
                            f"{name}_ci_low": np.percentile(boots, 2.5) if ok else np.nan,
                            f"{name}_ci_high": np.percentile(boots, 97.5) if ok else np.nan,
                            f"{name}_n_questions": len(d)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("cls")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=HERE / "output" / "sample200" / "Qwen3-1.7B-Base")
    p.add_argument("--data", type=Path, default=HERE.parent / "DATASET" / "eli5c_multi_1k.csv")
    p.add_argument("--measure", default="recv", choices=["recv", "vrecv", "grel"],
                   help="recv = attention, vrecv = value-weighted attention, grel = attention x gradient")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.run / ("analysis" if args.measure == "recv" else f"analysis_{args.measure}")
    out.mkdir(exist_ok=True)

    words = add_measure(load_words(args.run, args.data, args.measure))
    words.to_parquet(out / "words.parquet", index=False)
    print(f"{words['answer_id'].nunique()} explanations, {words['question_id'].nunique()} questions, "
          f"{len(words)} words\n")

    by_class = summarize(words, "cls", "att", args.boot, rng).reindex(CLASS_ORDER)
    by_class.to_csv(out / "by_class.csv")
    by_pos = summarize(words, "pos", "att", args.boot, rng, min_n=50).sort_values("attention", ascending=False)
    by_pos.to_csv(out / "by_pos.csv")
    bands = [c[5:] for c in band_columns(words)]
    band = pd.DataFrame({b: words.groupby("cls")[f"att_{b}"].mean() for b in bands}).reindex(CLASS_ORDER)
    band.to_csv(out / "by_class_band.csv")
    paired = paired_highest_lowest(words, args.boot, rng)
    paired.to_csv(out / "highest_vs_lowest.csv")

    pd.set_option("display.width", 160)
    cols = ["n_words", "n_texts", "share_of_words_%", "attention", "ci_low", "ci_high", "x_typical"]
    print("Attention by word class (0 = typical, CI over questions):")
    print(by_class[cols].round(3).to_string(), "\n")
    print("Attention by part of speech (>= 50 words):")
    print(by_pos[cols].round(3).to_string(), "\n")
    print("Attention by word class and layer band:")
    print(band.round(3).to_string(), "\n")
    print("Highest minus lowest answer of the same question "
          f"(no CI when fewer than {MIN_PAIRED} questions have the class in both answers):")
    print(paired.round(3).to_string())
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
