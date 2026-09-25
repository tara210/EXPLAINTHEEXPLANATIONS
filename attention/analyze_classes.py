"""Attention by word class, controlled: does a word class get more attention than other
words *with the same part of speech, frequency, length and position*?

Reads a run (default: attention x gradient, Qwen3-4B, 1,000 explanations), tags words
(word_classes.py, with spaCy disambiguation) and fits, per word:

    log2(score) ~ class + controls + text fixed effect

  score     attention received (measure --measure, averaged over the word's tokens and layers)
  class     causal, contrast, ... ; "other" is the baseline
  controls  part of speech, log word frequency (all ELI5 answers), number of sub-word
            tokens, sentence-initial, relative position in the text (deciles),
            tokens after the word (bins: how many later tokens can attend to it)
  text FE   every explanation gets its own intercept (the model compares words within a
            text), fitted by demeaning within text
  SEs       clustered by question (the explanations of one question are not independent)

Three models, so you can see what the controls change:
  naive     class only (+ text FE)
  pos       class + part of speech
  full      class + all controls

Writes to <run>/analysis_classes/:
  classes.csv        class coefficients per model, as x-factor with 95% CI
  classes_bands.csv  full model per layer band
  position.csv       position check: mean score per position decile, raw / after adjustment
  disambiguation.csv how many words each class loses to the spaCy rules
  summary.json       everything above, for the results page

Usage:
  python attention/analyze_classes.py
  python attention/analyze_classes.py --run attention/output/sample200/Qwen3-1.7B-Base --measure recv
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from analyze_words import band_ranges  # noqa: E402
from word_classes import CLASS_ORDER, WORD, norm, tag_words  # noqa: E402

CLASSES = [c for c in CLASS_ORDER if c != "other"]
FREQ_CACHE = HERE / "output" / "word_freq.parquet"
# Tokens after the word = how many later tokens can attend to it; scores depend strongly on it.
AFTER_BINS = [-1, 0, 2, 5, 10, 20, 40, 80, 160, 10_000]
AFTER_LABELS = ["0", "1-2", "3-5", "6-10", "11-20", "21-40", "41-80", "81-160", "161+"]


def word_frequencies() -> pd.Series:
    """log10 frequency per million words over all ELI5-Category answers (cached)."""
    if FREQ_CACHE.exists():
        return pd.read_parquet(FREQ_CACHE)["log_freq"]
    answers = pd.read_parquet(ROOT / "DATASET" / "processed" / "eli5c_answers.parquet", columns=["text"])
    counts = Counter()
    for t in answers["text"]:
        counts.update(norm(w) for w in WORD.findall(t))
    counts.pop("", None)
    total = sum(counts.values())
    freq = pd.DataFrame({"word_norm": list(counts), "count": list(counts.values())})
    freq["log_freq"] = np.log10((freq["count"] + 1) / total * 1e6)
    freq = freq.set_index("word_norm")
    FREQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    freq.to_parquet(FREQ_CACHE)
    return freq["log_freq"]


def load(run: Path, data: Path, measure: str, texts_limit: int = 0) -> pd.DataFrame:
    tokens = pd.concat(pd.read_parquet(p) for p in sorted(run.glob("tokens-*.parquet")))
    cols = sorted(c for c in tokens.columns if c.startswith(f"{measure}_L"))
    if not cols:
        raise SystemExit(f"no {measure}_L.. columns in {run}")
    if texts_limit:
        keep = tokens["answer_id"].drop_duplicates().head(texts_limit)
        tokens = tokens[tokens["answer_id"].isin(keep)]
    tokens = tokens[tokens["word_index"] >= 0]
    g = tokens.groupby(["answer_id", "word_index"])
    per_word = g[cols].mean()
    last_pos = tokens.groupby("answer_id")["position"].transform("max")
    tokens = tokens.assign(n_after=last_pos - tokens["position"])
    g = tokens.groupby(["answer_id", "word_index"])
    out = pd.DataFrame({"score": per_word.mean(axis=1), "n_subtokens": g.size(), "n_after": g["n_after"].min()})
    for band, layers in band_ranges(len(cols)).items():
        out[f"score_{band}"] = per_word[[c for c in cols if int(c[-2:]) in layers]].mean(axis=1)
    out = out.reset_index()

    texts = pd.read_csv(data, encoding="utf-8-sig")
    texts = texts[texts["answer_id"].isin(out["answer_id"].unique())]
    rows = []
    for r in texts.itertuples(index=False):
        for w in tag_words(r.answer_text):
            rows.append({"answer_id": r.answer_id, "question_id": r.question_id, "category": r.category,
                         "score_group": r.score_group, **{k: w[k] for k in
                         ["word_index", "word", "word_norm", "pos", "dep", "sent_start", "cls", "cls_raw"]}})
    words = pd.DataFrame(rows).merge(out, on=["answer_id", "word_index"], how="inner")
    words = words.dropna(subset=["score"])  # the last word has no later tokens
    words = words[words["score"] > 0]
    words["rel_pos"] = words["word_index"] / words.groupby("answer_id")["word_index"].transform("max")
    words["pos_decile"] = (words["rel_pos"] * 10).clip(upper=9.99).astype(int)
    words["pos_20"] = (words["rel_pos"] * 20).clip(upper=19.99).astype(int)
    words["after_bin"] = pd.cut(words["n_after"], AFTER_BINS, labels=AFTER_LABELS).astype(str)
    freq = word_frequencies()
    words["log_freq"] = words["word_norm"].map(freq).fillna(freq.min())
    return words


def design(words: pd.DataFrame, model: str, cls_col: str = "cls") -> pd.DataFrame:
    X = pd.DataFrame(index=words.index)
    for c in CLASSES:
        X[f"cls:{c}"] = (words[cls_col] == c).astype(float)
    if model in ("pos", "full"):
        for p in sorted(words["pos"].unique()):
            if p != "NOUN":  # baseline
                X[f"pos:{p}"] = (words["pos"] == p).astype(float)
    if model == "full":
        X["log_freq"] = words["log_freq"]
        X["n_subtokens"] = words["n_subtokens"].astype(float)
        X["sent_start"] = words["sent_start"].astype(float)
        for d in range(1, 10):
            X[f"decile:{d}"] = (words["pos_decile"] == d).astype(float)
        for b in AFTER_LABELS[:-1]:  # "161+" is the baseline
            X[f"after:{b}"] = (words["after_bin"] == b).astype(float)
        # smooth terms on top of the steps, so the steep end-of-text rise is not left in the residuals
        la = np.log1p(words["n_after"].astype(float))
        X["log_after"], X["log_after^2"] = la, la ** 2
        X["rel_pos"], X["rel_pos^2"] = words["rel_pos"], words["rel_pos"] ** 2
    return X.loc[:, X.std() > 0]


def fe_ols(y: pd.Series, X: pd.DataFrame, text: pd.Series, cluster: pd.Series) -> pd.DataFrame:
    """OLS with text fixed effects (within-text demeaning) and question-clustered SEs."""
    yd = (y - y.groupby(text).transform("mean")).to_numpy()
    Xd = (X - X.groupby(text).transform("mean")).to_numpy()
    keep = Xd.std(0) > 1e-12
    Xd, names = Xd[:, keep], X.columns[keep]
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ yd
    u = yd - Xd @ beta
    n, k = Xd.shape
    groups = pd.factorize(cluster)[0]
    G = groups.max() + 1
    score = np.zeros((G, k))
    np.add.at(score, groups, Xd * u[:, None])
    k_fe = text.nunique()
    corr = G / (G - 1) * (n - 1) / max(n - k - k_fe, 1)
    V = corr * XtX_inv @ (score.T @ score) @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return pd.DataFrame({"coef": beta, "se": se}, index=names)


def control_blocks(X: pd.DataFrame) -> dict:
    return {
        "position": [c for c in X if c.startswith(("decile:", "after:")) or c in ("log_after", "log_after^2", "rel_pos", "rel_pos^2")],
        "part of speech": [c for c in X if c.startswith("pos:")],
        "frequency": ["log_freq"],
        "word length": ["n_subtokens"],
        "sentence start": ["sent_start"],
        "word class": [c for c in X if c.startswith("cls:")],
    }


def within_r2(y: pd.Series, X: pd.DataFrame, text: pd.Series) -> float:
    """Share of the within-text variance of y explained by the columns of X."""
    yd = (y - y.groupby(text).transform("mean")).to_numpy()
    if X.shape[1] == 0:
        return 0.0
    Xd = (X - X.groupby(text).transform("mean")).to_numpy()
    b = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    return float(1 - ((yd - Xd @ b) ** 2).sum() / (yd ** 2).sum())


def controls_impact(y, words) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How much each block of controls explains, and how the class effects change without it."""
    X = design(words, "full")
    blocks = control_blocks(X)
    cols = [c for b in blocks.values() for c in b]
    full = within_r2(y, X[cols], words["answer_id"])
    var = pd.DataFrame([{"block": k, "alone": within_r2(y, X[b], words["answer_id"]),
                         "unique": full - within_r2(y, X[[c for c in cols if c not in b]], words["answer_id"])}
                        for k, b in blocks.items()] + [{"block": "all together", "alone": full, "unique": np.nan}])
    variants = {"full": [], "without frequency": blocks["frequency"], "without part of speech": blocks["part of speech"],
                "without both": blocks["frequency"] + blocks["part of speech"]}
    sens = pd.DataFrame({name: class_table(fe_ols(y, X[[c for c in X if c not in drop]], words["answer_id"],
                                                  words["question_id"]), words)["x"]
                         for name, drop in variants.items()})
    return var.set_index("block"), sens


def class_table(fit: pd.DataFrame, words: pd.DataFrame, cls_col: str = "cls") -> pd.DataFrame:
    rows = []
    for c in CLASSES:
        n = int((words[cls_col] == c).sum())
        if f"cls:{c}" not in fit.index:
            rows.append({"cls": c, "n_words": n})
            continue
        b, s = fit.loc[f"cls:{c}", ["coef", "se"]]
        rows.append({"cls": c, "n_words": n, "n_texts": int(words.loc[words[cls_col] == c, "answer_id"].nunique()),
                     "log2": b, "ci_low": b - 1.96 * s, "ci_high": b + 1.96 * s,
                     "x": 2 ** b, "x_low": 2 ** (b - 1.96 * s), "x_high": 2 ** (b + 1.96 * s)})
    return pd.DataFrame(rows).set_index("cls")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=HERE / "output" / "grad1k" / "Qwen3-4B-Base")
    p.add_argument("--measure", default="grel", choices=["recv", "vrecv", "grel"])
    p.add_argument("--data", type=Path, default=ROOT / "DATASET" / "eli5c_multi_1k.csv")
    p.add_argument("--texts", type=int, default=0, help="only the first N finished explanations (0 = all)")
    p.add_argument("--tag", default="", help="write to analysis_classes_<tag> instead (e.g. first200)")
    p.add_argument("--min-after", type=int, default=20,
                   help="leave out words fewer than N later tokens can attend to (default 20): "
                        "their scores rest on a handful of tokens and are extreme")
    args = p.parse_args()
    out = args.run / ("analysis_classes" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(exist_ok=True)

    words = load(args.run, args.data, args.measure, args.texts)
    n_all = len(words)
    words = words[words["n_after"] >= args.min_after].copy()
    print(f"left out {n_all - len(words)} of {n_all} words with fewer than {args.min_after} later tokens")
    y = np.log2(words["score"])
    n_texts, n_q = words["answer_id"].nunique(), words["question_id"].nunique()
    print(f"{n_texts} explanations, {n_q} questions, {len(words)} words ({args.measure})\n")

    fits, tables = {}, {}
    for model in ["naive", "pos", "full"]:
        fits[model] = fe_ols(y, design(words, model), words["answer_id"], words["question_id"])
        tables[model] = class_table(fits[model], words)
    comp = pd.concat({m: t[["x", "x_low", "x_high"]] for m, t in tables.items()}, axis=1)
    comp.insert(0, "n_words", tables["full"]["n_words"])
    comp.to_csv(out / "classes.csv")
    pd.set_option("display.width", 180)
    print("Attention by word class, relative to other words (x-factor, 95% CI):")
    print(comp.round(2).to_string(), "\n")

    # Same full model with the plain list classes, to see what disambiguation changes.
    raw = class_table(fe_ols(y, design(words, "full", "cls_raw"), words["answer_id"], words["question_id"]),
                      words, "cls_raw")
    dis = pd.DataFrame({"n_raw": raw["n_words"], "n_disambiguated": tables["full"]["n_words"],
                        "x_raw": raw["x"], "x_disambiguated": tables["full"]["x"]})
    dis.to_csv(out / "disambiguation.csv")
    print("Effect of the spaCy disambiguation (full model):")
    print(dis.round(2).to_string(), "\n")

    # Per layer band.
    bands = [c for c in words.columns if c.startswith("score_L")]
    band_tab = pd.DataFrame({b[6:]: class_table(fe_ols(np.log2(words[b].clip(lower=1e-12)), design(words, "full"),
                                                       words["answer_id"], words["question_id"]), words)["x"]
                             for b in bands})
    band_tab.to_csv(out / "classes_bands.csv")
    print("Full model per layer band (x-factor):")
    print(band_tab.round(2).to_string(), "\n")

    # Position check: does the score depend on position before / after adjustment?
    X_full = design(words, "full")
    f = fits["full"]
    Xd = (X_full - X_full.groupby(words["answer_id"]).transform("mean"))[f.index]
    yd = y - y.groupby(words["answer_id"]).transform("mean")
    resid = yd - Xd.to_numpy() @ f["coef"].to_numpy()
    # Check on bins the model does not use directly (20 position bins) and on tokens-after bins.
    def check(key):
        t = pd.DataFrame({"within_text": yd.groupby(words[key]).mean(),
                          "residual_full_model": resid.groupby(words[key]).mean(),
                          "n_words": words.groupby(key).size()})
        t.index.name = key
        return t
    pos = check("pos_20")
    after = check("after_bin").reindex(AFTER_LABELS)
    pos.to_csv(out / "position.csv")
    after.to_csv(out / "position_after.csv")
    print("Position check, 20 bins of relative position (mean log2 score, before / after adjustment):")
    print(pos.round(3).to_string(), "\n")
    print("Position check, by number of tokens after the word:")
    print(after.round(3).to_string(), "\n")

    var, sens = controls_impact(y, words)
    var.to_csv(out / "variance_explained.csv")
    sens.to_csv(out / "controls_sensitivity.csv")
    print("Share of within-text differences explained (alone / unique = lost when the block is removed):")
    print(var.round(3).to_string(), "\n")
    print("Class effects when a control is left out (x-factor):")
    print(sens.round(2).to_string(), "\n")

    pos_coef = f[f.index.str.startswith("pos:") | f.index.isin(["log_freq", "n_subtokens", "sent_start"])]
    summary = {
        "run": str(args.run.relative_to(ROOT)) if args.run.is_relative_to(ROOT) else str(args.run),
        "measure": args.measure, "n_texts": int(n_texts), "n_questions": int(n_q), "n_words": int(len(words)),
        "min_after": args.min_after, "n_words_left_out": int(n_all - len(words)),
        "classes": {m: json.loads(t.round(4).reset_index().to_json(orient="records")) for m, t in tables.items()},
        "bands": json.loads(band_tab.round(4).reset_index().to_json(orient="records")),
        "position": json.loads(pos.round(4).reset_index().to_json(orient="records")),
        "position_after": json.loads(after.round(4).reset_index().to_json(orient="records")),
        "disambiguation": json.loads(dis.round(4).reset_index().to_json(orient="records")),
        "controls": {k: {"x": round(float(2 ** v), 4), "coef": round(float(v), 4)} for k, v in pos_coef["coef"].items()},
        "variance": json.loads(var.round(4).reset_index().to_json(orient="records")),
        "sensitivity": json.loads(sens.round(4).reset_index().rename(columns={"index": "cls"}).to_json(orient="records")),
        "log_freq_p10_p90": [round(float(v), 3) for v in np.percentile(words["log_freq"], [10, 90])],
        "log_freq_class_vs_other": {"class": round(float(words.loc[words["cls"] != "other", "log_freq"].mean()), 3),
                                    "other": round(float(words.loc[words["cls"] == "other", "log_freq"].mean()), 3)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("Controls in the full model (x-factor; POS relative to nouns):")
    print(pos_coef.assign(x=2 ** pos_coef["coef"]).round(3).to_string())
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
