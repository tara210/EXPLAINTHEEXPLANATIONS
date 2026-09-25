"""Collect the analysis results into attention/results/data.js for the results page.

Sections (each skipped if its inputs do not exist yet):
  sample200   1.7B, 200 explanations: word classes, POS, layer bands, highest vs lowest
              (from analyze_words.py output)
  pairs       4B, 5 highest/lowest pairs:
                sink share per layer: plain vs value-weighted vs attention x gradient
                word classes per measure, split by highest / lowest
                most attended words per measure
                agreement between measures per word (Spearman), including occlusion
                heads that prefer causal / contrast / illustration words
                answer surprisal (how predictable each explanation is)

Usage:
  python attention/build_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_words import add_measure, load_words  # noqa: E402
from word_classes import CLASS_ORDER  # noqa: E402

DATA = HERE.parent / "DATASET" / "eli5c_multi_1k.csv"
OUT = HERE / "results" / "data.js"
MAX_HEAD_SINK = 0.75  # skip heads that put more than this on the sink (their rest is amplified noise)
MEASURES = {"recv": "Attention", "vrecv": "Value-weighted", "grel": "Attention × gradient"}
OCC_LATER = "Occlusion, tokens 2–20"


def records(df: pd.DataFrame, digits: int = 3) -> list:
    return json.loads(df.round(digits).reset_index().to_json(orient="records"))


def sample200() -> dict | None:
    a = HERE / "output" / "sample200" / "Qwen3-1.7B-Base" / "analysis"
    if not (a / "by_class.csv").exists():
        return None
    return {
        "model": "Qwen/Qwen3-1.7B-Base",
        "n_texts": 200, "n_questions": 51,
        "by_class": records(pd.read_csv(a / "by_class.csv")),
        "by_pos": records(pd.read_csv(a / "by_pos.csv")),
        "by_band": records(pd.read_csv(a / "by_class_band.csv")),
        "paired": records(pd.read_csv(a / "highest_vs_lowest.csv")),
    }


def layer_series(texts: pd.DataFrame, prefix: str) -> list:
    cols = sorted(c for c in texts.columns if c.startswith(prefix + "_L") and "_H" not in c)
    return [round(float(texts[c].mean()), 4) for c in cols]


def heads_preference(run: Path, words: pd.DataFrame, texts: pd.DataFrame) -> list:
    parts = sorted(run.glob("heads-*.parquet"))
    if not parts:
        return []
    h = pd.concat(pd.read_parquet(p) for p in parts)
    h = h[h["word_index"] >= 0]
    cols = [c for c in h.columns if c.startswith("h_L")]
    per_word = h.groupby(["answer_id", "word_index"])[cols].mean().astype("float32")
    y = np.log2(per_word.clip(lower=1e-6))
    y = y - y.groupby(level="answer_id").transform("mean")  # centre within text, per head
    cls = words.set_index(["answer_id", "word_index"])["cls"].reindex(y.index)
    sink = texts[[c.replace("h_", "hsink_") for c in cols]].mean().to_numpy()
    mass = texts[[c.replace("h_", "hmass_") for c in cols]].mean().to_numpy() if "hmass_L00_H00" in texts else None
    other = y[cls == "other"].mean()
    out = []
    for c in ["causal", "contrast", "illustration", "condition", "reader"]:
        sel = y[cls == c]
        if len(sel) < 5:
            continue
        pref = (sel.mean() - other).to_numpy()
        active = sink < MAX_HEAD_SINK  # heads that mostly idle on the sink give noisy ratios
        order = np.argsort(-np.where(active, pref, -np.inf))[:6]
        for k in order:
            layer, head = int(cols[k][3:5]), int(cols[k][7:9])
            out.append({"cls": c, "layer": layer + 1, "head": head + 1, "preference": round(float(pref[k]), 3),
                        "x": round(float(2 ** pref[k]), 2), "sink": round(float(sink[k]), 3),
                        "mass": round(float(mass[k]), 3) if mass is not None else None, "n_words": int(len(sel))})
    return out


def pairs() -> dict | None:
    run = HERE / "output" / "pairs5" / "Qwen3-4B-Base"
    grad = HERE / "output" / "pairs5_grad" / "Qwen3-4B-Base"
    occ_dir = HERE / "output" / "occlusion" / "Qwen3-4B-Base"
    if not list(run.glob("texts-*.parquet")):
        return None
    texts = pd.concat(pd.read_parquet(p) for p in sorted(run.glob("texts-*.parquet")))
    res = {"model": "Qwen/Qwen3-4B-Base", "n_texts": len(texts)}

    # Sink share per layer under each measure.
    res["sink_by_layer"] = {"Attention": layer_series(texts, "sink"), "Value-weighted": layer_series(texts, "vsink")}
    gtexts = None
    if list(grad.glob("texts-*.parquet")):
        gtexts = pd.concat(pd.read_parquet(p) for p in sorted(grad.glob("texts-*.parquet")))
        res["sink_by_layer"]["Attention × gradient"] = layer_series(gtexts, "gsink")
        res["loss"] = records(gtexts[["answer_id", "question_id", "category", "score_group", "answer_score", "loss_bits"]]
                              .set_index("answer_id"))
    res["qshare"] = {"Attention": layer_series(texts, "qshare"), "Value-weighted": layer_series(texts, "vqshare")}
    if gtexts is not None:
        res["qshare"]["Attention × gradient"] = layer_series(gtexts, "gqshare")

    # Word-level tables per measure.
    per_measure, merged = {}, None
    for m, label in MEASURES.items():
        src = grad if m == "grel" else run
        if not list(src.glob("tokens-*.parquet")):
            continue
        w = add_measure(load_words(src, DATA, m))
        per_measure[label] = w
        cols = w[["answer_id", "word_index", "att"]].rename(columns={"att": label})
        merged = cols if merged is None else merged.merge(cols, on=["answer_id", "word_index"], how="outer")
    base = per_measure["Attention"]
    merged = base[["answer_id", "word_index", "word_norm", "cls", "pos", "score_group"]].merge(merged, on=["answer_id", "word_index"])

    occ_files = sorted(occ_dir.glob("*.csv"))
    if occ_files:
        occ = pd.concat(pd.read_csv(f, encoding="utf-8-sig") for f in occ_files)
        # Most of the total falls on the very next token (the model reacting to the gap);
        # tokens 2-20 are closer to "lost content".
        occ["Occlusion"] = occ["delta_bits"]
        occ[OCC_LATER] = occ["delta_bits"] - occ["delta_next_token_bits"]
        merged = merged.merge(occ[["answer_id", "word_index", "Occlusion", OCC_LATER]],
                              on=["answer_id", "word_index"], how="left")
        res["n_occluded_texts"] = int(occ["answer_id"].nunique())
        res["occlusion_next_share"] = round(float(occ["delta_next_token_bits"].sum() / occ["delta_bits"].sum()), 3)
    measures = [c for c in list(MEASURES.values()) + ["Occlusion", OCC_LATER] if c in merged]

    # Agreement between measures (Spearman over words).
    corr = merged[measures].corr(method="spearman")
    res["agreement"] = {"measures": measures, "matrix": corr.round(3).values.tolist(),
                        "n_words": {m: int(merged[m].notna().sum()) for m in measures}}

    # Word classes: mean per measure, by score group.
    rows = []
    for c in CLASS_ORDER:
        sub = merged[merged["cls"] == c]
        row = {"cls": c, "n_highest": int((sub["score_group"] == "highest").sum()),
               "n_lowest": int((sub["score_group"] == "lowest").sum())}
        for m in measures:
            for g in ["highest", "lowest"]:
                v = sub.loc[sub["score_group"] == g, m].mean()
                row[f"{m}|{g}"] = None if pd.isna(v) else round(float(v), 3)
        rows.append(row)
    res["classes"] = {"measures": measures, "rows": rows}

    # Most attended / most important words (seen at least twice), per measure.
    top = {}
    for m in measures:
        g = merged.dropna(subset=[m]).groupby("word_norm")[m].agg(["mean", "count"])
        g = g[(g["count"] >= 2) & (g.index != "")].sort_values("mean", ascending=False).head(15)
        top[m] = [{"word": k, "value": round(float(v), 3), "n": int(n)} for k, (v, n) in g.iterrows()]
    res["top_words"] = top

    res["heads"] = heads_preference(run, base, texts)
    return res


def grad_classes() -> dict | None:
    """Controlled word-class analysis of the attention x gradient run (analyze_classes.py)."""
    f = HERE / "output" / "grad1k" / "Qwen3-4B-Base" / "analysis_classes" / "summary.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text(encoding="utf-8"))
    s["model"] = "Qwen/Qwen3-4B-Base"
    return s


def focus() -> dict | None:
    """Data for attention/focus/index.html: the attention x gradient word-class analysis,
    plus the same controlled model for the 1.7B plain-attention run on the same 200 texts."""
    run = HERE / "output" / "grad1k" / "Qwen3-4B-Base"
    main_f = run / "analysis_classes" / "summary.json"
    if not main_f.exists():
        return None
    load = lambda f: json.loads(f.read_text(encoding="utf-8")) if f.exists() else None  # noqa: E731
    return {
        "main": load(main_f),
        "same200_4b": load(run / "analysis_classes_first200" / "summary.json"),
        "same200_17b": load(HERE / "output" / "sample200" / "Qwen3-1.7B-Base" / "analysis_classes" / "summary.json"),
        "target_texts": 1000,
        "updated": pd.Timestamp.now().strftime("%d %b %Y, %H:%M"),
    }


def main() -> None:
    f = focus()
    if f:
        out = HERE / "focus" / "data.js"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("window.FOCUS = " + json.dumps(f, ensure_ascii=False) + ";\n", encoding="utf-8")
        print(f"wrote {out}")
    data = {"grad_classes": grad_classes(), "sample200": sample200(), "pairs": pairs()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.RESULTS = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB); sections: "
          + ", ".join(k for k, v in data.items() if v))


if __name__ == "__main__":
    main()
