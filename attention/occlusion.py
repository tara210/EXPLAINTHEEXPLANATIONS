"""Occlusion: how much does each word of an explanation help predict what follows?

For a few explanations (default: the highest- and the lowest-scored answer to the first
question in the dataset) every answer word is removed in turn, and we measure how much
harder the next --window tokens become for the model to predict:

    delta_bits = sum over the next tokens of  surprisal(without the word) - surprisal(original)

(surprisal = -log2 p(token | text before it)). Positive = the word helped the model
predict the continuation; around 0 = the model did not need it. The continuation is
tokenised separately so it is exactly the same tokens in both versions.

For comparison each word also gets the attention it receives (same measure as
extract_attention.py: all layers, heads averaged, sink removed, 1.0 = even spread).

Writes attention/output/occlusion/<model>/<answer_id>.csv and prints a short summary.

Usage:
  python attention/occlusion.py
  python attention/occlusion.py --answers dbzu7je dc02hpt --window 30
  python attention/occlusion.py --answers-file attention/pairs5.txt --model Qwen/Qwen3-4B-Base
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_attention import PROMPT, attention_stats, question_span, read_ids  # noqa: E402
from word_classes import tag_words  # noqa: E402

LN2 = float(np.log(2))


@torch.no_grad()
def surprisal(model, ids: list[int], attentions: bool = False):
    """Surprisal in bits of every token given the ones before it (position 0 gets nan)."""
    x = torch.tensor([ids])
    out = model(input_ids=x, use_cache=False, output_attentions=attentions)
    logp = torch.log_softmax(out.logits[0, :-1].float(), -1)
    s = -logp.gather(1, x[0, 1:, None])[:, 0] / LN2
    return torch.cat([torch.tensor([float("nan")]), s]).numpy(), out


@torch.no_grad()
def batch_surprisal(model, seqs: list[list[int]], pad_id: int) -> list[np.ndarray]:
    """surprisal() for several sequences at once. Right padding is safe in a causal model:
    padding sits after every real token, so no real token can see it."""
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), pad_id)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for k, s in enumerate(seqs):
        x[k, : len(s)] = torch.tensor(s)
        mask[k, : len(s)] = 1
    logp = torch.log_softmax(model(input_ids=x, attention_mask=mask, use_cache=False).logits[:, :-1].float(), -1)
    s = -logp.gather(2, x[:, 1:, None])[..., 0] / LN2
    out = torch.cat([torch.full((len(seqs), 1), float("nan")), s], 1).numpy()
    return [out[k, : len(q)] for k, q in enumerate(seqs)]


def set_attention(model, impl: str) -> None:
    """Eager attention returns weights (needed once per text); SDPA is much faster otherwise."""
    model.set_attn_implementation(impl)


def occlude_text(row, tok, model, window: int, batch: int = 8) -> pd.DataFrame:
    question = row.question_title.strip()
    prefix = PROMPT.format(question=question)
    text = prefix + row.answer_text
    enc = tok(text, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    # Original text: surprisal of every token, plus attention received per token.
    set_attention(model, "eager")
    base, out = surprisal(model, ids, attentions=True)
    answer_start = next(i for i, (_, e) in enumerate(offsets) if e > len(prefix))
    q0 = prefix.index(question)
    recv, _ = attention_stats(out.attentions, answer_start, question_span(offsets, q0, q0 + len(question)))
    recv = recv.mean(1)  # all layers
    del out
    set_attention(model, "sdpa")

    # Build every occluded variant first (plus a separate original where the split
    # tokenises differently from the full text), then run them in batches.
    jobs, seqs = [], []
    for w in tag_words(row.answer_text):
        s, e = len(prefix) + w["start"], len(prefix) + w["end"]
        cont = text[e:]
        if not cont.strip():
            continue
        cont_ids = tok(cont)["input_ids"][:window]
        ctx_orig = tok(text[:e])["input_ids"]
        ctx_occ = tok(text[:s].rstrip(" "))["input_ids"]
        n = len(ctx_orig)
        job = {"w": w, "s": s, "e": e, "n_cont": len(cont_ids), "occ": (len(seqs), len(ctx_occ))}
        seqs.append(ctx_occ + cont_ids)
        if ids[: n + len(cont_ids)] == ctx_orig + cont_ids:
            job["orig"] = base[n : n + len(cont_ids)]  # reuse the original pass
        else:
            job["orig_seq"] = (len(seqs), n)
            seqs.append(ctx_orig + cont_ids)
        jobs.append(job)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    results = []
    for k in range(0, len(seqs), batch):
        results += batch_surprisal(model, seqs[k : k + batch], pad)

    rows = []
    for job in jobs:
        w, s, e, m = job["w"], job["s"], job["e"], job["n_cont"]
        k, start = job["occ"]
        occ = results[k][start : start + m]
        if "orig" in job:
            orig = job["orig"]
        else:
            k2, n = job["orig_seq"]
            orig = results[k2][n : n + m]
        word_tokens = [i for i, (a, b) in enumerate(offsets) if a < e and b > s and i >= answer_start]
        rows.append({
            "answer_id": row.answer_id, "question_id": row.question_id, "score_group": row.score_group,
            **{k: w[k] for k in ["word_index", "word", "word_norm", "pos", "cls"]},
            "delta_bits": float(occ.sum() - orig.sum()),
            "delta_next_token_bits": float(occ[0] - orig[0]),
            "n_window": m,
            "recv": float(np.nanmean(recv[word_tokens])) if word_tokens else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--data", type=Path, default=HERE.parent / "DATASET" / "eli5c_multi_1k.csv")
    p.add_argument("--answers", nargs="*", help="answer_ids (default: highest + lowest of the first question)")
    p.add_argument("--answers-file", type=Path, help="answer_ids from a file, e.g. attention/pairs5.txt")
    p.add_argument("--window", type=int, default=20, help="continuation tokens to score (default 20)")
    p.add_argument("--out", type=Path, default=HERE / "output" / "occlusion")
    args = p.parse_args()
    args.out = args.out / Path(args.model).name

    df = pd.read_csv(args.data, encoding="utf-8-sig")
    if args.answers_file:
        args.answers = read_ids(args.answers_file)
    if args.answers:
        todo = df.set_index("answer_id").loc[args.answers].reset_index()
    else:
        first_q = df[df["question_id"] == df["question_id"].iloc[0]]
        todo = first_q[first_q["score_group"].isin(["highest", "lowest"])]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    args.out.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 160)
    results = []
    for k, row in enumerate(todo.itertuples(index=False), 1):
        path = args.out / f"{row.answer_id}.csv"
        if path.exists():  # finished in an earlier run
            results.append(pd.read_csv(path, encoding="utf-8-sig"))
            print(f"\n=== {row.answer_id}: already done, skipped")
            continue
        t0 = time.time()
        res = occlude_text(row, tok, model, args.window)
        res.to_csv(path, index=False, encoding="utf-8-sig")
        results.append(res)
        print(f"\n[{k}/{len(todo)}]", end="")
        print(f"\n=== {row.answer_id} ({row.score_group}, score {row.answer_score}, {len(res)} words, "
              f"{time.time() - t0:.0f} s)\nQ: {row.question_title}")
        print("words whose removal hurts prediction most (bits over the next "
              f"{args.window} tokens):")
        print(res.nlargest(12, "delta_bits")[["word", "pos", "cls", "delta_bits", "recv"]]
              .round(2).to_string(index=False))

    allres = pd.concat(results)
    print("\nmean effect of removing a word, by class (both texts):")
    print(allres.groupby("cls")["delta_bits"].agg(["mean", "median", "count"]).round(2)
          .sort_values("mean", ascending=False).to_string())
    print("\nmean effect by part of speech (>= 5 words):")
    pos = allres.groupby("pos")["delta_bits"].agg(["mean", "median", "count"])
    print(pos[pos["count"] >= 5].round(2).sort_values("mean", ascending=False).to_string())
    rho = allres[["delta_bits", "recv"]].corr(method="spearman").iloc[0, 1]
    print(f"\ndoes attention agree with occlusion? Spearman rho(delta_bits, attention received) = {rho:.2f}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
