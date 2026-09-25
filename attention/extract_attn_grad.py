"""Attention x gradient: which attention edges actually help the model predict the explanation?

For each explanation the model reads "Question: ...\nAnswer: ..." and we take the loss
L = mean surprisal of the answer tokens. One backward pass gives dL/dA for every attention
weight A. An edge's relevance is

    R_ij = mean over heads of max(0, A_ij * -dL/dA_ij)

i.e. attention that, if strengthened, would make the explanation easier to predict
(positive part only, following Chefer et al. 2021). Per token j we sum the relevance of
all edges into it from later tokens (sink column excluded), per layer:

  tokens-*.parquet   grel_L00..L27  relevance received (sum over later queries)
                     plus word, word_index, ... as in extract_attention.py
  texts-*.parquet    gqshare_L..    share of the answer's relevance that goes to the question text
                     gsink_L..      share of all relevance that lands on the sink
                     loss_bits      mean surprisal of the answer tokens (bits)

Relevance is not a probability distribution, so there is no "1.0 = even spread"; compare
tokens within a text (analyze_words.py centres per text and position).

Costs about 3x a plain attention pass (one backward). Default: 100 explanations from whole
questions spread over the categories (the same sample as --sample in extract_attention.py,
so a subset of the 200 sample).

Usage:
  python attention/extract_attn_grad.py                     # 100-explanation sample
  python attention/extract_attn_grad.py --answers-file attention/pairs5.txt --model Qwen/Qwen3-4B-Base --out attention/output/pairs5
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
from extract_attention import (EDGE_PUNCT, META_COLS, PROMPT, add_selection_args,  # noqa: E402
                               done_ids, question_span, select_rows, word_of, word_spans)

LN2 = float(np.log(2))


def relevance_stats(attentions, answer_start: int, question: tuple):
    T = attentions[0].shape[-1]
    receivers = torch.tril(torch.ones(T, T), diagonal=-1)
    receivers[:, 0] = 0
    grel, gq, gs = [], [], []
    for a in attentions:
        R = (a.detach()[0] * -a.grad[0]).clamp_min(0).mean(0)  # (T, T), heads averaged
        total = R[answer_start:].sum()
        gs.append(R[answer_start:, 0].sum() / total.clamp_min(1e-30))
        R[:, 0] = 0
        grel.append((R * receivers).sum(0))
        gq.append(R[answer_start:, question[0]:question[1]].sum() / R[answer_start:].sum().clamp_min(1e-30))
    return torch.stack(grel, 1).numpy(), torch.stack(gq).numpy(), torch.stack(gs).numpy()


def process(row, tok, model, max_tokens: int):
    question = row.question_title.strip()
    prefix = PROMPT.format(question=question)
    text = prefix + row.answer_text
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=max_tokens)
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"]
    answer_start = next(i for i, (_, e) in enumerate(offsets) if e > len(prefix))
    q0 = prefix.index(question)
    q_span = question_span(offsets, q0, q0 + len(question))

    # Gradients flow from the input embeddings; the weights themselves need none.
    emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    out = model(inputs_embeds=emb, attention_mask=enc["attention_mask"], output_attentions=True, use_cache=False)
    for a in out.attentions:
        a.retain_grad()
    logp = torch.log_softmax(out.logits[0, answer_start - 1:-1].float(), -1)
    loss = -logp.gather(1, ids[0, answer_start:, None]).mean()
    loss.backward()
    grel, gq, gs = relevance_stats(out.attentions, answer_start, q_span)
    n_layers = grel.shape[1]
    del out, emb

    meta = {c: getattr(row, c) for c in META_COLS}
    starts, spans = word_spans(text, len(prefix))
    tokens = []
    for pos in range(answer_start, len(offsets)):
        s, e = offsets[pos]
        w = word_of(e, starts, spans)
        word = spans[w][2] if w >= 0 else ""
        tokens.append({
            **meta, "position": pos, "token": text[s:e], "word_index": w,
            "word": word, "word_norm": EDGE_PUNCT.sub("", word.lower()),
            # the last token receives nothing (no later tokens), like recv_ in extract_attention.py
            **{f"grel_L{l:02d}": float(grel[pos, l]) if pos < len(offsets) - 1 else float("nan")
               for l in range(n_layers)},
        })
    text_row = {
        **meta, "n_tokens": len(offsets), "n_answer_tokens": len(offsets) - answer_start,
        "loss_bits": float(loss.detach()) / LN2,
        **{f"gqshare_L{l:02d}": float(gq[l]) for l in range(n_layers)},
        **{f"gsink_L{l:02d}": float(gs[l]) for l in range(n_layers)},
    }
    return tokens, text_row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--data", type=Path, default=HERE.parent / "DATASET" / "eli5c_multi_1k.csv")
    p.add_argument("--out", type=Path, default=HERE / "output" / "attgrad100")
    add_selection_args(p)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--chunk", type=int, default=25)
    args = p.parse_args()

    out_dir = args.out / Path(args.model).name
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (args.sample or args.limit or args.answers_file):
        args.sample = 100
    df = select_rows(pd.read_csv(args.data, encoding="utf-8-sig"), args)
    skip = done_ids(out_dir)
    todo = df[~df["answer_id"].isin(skip)]
    print(f"{len(todo)} explanations to do ({len(skip)} already done) -> {out_dir}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager")
    model.eval().requires_grad_(False)

    tokens, texts = [], []
    t0 = time.time()
    for k, row in enumerate(todo.itertuples(index=False), 1):
        tk, tx = process(row, tok, model, args.max_tokens)
        tokens += tk
        texts.append(tx)
        if len(texts) == args.chunk or k == len(todo):
            n = len(list(out_dir.glob("texts-*.parquet")))
            pd.DataFrame(tokens).to_parquet(out_dir / f"tokens-{n:04d}.parquet", index=False)
            pd.DataFrame(texts).to_parquet(out_dir / f"texts-{n:04d}.parquet", index=False)
            tokens, texts = [], []
        rate = (time.time() - t0) / k
        print(f"\r{k}/{len(todo)}  {rate:.1f} s/text  ~{rate * (len(todo) - k) / 60:.0f} min left",
              end="", flush=True)
    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
