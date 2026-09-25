"""Export full attention patterns for a few explanations, for attention/viewer/index.html.

For each explanation the model reads the text twice:
  "qa"      Question: <title>\nAnswer: <explanation>
  "answer"  <explanation> only

For every token (query) we keep its attention over all earlier tokens (keys) in several
views, each a row-normalised (T, T) matrix:
  - heads averaged, for all layers together and for four layer bands
  - attention rollout across all layers (Abnar & Zuidema 2020)
  - value-weighted attention, all layers (Kobayashi et al. 2020): attention times the size
    of what the attended token passes on through each head, summed over heads
  - attention x gradient, all layers: positive part of A * -dL/dA with L the mean
    surprisal of the answer tokens (Chefer et al. 2021), heads averaged, summed over layers
The first token is the attention sink: its share is kept separately per query, and the
remaining attention is renormalised over the other tokens. Values are stored per query row
as uint8 relative to that row's maximum (plus the row maximum), to keep the file small.

If occlusion results exist (occlusion.py, same model) the per-word effect is attached
to the tokens of each word.

Each run writes one dataset file that the page picks up next to the others:
  window.ATTN_DATASETS.push({...})

Usage:
  python attention/export_viewer.py                                        # first 10 -> viewer/data.js
  python attention/export_viewer.py --answers-file attention/pairs5.txt --model Qwen/Qwen3-4B-Base \\
      --out attention/viewer/data_pairs5.js --label "5 pairs: highest vs lowest"
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from extract_attention import (PROMPT, question_span, read_ids, value_grams,  # noqa: E402
                               value_norms, word_of, word_spans)


def band_ranges(n_layers: int):
    """All layers plus four equal bands, labelled 1-based for display."""
    edges = np.linspace(0, n_layers, 5).round().astype(int)
    bands = [("All layers", 0, n_layers)]
    bands += [(f"Layers {a + 1}–{b}", a, b) for a, b in zip(edges[:-1], edges[1:])]
    return bands


def rollout(per_layer: torch.Tensor) -> torch.Tensor:
    """Attention rollout (Abnar & Zuidema 2020): how much each input token flows into each
    position after all layers, with the residual connection modelled as 0.5 * identity."""
    T = per_layer.shape[-1]
    eye = torch.eye(T)
    R = eye.clone()
    for A in per_layer:
        R = (0.5 * A + 0.5 * eye) @ R  # rows of both factors sum to 1, so R's rows do too
    return R


def pack(label: str, A: torch.Tensor) -> dict:
    """Row-normalised (T, T) view -> sink share per row + uint8 rows relative to their max."""
    T = A.shape[-1]
    A = A / A.sum(-1, keepdim=True).clamp_min(1e-30)
    sink = A[:, 0].clone()
    A[:, 0] = 0
    A = A / A.sum(-1, keepdim=True).clamp_min(1e-30)
    row_max, packed = [0.0], bytearray()
    for i in range(1, T):
        row = A[i, 1 : i + 1]
        m = float(row.max())
        row_max.append(round(m, 5))
        packed += (row / max(m, 1e-30) * 255).round().to(torch.uint8).numpy().tobytes()
    return {"label": label, "sink": [round(float(x), 4) for x in sink], "row_max": row_max,
            "rows": base64.b64encode(bytes(packed)).decode("ascii")}


def encode_condition(text: str, answer_char_start: int, question_chars, tok, model, max_tokens: int,
                     bands, grams, values, occlusion=None):
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=max_tokens)
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"]
    n_question = next(i for i, (_, e) in enumerate(offsets) if e > answer_char_start) if answer_char_start else 0
    # Tokens of the question text itself, without the "Question:" / "Answer:" template.
    q_span = question_span(offsets, *question_chars) if question_chars else (0, 0)

    # One pass with gradients: attention, value vectors (hooks) and dL/dA.
    emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    out = model(inputs_embeds=emb, attention_mask=enc["attention_mask"], output_attentions=True, use_cache=False)
    for a in out.attentions:
        a.retain_grad()
    first = max(n_question, 1)  # loss over the answer tokens
    logp = torch.log_softmax(out.logits[0, first - 1:-1].float(), -1)
    loss = -logp.gather(1, ids[0, first:, None]).mean()
    loss.backward()

    with torch.no_grad():
        att = [a.detach()[0].float() for a in out.attentions]  # (H, T, T) per layer
        per_layer = torch.stack([a.mean(0) for a in att])
        head_dim = model.config.head_dim
        vw = []
        for l, a in enumerate(att):
            W = (a * value_norms(values[l], grams[l], head_dim)[:, None, :]).sum(0)
            vw.append(W / W.sum(-1, keepdim=True).clamp_min(1e-30))
        grad_rel = sum((a.detach()[0].float() * -a.grad[0].float()).clamp_min(0).mean(0) for a in out.attentions)

        views = [(label, per_layer[a:b].mean(0)) for label, a, b in bands]
        views += [("Rollout (all layers)", rollout(per_layer)),
                  ("Value-weighted (all layers)", torch.stack(vw).mean(0)),
                  ("Attention × gradient (all layers)", grad_rel)]
        result = {"tokens": [text[s:e] for s, e in offsets], "n_question": n_question,
                  "q_start": q_span[0], "q_end": q_span[1], "loss_bits": round(float(loss) / np.log(2), 3),
                  "bands": [pack(label, A.clone()) for label, A in views]}

    # Answer word each token belongs to (-1 for question / template), for word-level statistics.
    starts, spans = word_spans(text, answer_char_start)
    result["word"] = [word_of(e, starts, spans) if i >= n_question else -1 for i, (_, e) in enumerate(offsets)]
    if occlusion is not None:
        # total effect over the next tokens, and the part that falls on the very next token
        # (mostly the model reacting to the gap itself rather than to lost content)
        for key, col in [("occlusion", "delta_bits"), ("occlusion_next", "delta_next_token_bits")]:
            vals = []
            for w in result["word"]:
                v = occlusion[col].get(w) if w >= 0 else None
                vals.append(None if v is None or np.isnan(v) else round(float(v), 3))
            result[key] = vals
    del out, emb
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--data", type=Path, default=ROOT / "DATASET" / "eli5c_multi_1k.csv")
    p.add_argument("--n", type=int, default=10, help="number of explanations (default 10)")
    p.add_argument("--answers-file", type=Path, help="these answer_ids instead, e.g. attention/pairs5.txt")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--occlusion-dir", type=Path, default=HERE / "output" / "occlusion")
    p.add_argument("--label", default=None, help="dataset name shown in the page")
    p.add_argument("--out", type=Path, default=HERE / "viewer" / "data.js")
    args = p.parse_args()

    df = pd.read_csv(args.data, encoding="utf-8-sig")
    if args.answers_file:
        df = df.set_index("answer_id").loc[read_ids(args.answers_file)].reset_index()
    else:
        df = df.head(args.n)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager")
    model.eval().requires_grad_(False)
    bands = band_ranges(model.config.num_hidden_layers)
    grams, values = value_grams(model), {}
    for l, layer in enumerate(model.model.layers):
        layer.self_attn.v_proj.register_forward_hook(lambda mod, inp, o, l=l: values.__setitem__(l, o.detach()))
    occ_dir = args.occlusion_dir / Path(args.model).name

    examples = []
    for k, row in enumerate(df.itertuples(index=False), 1):
        question = row.question_title.strip()
        prefix = PROMPT.format(question=question)
        q0 = prefix.index(question)
        occ_file = occ_dir / f"{row.answer_id}.csv"
        occ = (pd.read_csv(occ_file, encoding="utf-8-sig").set_index("word_index")
               [["delta_bits", "delta_next_token_bits"]].to_dict() if occ_file.exists() else None)
        examples.append({
            "answer_id": row.answer_id, "question_id": row.question_id, "category": row.category,
            "question": row.question_title, "score_group": row.score_group,
            "answer_score": int(row.answer_score), "answer_rank": int(row.answer_rank),
            "n_explanations": int(row.n_explanations),
            "qa": encode_condition(prefix + row.answer_text, len(prefix), (q0, q0 + len(question)),
                                   tok, model, args.max_tokens, bands, grams, values, occ),
            "answer": encode_condition(row.answer_text, 0, None, tok, model, args.max_tokens,
                                       bands, grams, values, occ),
        })
        print(f"{k}/{len(df)} {row.answer_id}: {len(examples[-1]['qa']['tokens'])} tokens"
              + (" + occlusion" if occ is not None else ""), flush=True)

    name = Path(args.model).name
    data = {"id": args.out.stem, "label": f"{name} · {args.label or f'{len(examples)} explanations'}",
            "model": args.model, "n_layers": model.config.num_hidden_layers, "examples": examples}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("(window.ATTN_DATASETS = window.ATTN_DATASETS || []).push("
                        + json.dumps(data, ensure_ascii=False) + ");\n", encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
