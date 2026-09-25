"""Extract attention patterns from Qwen3-1.7B-Base for the ELI5 explanations.

For every explanation in DATASET/eli5c_multi_1k.csv the model reads

    Question: <question title>
    Answer: <explanation>

in a single forward pass (no generation), and we record per layer (heads averaged):

  tokens-*.parquet   one row per answer token
      recv_L00..L27  attention the token *receives* from later tokens, as a ratio to
                     uniform attention (1.0 = average, 3.0 = three times the average).
                     Normalising by position matters: in a causal model query i spreads
                     its attention over i tokens, so early tokens would otherwise always
                     look more important.
      word, word_norm, word_index   the whitespace word the token belongs to

  texts-*.parquet    one row per explanation
      qshare_L00..L27  share of the answer tokens' attention that goes to the question text
                       (the "Question:" / "Answer:" template tokens are not counted)

      sink_L00..L27    share of the answer tokens' attention on the first token (sink)

With --value-norm (Kobayashi et al. 2020), attention is weighted by the size of what the
attended token actually passes on through each head, ||W_O,h v_j||, and summed over heads:
      vrecv_L..  (tokens)   received, as recv_ but value-weighted
      vqshare_L.., vsink_L..  (texts)  question and sink share, value-weighted

With --heads, per-head data (heads not averaged):
  heads-*.parquet    one row per answer token: h_Lxx_Hyy = received ratio in layer xx, head yy
  texts: hsink_Lxx_Hyy (share on the sink; ~1 = the head mostly idles) and, with
         --value-norm, hmass_Lxx_Hyy (how much the head writes, sink excluded)

The first token of every text ("Question") works as an attention sink: models dump a
large part of their attention there regardless of content. It is removed and each row
of attention is renormalised before the received ratios and question shares are computed.

Results are written in chunks, so a run can be stopped and restarted; texts already
done are skipped. At the end (or with --summarize) a short word-level summary is printed
and written to word_attention.csv.

Usage:
  python attention/extract_attention.py --limit 20          # quick test
  python attention/extract_attention.py --sample 200        # ~200, whole questions, all categories
  python attention/extract_attention.py --sample 200 --value-norm --heads
  python attention/extract_attention.py --answers-file attention/pairs5.txt --model Qwen/Qwen3-4B-Base --value-norm --heads
  python attention/extract_attention.py                     # all ~3,900 explanations
  python attention/extract_attention.py --summarize         # summary of what is done
"""

import argparse
import bisect
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "DATASET" / "eli5c_multi_1k.csv"
DEFAULT_OUT = ROOT / "attention" / "output"
PROMPT = "Question: {question}\nAnswer: "
WORD = re.compile(r"\S+")
EDGE_PUNCT = re.compile(r"^[^\w']+|[^\w']+$")
META_COLS = ["answer_id", "question_id", "category", "score_group", "answer_score", "relative_score"]
# Words often associated with explaining; reported separately in the summary.
MARKERS = ["because", "so", "since", "therefore", "means", "imagine", "like", "think",
           "basically", "example", "which", "when", "if", "why", "how", "actually"]


def word_spans(text: str, start: int):
    spans = [(m.start(), m.end(), m.group()) for m in WORD.finditer(text, start)]
    return [s for s, _, _ in spans], spans


def word_of(char_end: int, starts, spans):
    """Index of the whitespace word containing the token's last character, or -1."""
    k = bisect.bisect_right(starts, char_end - 1) - 1
    if k >= 0 and char_end - 1 < spans[k][1]:
        return k
    return -1


def question_span(offsets, question_start: int, question_end: int):
    """Token range [q0, q1) of the question text itself, without the "Question:" / "Answer:" template."""
    q0 = next(i for i, (_, e) in enumerate(offsets) if e > question_start)
    q1 = next(i for i, (s, _) in enumerate(offsets) if s >= question_end)
    return q0, q1


@torch.no_grad()
def received(A: torch.Tensor, answer_start: int, question: tuple):
    """From attention rows A (..., T, T) including the sink column 0: drop the sink, rescale
    rows to 1, and return (received ratio per token (..., T), question share (...))."""
    T = A.shape[-1]
    A = A.clone()
    A[..., 0] = 0
    A = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
    n_visible = torch.arange(T, dtype=A.dtype)  # query i sees keys 1..i after removing the sink
    receivers = torch.tril(torch.ones(T, T), diagonal=-1)  # only later tokens (i > j)
    receivers[:, 0] = 0
    n_receivers = receivers.sum(0)
    ratio = A * n_visible[:, None]  # 1.0 = uniform attention for that query
    r = (ratio * receivers).sum(-2) / n_receivers.clamp_min(1)
    r[..., n_receivers == 0] = float("nan")  # the last token has no later tokens
    q = A[..., answer_start:, question[0]:question[1]].sum(-1).mean(-1)
    return r, q


@torch.no_grad()
def attention_stats(attentions, answer_start: int, question: tuple):
    """Position-normalised attention received per token, and question share, per layer
    (heads averaged)."""
    recv, qshare = zip(*(received(layer[0].float().mean(0), answer_start, question) for layer in attentions))
    return torch.stack(recv, 1).numpy(), torch.stack(qshare).numpy()


def value_grams(model) -> list[torch.Tensor]:
    """Per layer and query head h: G_h = W_O,h^T W_O,h, so that ||W_O,h v|| = sqrt(v^T G_h v).
    W_O,h is the slice of the output projection that maps head h back into the residual stream."""
    cfg = model.config
    grams = []
    for layer in model.model.layers:
        W = layer.self_attn.o_proj.weight.detach().float()  # (d_model, H * d_head)
        W = W.view(W.shape[0], cfg.num_attention_heads, cfg.head_dim)
        grams.append(torch.einsum("Dhd,Dhe->hde", W, W))
    return grams


@torch.no_grad()
def value_norms(v: torch.Tensor, gram: torch.Tensor, head_dim: int) -> torch.Tensor:
    """||W_O,h v_j||: size of what token j would write into the residual stream through
    query head h (H, T). v are the key/value-head outputs of v_proj (1, T, KV * d_head)."""
    T = v.shape[1]
    kv = v[0].float().view(T, -1, head_dim)
    per_query_head = kv.repeat_interleave(gram.shape[0] // kv.shape[1], dim=1)  # grouped-query attention
    return torch.einsum("thd,hde,the->ht", per_query_head, gram, per_query_head).clamp_min(0).sqrt()


@torch.no_grad()
def layer_measures(att: torch.Tensor, vnorm, answer_start: int, question: tuple, heads: bool) -> dict:
    """All measures for one layer. att: (H, T, T) attention of the layer, vnorm: (H, T) or None."""
    out = {}
    A = att.float()
    out["sink"] = A.mean(0)[answer_start:, 0].mean()
    if vnorm is not None:
        # Kobayashi et al. 2020: weight attention by the size of what the attended token passes on,
        # summed over heads (heads add up in the residual stream), rows rescaled to 1.
        W = (A * vnorm[:, None, :]).sum(0)
        V = W / W.sum(-1, keepdim=True).clamp_min(1e-12)
        out["vsink"] = V[answer_start:, 0].mean()
        out["vrecv"], out["vqshare"] = received(V, answer_start, question)
    if heads:
        out["hrecv"], _ = received(A, answer_start, question)  # (H, T)
        out["hsink"] = A[:, answer_start:, 0].mean(-1)  # (H,) heads that mostly idle on the sink
        if vnorm is not None:
            # how much the head writes per query, sink excluded: low = head contributes little
            out["hmass"] = (A[:, answer_start:, 1:] * vnorm[:, None, 1:]).sum(-1).mean(-1)
    return out


@torch.no_grad()
def process(row, tok, model, max_tokens: int, values=None, grams=None, heads: bool = False):
    question = row.question_title.strip()
    prefix = PROMPT.format(question=question)
    text = prefix + row.answer_text
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt",
              truncation=True, max_length=max_tokens)
    offsets = enc.pop("offset_mapping")[0].tolist()
    # First token that reaches into the answer (its leading space may belong to the prefix).
    answer_start = next(i for i, (_, e) in enumerate(offsets) if e > len(prefix))
    q_start = prefix.index(question)
    q_span = question_span(offsets, q_start, q_start + len(question))

    out = model(**enc, output_attentions=True, use_cache=False)
    recv, qshare = attention_stats(out.attentions, answer_start, q_span)
    n_layers = recv.shape[1]
    head_dim = model.config.head_dim
    extra = [layer_measures(out.attentions[l][0],
                            value_norms(values[l], grams[l], head_dim) if grams is not None else None,
                            answer_start, q_span, heads)
             for l in range(n_layers)] if (grams is not None or heads) else []
    del out

    meta = {c: getattr(row, c) for c in META_COLS}
    starts, spans = word_spans(text, len(prefix))
    tokens, head_rows = [], []
    for pos in range(answer_start, len(offsets)):
        s, e = offsets[pos]
        w = word_of(e, starts, spans)
        word = spans[w][2] if w >= 0 else ""
        tok_row = {
            **meta, "position": pos, "token": text[s:e], "word_index": w,
            "word": word, "word_norm": EDGE_PUNCT.sub("", word.lower()),
            **{f"recv_L{l:02d}": float(recv[pos, l]) for l in range(n_layers)},
        }
        if grams is not None:
            tok_row.update({f"vrecv_L{l:02d}": float(extra[l]["vrecv"][pos]) for l in range(n_layers)})
        tokens.append(tok_row)
        if heads:
            head_rows.append({
                "answer_id": row.answer_id, "position": pos, "word_index": w,
                **{f"h_L{l:02d}_H{h:02d}": float(extra[l]["hrecv"][h, pos])
                   for l in range(n_layers) for h in range(extra[l]["hrecv"].shape[0])},
            })

    text_row = {
        **meta, "n_tokens": len(offsets), "n_answer_tokens": len(offsets) - answer_start,
        "truncated": len(offsets) >= max_tokens,
        **{f"qshare_L{l:02d}": float(qshare[l]) for l in range(n_layers)},
    }
    for l, m in enumerate(extra):
        text_row[f"sink_L{l:02d}"] = float(m["sink"])
        if grams is not None:
            text_row[f"vqshare_L{l:02d}"] = float(m["vqshare"])
            text_row[f"vsink_L{l:02d}"] = float(m["vsink"])
        if heads:
            for h in range(m["hsink"].shape[0]):
                text_row[f"hsink_L{l:02d}_H{h:02d}"] = float(m["hsink"][h])
                if "hmass" in m:
                    text_row[f"hmass_L{l:02d}_H{h:02d}"] = float(m["hmass"][h])
    return tokens, text_row, head_rows


def balanced_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Whole questions, taking turns across categories, until adding one would exceed n rows.
    Rows come back in that turn order, so any prefix (e.g. the first 100 done) is balanced too."""
    qs = df.drop_duplicates("question_id")[["question_id", "category"]].sample(frac=1, random_state=seed)
    qs["turn"] = qs.groupby("category").cumcount()
    sizes = df.groupby("question_id").size()
    picked, total = [], 0
    for q in qs.sort_values(["turn", "category"])["question_id"]:
        if total + sizes[q] > n:
            break
        picked.append(q)
        total += sizes[q]
    order = {q: k for k, q in enumerate(picked)}
    out = df[df["question_id"].isin(picked)]
    return out.iloc[sorted(range(len(out)), key=lambda i: (order[out["question_id"].iloc[i]], i))]


def read_ids(path: Path) -> list[str]:
    """answer_ids from a text file, one per line; '#' starts a comment."""
    lines = (l.split("#")[0].strip() for l in path.read_text(encoding="utf-8").splitlines())
    return [l for l in lines if l]


def add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--limit", type=int, default=0, help="only the first N explanations (0 = all)")
    p.add_argument("--sample", type=int, default=0,
                   help="up to N explanations from whole questions, spread over categories (0 = off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--answers-file", type=Path, help="only these answer_ids (e.g. attention/pairs5.txt)")


def select_rows(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.answers_file:
        ids = read_ids(args.answers_file)
        missing = set(ids) - set(df["answer_id"])
        if missing:
            raise SystemExit(f"answer_ids not in the data: {sorted(missing)}")
        df = df.set_index("answer_id").loc[ids].reset_index()  # keep the file's order
    if args.sample:
        df = balanced_sample(df, args.sample, args.seed)
    if args.limit:
        df = df.head(args.limit)
    return df


def done_ids(out_dir: Path) -> set:
    parts = sorted(out_dir.glob("texts-*.parquet"))
    if not parts:
        return set()
    return set(pd.concat(pd.read_parquet(p, columns=["answer_id"]) for p in parts)["answer_id"])


def flush(out_dir: Path, tokens: list, texts: list, head_rows: list) -> None:
    n = len(list(out_dir.glob("texts-*.parquet")))
    pd.DataFrame(tokens).to_parquet(out_dir / f"tokens-{n:04d}.parquet", index=False)
    if head_rows:
        h = pd.DataFrame(head_rows)
        cols = [c for c in h.columns if c.startswith("h_L")]
        h[cols] = h[cols].astype("float16")  # 448 columns per token; half precision is plenty
        h.to_parquet(out_dir / f"heads-{n:04d}.parquet", index=False)
    # texts last: its presence marks the chunk as done (see done_ids)
    pd.DataFrame(texts).to_parquet(out_dir / f"texts-{n:04d}.parquet", index=False)


def summarize(out_dir: Path, min_count: int) -> None:
    tokens = pd.concat(pd.read_parquet(p) for p in sorted(out_dir.glob("tokens-*.parquet")))
    texts = pd.concat(pd.read_parquet(p) for p in sorted(out_dir.glob("texts-*.parquet")))
    layer_cols = [c for c in tokens.columns if c.startswith("recv_L")]
    tokens["recv"] = tokens[layer_cols].mean(axis=1)

    # Token -> word (mean over sub-word tokens), then word type across all texts.
    words = (tokens[tokens["word_index"] >= 0]
             .groupby(["answer_id", "word_index"])
             .agg(word_norm=("word_norm", "first"), score_group=("score_group", "first"),
                  recv=("recv", "mean")))
    words = words[words["word_norm"] != ""]
    per_word = (words.groupby("word_norm")["recv"]
                .agg(["mean", "count"]).rename(columns={"mean": "recv_mean"}))
    per_word["n_texts"] = words.reset_index().groupby("word_norm")["answer_id"].nunique()
    per_word = per_word.sort_values("recv_mean", ascending=False)
    per_word.to_csv(out_dir / "word_attention.csv", encoding="utf-8-sig")

    frequent = per_word[per_word["count"] >= min_count]
    print(f"\n{texts['answer_id'].nunique()} explanations, {len(words)} words, "
          f"{len(frequent)} word types seen >= {min_count} times")
    print("\nwords receiving the most attention (all layers averaged, 1.0 = uniform):")
    print(frequent.head(30).round(3).to_string())

    print("\nexplanation markers:")
    print(per_word.reindex(MARKERS).dropna().round(3).to_string())

    q_cols = [c for c in texts.columns if c.startswith("qshare_L")]
    texts["qshare"] = texts[q_cols].mean(axis=1)
    print("\nshare of attention on the question, by score group:")
    print(texts.groupby("score_group")["qshare"].agg(["mean", "std", "count"]).round(4).to_string())
    print(f"\nfull word table: {out_dir / 'word_attention.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    add_selection_args(p)
    p.add_argument("--max-tokens", type=int, default=768, help="truncate longer texts (default 768)")
    p.add_argument("--chunk", type=int, default=50, help="write results every N texts (default 50)")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    p.add_argument("--min-count", type=int, default=30, help="min occurrences for the word summary")
    p.add_argument("--summarize", action="store_true", help="only print the summary of existing results")
    p.add_argument("--value-norm", action="store_true",
                   help="also attention weighted by the size of what each token passes on (vrecv_*, vqshare_*, vsink_*)")
    p.add_argument("--heads", action="store_true",
                   help="also per-head received attention (heads-*.parquet) and per-head sink/mass per text")
    args = p.parse_args()

    out_dir = args.out / Path(args.model).name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(out_dir, args.min_count)
        return

    df = select_rows(pd.read_csv(args.data, encoding="utf-8-sig"), args)
    skip = done_ids(out_dir)
    todo = df[~df["answer_id"].isin(skip)]
    print(f"{len(todo)} explanations to do ({len(skip)} already done) -> {out_dir}")
    if todo.empty:
        summarize(out_dir, args.min_count)
        return

    if args.threads:
        torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model)
    # float32: bfloat16 matmuls are slow on CPUs without native bf16 support.
    # eager attention: the fast SDPA kernels do not return attention weights.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager")
    model.eval()

    values, grams = {}, None
    if args.value_norm:
        grams = value_grams(model)
        for l, layer in enumerate(model.model.layers):
            layer.self_attn.v_proj.register_forward_hook(
                lambda mod, inp, out, l=l: values.__setitem__(l, out.detach()))

    tokens, texts, head_rows = [], [], []
    t0 = time.time()
    for k, row in enumerate(todo.itertuples(index=False), 1):
        tk, tx, hr = process(row, tok, model, args.max_tokens, values, grams, args.heads)
        tokens += tk
        texts.append(tx)
        head_rows += hr
        if len(texts) == args.chunk or k == len(todo):
            flush(out_dir, tokens, texts, head_rows)
            tokens, texts, head_rows = [], [], []
        rate = (time.time() - t0) / k
        print(f"\r{k}/{len(todo)}  {rate:.1f} s/text  ~{rate * (len(todo) - k) / 60:.0f} min left",
              end="", flush=True)
    print()
    summarize(out_dir, args.min_count)


if __name__ == "__main__":
    main()
