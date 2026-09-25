# Attention analysis

Attention patterns of `Qwen/Qwen3-1.7B-Base` while it reads the ELI5 explanations in
`DATASET/eli5c_multi_1k.csv`. Runs on CPU; the model (3.4 GB) is downloaded on first use.

| File | What it does |
|---|---|
| `extract_attention.py` | Per-token and per-explanation attention statistics for the whole dataset, saved in chunks to `output/` (restartable) plus a word-level summary |
| `extract_attn_grad.py` | Attention × gradient (Chefer et al. 2021): attention edges that make the explanation easier to predict |
| `pairs5.txt` | The 5 highest/lowest answer pairs used for the expensive analyses |
| `build_results.py` | Collects all results into `results/data.js` for `results/index.html` (results overview page) |
| `analyze_words.py` | Attention by word class and part of speech, and highest vs lowest answer of the same question (bootstrap CIs over questions) |
| `occlusion.py` | Removes each word of one or two explanations and measures how much harder the continuation gets to predict |
| `word_classes.py` | Word classes (causal, contrast, illustration, reformulation, hedge, condition, reader, writer) and spaCy POS tagging |
| `export_viewer.py` | Full token-by-token attention for a few explanations, as `viewer/data.js` |
| `viewer/index.html` | Interactive viewer: step through an explanation and see where each token looks |
| `output/`, `logs/` | Results and run logs (not in git; regenerate with the scripts) |

```bash
python attention/extract_attention.py --limit 20     # quick test
python attention/extract_attention.py --sample 200 --out attention/output/sample200
python attention/analyze_words.py                    # word classes, highest vs lowest
python attention/occlusion.py                        # 2 explanations, every word

# 5 highest/lowest pairs with Qwen3-4B-Base (value-weighted, per head, gradient, occlusion)
python attention/extract_attention.py --answers-file attention/pairs5.txt --model Qwen/Qwen3-4B-Base --value-norm --heads --out attention/output/pairs5
python attention/extract_attn_grad.py --answers-file attention/pairs5.txt --model Qwen/Qwen3-4B-Base --out attention/output/pairs5_grad
python attention/occlusion.py --answers-file attention/pairs5.txt --model Qwen/Qwen3-4B-Base
python attention/export_viewer.py --answers-file attention/pairs5.txt --model Qwen/Qwen3-4B-Base --out attention/viewer/data_pairs5.js --label "5 pairs: highest vs lowest"
python attention/build_results.py
python attention/extract_attention.py                # all explanations (hours on CPU)
python attention/extract_attention.py --summarize    # summary of finished results
python attention/export_viewer.py --n 10             # data for the viewer
```

## How attention is measured

- Attention weights come from every layer (`output_attentions=True`, eager attention) and are
  averaged over heads; the viewer also groups the 28 layers into four bands.
- The first token acts as an attention sink (it receives a large share regardless of content).
  Its share is recorded separately, then removed, and each row is rescaled to sum to 1.
- **Received attention** (extract script): how much a token is attended to by later tokens,
  relative to an even spread (1.0 = average). Without this correction early tokens always look
  more important, because a query at position *i* spreads its attention over only *i* tokens.
- **Question share**: part of an answer token's attention that goes back to the question text
  (the `Question:` / `Answer:` template tokens are not counted).

- **Value-weighted** (`--value-norm`): attention × ‖W_O,h v_j‖, the size of what the attended
  token passes on through head h, summed over heads (Kobayashi et al. 2020). The sink carries
  almost nothing: in Qwen3-4B its share drops from 60–80% to 10–14% in the upper layers.
- **Attention × gradient**: positive part of A · −∂L/∂A, L = mean surprisal of the answer.
- **Occlusion**: remove a word, measure the extra surprisal (bits) on the next 20 tokens.

Caveat: attention weights show where the model looks, not what drives its output
(Jain & Wallace 2019; Wiegreffe & Pinter 2019). Head-averaging also hides specialised heads.
