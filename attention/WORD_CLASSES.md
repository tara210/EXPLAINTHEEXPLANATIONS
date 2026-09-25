# Word classes

Word lists used to group explanation words in the attention analysis
(`attention/word_classes.py`, used by `analyze_words.py`, `build_results.py` and `occlusion.py`).

## Classes

| Class | What it marks | Words and phrases |
|---|---|---|
| **causal** | cause and effect | because, so, therefore, thus, hence, since, cause, causes, caused, causing, due to, as a result, that's why, which is why, leads to, results in |
| **contrast** | opposition, correction | but, however, although, though, whereas, instead, unlike, actually, otherwise, on the other hand |
| **illustration** | examples, analogies | like, imagine, think of, for example, for instance, example, let's say, picture, similar, similarly, kind of like, analogy |
| **reformulation** | restating, defining | basically, essentially, in other words, means, meaning, that is, i.e, called, refers to, simply put |
| **hedge** | softening, uncertainty | kind of, sort of, probably, might, maybe, perhaps, usually, generally, roughly, pretty much, mostly, typically, i think |
| **condition** | conditions, circumstances | if, when, unless, whenever, then |
| **reader** | addressing the reader | you, your, you're, yourself, you'll, you've |
| **writer** | the writer's own voice | i, i'm, my, me, i've, i'd |

Every other word is **other**.

## How words are matched

- Words are the whitespace-separated words of the answer, compared in lowercase with
  punctuation at the edges stripped (`"Because,"` → `because`, `’` treated as `'`).
- Longer phrases win: *kind of like* is illustration, not hedge + illustration.
- A word belongs to at most one class.
- **like** counts as illustration only when spaCy does not tag it as a verb
  (*works like a pump* counts, *I like it* does not).
- Phrases count once per word in all word counts: *for example* adds 2 illustration words,
  *sort of* and *i think* add 2 hedge words.
- Part of speech (`pos` column) comes from spaCy `en_core_web_sm`: the tag of the first
  non-punctuation token inside the word.

## Known ambiguities

The lists are keyword matches, so some words are counted even when they do not play that role:

| Word | Class | Also used as |
|---|---|---|
| so | causal | intensifier (*so big*) |
| since | causal | temporal (*since 2010*) |
| when | condition | temporal (*when I was young*) |
| then | condition | sequence (*and then*) |
| actually | contrast | emphasis without real contrast |
| picture, example | illustration | plain nouns (*a picture of*) |

In large samples this adds noise rather than bias; in small ones it can decide a class average.

## Occurrences in the 5 highest/lowest pairs

Counts in the 10 texts of `attention/pairs5.txt` (highest / lowest answers):

| Class | Words found |
|---|---|
| causal | so 3/2, since 2/1, because 0/1, thus 0/1 |
| contrast | but 4/4, actually 3/0, though 0/1 |
| illustration | like 2/1, imagine 0/1, *for example* 0/1 |
| reformulation | essentially 0/1 |
| hedge | might 1/1, usually 1/0, mostly 1/0, roughly 1/0, *sort of* 1/0, maybe 0/1, probably 0/1, *i think* 0/1 |
| condition | if 1/6, when 2/3 |
| reader | you 11/19, you're 0/7, your 1/0, you've 0/1, yourself 0/1 |
| writer | me 2/0, i 1/1, i'm 1/0, i'd 0/1 |

Most classes rest on one or two words here (causal is mostly *so*, contrast mostly *but*),
so class averages for these 10 texts describe those words rather than the class. Use the
200-explanation sample (with confidence intervals) for comparisons.
