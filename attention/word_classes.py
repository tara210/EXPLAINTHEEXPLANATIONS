"""Word classes for explanation language, and word-level tagging of an answer text.

Words are the whitespace-separated words of the answer (the same `word_index` as in the
extract_attention.py output). Each word gets
  - `pos`: spaCy's coarse part of speech of the word's first non-punctuation token
  - `cls`: one of CLASSES below, or "other"

Classes are matched on normalised words (lowercase, edge punctuation stripped), longest
phrase first, so "kind of like" wins over "like". Single words are then disambiguated
with spaCy (keep_single): "so" not as intensifier, "since" only as conjunction, "like"
only as comparison, "then" only after if/when/unless in the same sentence, "picture"
only as a verb. `cls_raw` keeps the plain list match for comparison.
"""

import re

import spacy

CLASSES = {
    "causal": ["because", "so", "therefore", "thus", "hence", "since", "cause", "causes", "caused",
               "causing", "due to", "as a result", "that's why", "which is why", "leads to", "results in"],
    "contrast": ["but", "however", "although", "though", "whereas", "instead", "unlike",
                 "actually", "otherwise", "on the other hand"],
    "illustration": ["like", "imagine", "think of", "for example", "for instance", "example",
                     "let's say", "picture", "similar", "similarly", "kind of like", "analogy"],
    "reformulation": ["basically", "essentially", "in other words", "means", "meaning",
                      "that is", "i.e", "called", "refers to", "simply put"],
    "hedge": ["kind of", "sort of", "probably", "might", "maybe", "perhaps", "usually",
              "generally", "roughly", "pretty much", "mostly", "typically", "i think"],
    "condition": ["if", "when", "unless", "whenever", "then"],
    "reader": ["you", "your", "you're", "yourself", "you'll", "you've"],
    "writer": ["i", "i'm", "my", "me", "i've", "i'd"],
}
CLASS_ORDER = list(CLASSES) + ["other"]

WORD = re.compile(r"\S+")
EDGE_PUNCT = re.compile(r"^[^\w']+|[^\w']+$")
_PHRASES = sorted(((tuple(p.split()), c) for c, ps in CLASSES.items() for p in ps),
                  key=lambda x: -len(x[0]))
_nlp = None


def norm(word: str) -> str:
    return EDGE_PUNCT.sub("", word.lower().replace("’", "'"))


def nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _nlp


CONDITION_OPENERS = {"if", "when", "unless", "whenever"}


def keep_single(word: dict, normed_sentence_before: list[str], next_word: dict | None) -> bool:
    """Disambiguation for single-word matches, using spaCy's analysis of the word.
    Returns False when the word is used in another sense than its class."""
    w, pos, head = word["word_norm"], word["pos"], word["head_pos"]
    if w == "so":  # "so big", "so much" are intensifiers, not causal
        return head not in ("ADJ", "ADV")
    if w == "since":  # "since 2010" is temporal
        if next_word is not None and next_word["pos"] == "NUM":
            return False
        return pos == "SCONJ" or word["dep"] == "mark"
    if w == "like":  # comparison "is like a pump"; not the verb or the filler
        return pos in ("ADP", "SCONJ")
    if w == "then":  # "if ..., then ..." only; "and then" is sequence
        return bool(CONDITION_OPENERS & set(normed_sentence_before))
    if w == "picture":  # "picture a ball" (imperative), not "a picture"
        return pos == "VERB"
    return True


def tag_words(answer: str) -> list[dict]:
    """One dict per whitespace word: word_index, start, end, word, word_norm, pos, dep,
    head_pos, sent_start, cls (disambiguated) and cls_raw (plain list match)."""
    words = [{"word_index": k, "start": m.start(), "end": m.end(), "word": m.group(),
              "word_norm": norm(m.group()), "pos": "X", "dep": "", "head_pos": "",
              "sent": 0, "sent_start": False, "cls": "other", "cls_raw": "other"}
             for k, m in enumerate(WORD.finditer(answer))]

    # spaCy analysis of the first non-punctuation token inside each word.
    k, sent = 0, -1
    doc = nlp()(answer)
    for tok in doc:
        if tok.is_sent_start:
            sent += 1
        while k + 1 < len(words) and tok.idx >= words[k + 1]["start"]:
            k += 1
        w = words[k]
        if w["pos"] == "X" and tok.pos_ not in ("PUNCT", "SPACE") and w["start"] <= tok.idx < w["end"]:
            w.update(pos=tok.pos_, dep=tok.dep_, head_pos=tok.head.pos_, sent=max(sent, 0),
                     sent_start=all(t.is_punct for t in doc[tok.sent.start:tok.i]))

    # Classes: longest phrase first; a word belongs to at most one class.
    normed = [w["word_norm"] for w in words]
    for i in range(len(words)):
        if words[i]["cls_raw"] != "other":
            continue
        for phrase, cls in _PHRASES:
            n = len(phrase)
            if tuple(normed[i:i + n]) == phrase and all(words[j]["cls_raw"] == "other" for j in range(i, i + n)):
                keep = True
                if n == 1:
                    before = [normed[j] for j in range(i) if words[j]["sent"] == words[i]["sent"]]
                    keep = keep_single(words[i], before, words[i + 1] if i + 1 < len(words) else None)
                for j in range(i, i + n):
                    words[j]["cls_raw"] = cls
                    if keep:
                        words[j]["cls"] = cls
                break
    return words
