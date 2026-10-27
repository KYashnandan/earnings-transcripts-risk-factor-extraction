"""Build a sentence-level dataset: weak labels from the lexicon, with the matched words masked."""
import collections
import pandas as pd
from .extract import build_nlp, NEGATORS
from .lexicon import LEXICON

def analyse_sentence(sent, matcher, vocab):
    """Return (category_hits, text with matched words replaced by KW) for one spaCy sentence."""
    hits, mask = collections.Counter(), set()
    for mid, s, e in matcher(sent):
        mask.update(range(s, e))
        if any(t.lower_ in NEGATORS for t in sent[max(0, s - 3):s]):
            continue
        hits[vocab.strings[mid]] += 1
    masked = "".join(("KW" if i in mask else t.text) + t.whitespace_ for i, t in enumerate(sent)).strip()
    return hits, masked

def build_sentences(df, company_col="company", quarter_col="quarter", text_col="transcript", min_len=8, max_len=60):
    nlp, matcher = build_nlp()
    rows = []
    keys = list(zip(df[company_col], df[quarter_col].astype(str)))
    for (co, q), doc in zip(keys, nlp.pipe(df[text_col].tolist(), batch_size=2)):
        for sent in doc.sents:
            if not min_len <= len(sent) <= max_len:
                continue
            hits, masked = analyse_sentence(sent, matcher, nlp.vocab)
            label = max(LEXICON, key=lambda c: hits[c]) if hits else "none"
            rows.append((co, q, sent.text.strip(), masked, label))
    return pd.DataFrame(rows, columns=["company", "quarter", "text", "masked", "label"])

def main():
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--csv", default="data/transcripts.csv")
    a.add_argument("--out", default="data/sentences.csv")
    p = a.parse_args()
    df = pd.read_csv(p.csv)
    out = build_sentences(df)
    out.to_csv(p.out, index=False)
    print(out["label"].value_counts().to_string())

if __name__ == "__main__":
    main()
