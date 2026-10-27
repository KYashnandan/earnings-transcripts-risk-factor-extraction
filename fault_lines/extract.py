"""Extract risk-factor sentences from earnings-call transcripts with spaCy."""
import collections
import spacy
from spacy.matcher import Matcher
from .lexicon import LEXICON

NEGATORS = {"no", "not", "without", "never", "n't", "neither", "nor"}

def build_nlp(model="en_core_web_sm"):
    """Trained spaCy pipeline (sentence parser + lemmatizer) and a lemma-based Matcher."""
    nlp = spacy.load(model, disable=["ner"])
    matcher = Matcher(nlp.vocab)
    for cat, phrases in LEXICON.items():
        patterns = [[{"LEMMA": t.lemma_.lower()} for t in nlp(p)] for p in phrases]
        matcher.add(cat, patterns)
    return nlp, matcher

def risk_sentences(doc, nlp, matcher, min_len=8, max_len=60):
    """Yield (category_hits, sentence_text) for sentences with a non-negated risk phrase."""
    for sent in doc.sents:
        if not min_len <= len(sent) <= max_len:
            continue
        hits = collections.Counter()
        for mid, start, _ in matcher(sent):
            window = sent[max(0, start - 3):start]  # Matcher offsets are relative to the sentence span
            if any(t.lower_ in NEGATORS for t in window):
                continue  # "no supply chain disruption" is not a risk
            hits[nlp.vocab.strings[mid]] += 1
        if hits:
            yield hits, sent.text.strip()

def extract(df, company_col="company", quarter_col="quarter", text_col="transcript", quarters=6, top=5):
    """Return {company: {quarters, counts, sentences, total_sentences}} for the dashboard and website."""
    nlp, matcher = build_nlp()
    out = {}
    for co, g in df.dropna(subset=[text_col]).groupby(company_col):
        qs = sorted(g[quarter_col].astype(str).unique())[-quarters:]
        counts = {c: [0] * len(qs) for c in LEXICON}
        found = {c: [[] for _ in qs] for c in LEXICON}
        total = [0] * len(qs)
        for qi, q in enumerate(qs):
            for text in g[g[quarter_col].astype(str) == q][text_col]:
                doc = nlp(text)
                total[qi] += sum(1 for _ in doc.sents)
                for hits, sent in risk_sentences(doc, nlp, matcher):
                    for c, n in hits.items():
                        counts[c][qi] += 1
                        found[c][qi].append((n, sent))
        sents = {c: [[t for _, t in sorted(x, reverse=True)[:top]] for x in found[c]] for c in LEXICON}
        out[str(co)] = {"quarters": qs, "counts": counts, "sentences": sents, "total_sentences": total}
    return out
