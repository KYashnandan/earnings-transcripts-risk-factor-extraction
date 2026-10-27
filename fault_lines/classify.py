"""Run the trained classifier live on any text."""
import joblib
from .dataset import analyse_sentence

def load_model(path="models/risk_classifier.joblib"):
    return joblib.load(path)

def analyse_text(text, nlp, matcher, model, min_len=4):
    """Per sentence: keyword hits next to the classifier's guess from context (keywords masked)."""
    rows = []
    for sent in nlp(text).sents:
        if len(sent) < min_len:
            continue
        hits, masked = analyse_sentence(sent, matcher, nlp.vocab)
        probs = dict(zip(model.classes_, model.predict_proba([masked])[0]))
        best = max(probs, key=probs.get)
        rows.append({"Sentence": sent.text.strip(), "Keyword match": ", ".join(hits) or "none",
                     "Model says": best, "Confidence": round(float(probs[best]), 2)})
    return rows
