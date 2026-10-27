"""Train and compare sentence classifiers for risk type, then write metrics for the dashboard and site."""
import argparse, json
import joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from .lexicon import LEXICON

LABELS = list(LEXICON) + ["none"]

def pipe(clf):
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True), clf)

MODELS = {
    "Logistic regression": lambda: pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Linear SVM": lambda: pipe(LinearSVC(class_weight="balanced")),
    "Naive Bayes": lambda: pipe(ComplementNB()),
}

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--sentences", default="data/sentences.csv")
    a.add_argument("--neg-ratio", type=int, default=3)
    a.add_argument("--out", default="docs/data/metrics.json")
    a.add_argument("--model-out", default="models/risk_classifier.joblib")
    p = a.parse_args()

    df = pd.read_csv(p.sentences).dropna(subset=["masked"])
    pos = df[df.label != "none"]
    neg_all = df[df.label == "none"]
    neg = neg_all.sample(n=min(len(neg_all), len(pos) * p.neg_ratio), random_state=0)
    train = pd.concat([pos, neg]).sample(frac=1, random_state=0)
    X, y, groups = train["masked"], train["label"], train["company"]

    # Evaluation: every company is held out in turn, so the model is always scored on unseen companies.
    cv = GroupKFold(n_splits=5)
    res, preds = {}, {}
    for name, make in MODELS.items():
        pr = cross_val_predict(make(), X, y, groups=groups, cv=cv)
        preds[name] = pr
        res[name] = {"macro_f1": round(f1_score(y, pr, average="macro"), 3), "accuracy": round(accuracy_score(y, pr), 3)}
    best = max(res, key=lambda n: res[n]["macro_f1"])
    pr = preds[best]
    pc, rc, fc, sc = precision_recall_fscore_support(y, pr, labels=LABELS, zero_division=0)
    per_class = {l: {"precision": round(a_, 3), "recall": round(b, 3), "f1": round(c, 3), "support": int(d)}
                 for l, a_, b, c, d in zip(LABELS, pc, rc, fc, sc)}
    cm = confusion_matrix(y, pr, labels=LABELS).tolist()

    # Final model: logistic regression (interpretable, gives probabilities), fitted on all training sentences.
    final = MODELS["Logistic regression"]()
    final.fit(X, y)
    joblib.dump(final, p.model_out, compress=3)
    vec, lr = final.steps[0][1], final.steps[1][1]
    names = np.array(vec.get_feature_names_out())
    top = {c: [t for t in names[np.argsort(-lr.coef_[i])[:40]] if "kw" not in t][:8]
           for i, c in enumerate(lr.classes_) if c != "none"}

    out = {"models": res, "best": best, "per_class": per_class, "labels": LABELS, "confusion": cm,
           "top_features": top, "n_train": int(len(train)), "n_positive": int(len(pos)), "n_companies": int(df.company.nunique())}
    json.dump(out, open(p.out, "w"), indent=1)
    print(json.dumps({"models": res, "best": best, "per_class": per_class}, indent=1))

if __name__ == "__main__":
    main()
