"""Streamlit dashboard: risk factors from corporate earnings-call transcripts."""
import json, os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from fault_lines.extract import extract, build_nlp
from fault_lines.lexicon import LEXICON

st.set_page_config(page_title="Risk factors in earnings calls", layout="wide")
NAMES = {k: k.replace("_", " ").title() for k in LEXICON}

@st.cache_data(show_spinner="Reading transcripts with spaCy...")
def analyse(df, c, q, t):
    return extract(df, c, q, t)

st.sidebar.title("Earnings risk factors")
up = st.sidebar.file_uploader("Transcripts CSV (optional)", type="csv")
if up:
    raw = pd.read_csv(up)
    cols = list(raw.columns)
    c = st.sidebar.selectbox("Company column", cols, 0)
    q = st.sidebar.selectbox("Quarter column", cols, min(1, len(cols) - 1))
    t = st.sidebar.selectbox("Transcript column", cols, len(cols) - 1)
    data = analyse(raw, c, q, t)
else:
    path = "docs/data/risks.json"
    if not os.path.exists(path):
        st.error("Run `python -m fault_lines.cli` first, or upload a CSV in the sidebar.")
        st.stop()
    data = json.load(open(path))
    st.sidebar.caption("Precomputed from Motley Fool earnings-call transcripts (Kaggle, tpotterer).")

co = st.sidebar.selectbox("Company", sorted(data))
d = data[co]
counts = pd.DataFrame(d["counts"], index=d["quarters"]).rename(columns=NAMES)
total = pd.Series(d["total_sentences"], index=d["quarters"])
intensity = (counts.sum(axis=1) / total.clip(lower=1) * 100).round(1)

st.title(f"Risk factors in {co} earnings calls")
tab_dash, tab_sent, tab_ml, tab_live, tab_3d = st.tabs(["Dashboard", "Risk sentences", "Classifier", "Try it live", "3D view"])

with tab_dash:
    last = counts.iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk sentences, latest quarter", int(last.sum()),
              int(last.sum() - counts.iloc[-2].sum()) if len(counts) > 1 else None)
    m2.metric("Top risk type", last.idxmax() if last.sum() else "None")
    m3.metric("Share of sentences that mention risk", f"{intensity.iloc[-1]}%")
    a, b = st.columns(2)
    a.subheader("Latest quarter by risk type"); a.bar_chart(last)
    b.subheader("Trend by quarter"); b.line_chart(counts)
    st.subheader("Risk share of the call (%)"); st.line_chart(intensity)

with tab_sent:
    f1, f2 = st.columns(2)
    cats = f1.multiselect("Risk type", list(NAMES.values()), default=list(NAMES.values()))
    qs = f2.multiselect("Quarter", d["quarters"], default=d["quarters"])
    rows = [{"Quarter": qq, "Risk type": NAMES[k], "Sentence": s}
            for k in LEXICON if NAMES[k] in cats
            for qi, qq in enumerate(d["quarters"]) if qq in qs
            for s in d["sentences"][k][qi]]
    table = pd.DataFrame(rows)
    st.dataframe(table, width="stretch", hide_index=True)
    if not table.empty:
        st.download_button("Download CSV", table.to_csv(index=False), f"{co}_risks.csv", "text/csv")

with tab_ml:
    mp = "docs/data/metrics.json"
    if not os.path.exists(mp):
        st.info("Run `python -m fault_lines.dataset` then `python -m fault_lines.train` to train the classifiers.")
    else:
        m = json.load(open(mp))
        st.write(f"Three classifiers predict a sentence's risk type from its context alone: the matched keywords are masked. "
                 f"Scored with leave-companies-out cross-validation on {m['n_train']:,} sentences from {m['n_companies']} companies.")
        st.caption("Labels come from the keyword lexicon (weak labels), so scores show how well context recovers the lexicon's labels, not agreement with human annotators.")
        st.subheader("Model comparison"); st.dataframe(pd.DataFrame(m["models"]).T)
        st.subheader(f"Per-class results ({m['best']})"); st.dataframe(pd.DataFrame(m["per_class"]).T)
        st.subheader("Confusion matrix (rows true, columns predicted)")
        st.dataframe(pd.DataFrame(m["confusion"], index=m["labels"], columns=m["labels"]))
        st.subheader("Words the final model links to each risk type")
        for c, w in m["top_features"].items():
            st.markdown(f"**{NAMES[c]}**: " + ", ".join(w))

with tab_live:
    st.write("Paste any earnings-call text. Each sentence gets the keyword match and the trained classifier's guess from context.")
    if not os.path.exists("models/risk_classifier.joblib"):
        st.info("Train the model first: `python -m fault_lines.train`.")
    else:
        from fault_lines.classify import analyse_text, load_model

        @st.cache_resource
        def tools():
            nlp, matcher = build_nlp()
            return nlp, matcher, load_model()

        txt = st.text_area("Text", height=160, value="Supply chain disruption pushed our lead time past ten weeks. "
                           "Higher interest rates and persistent inflation are weighing on consumer demand. "
                           "We opened twelve new stores and customers responded warmly.")
        if txt.strip():
            nlp_, matcher_, model_ = tools()
            st.dataframe(pd.DataFrame(analyse_text(txt, nlp_, matcher_, model_)), width="stretch", hide_index=True)

with tab_3d:
    html = open("docs/index.html", encoding="utf-8").read()
    blob = json.dumps({co: d}).replace("<", "\\u003c")
    html = html.replace("<script src=\"https://cdnjs", f"<script>window.RISK_DATA={blob}</script>\n<script src=\"https://cdnjs", 1)
    if hasattr(st, "iframe"):
        st.iframe(html, height=760)
    else:
        components.html(html, height=760, scrolling=True)
