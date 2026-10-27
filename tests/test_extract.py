import pandas as pd
from fault_lines import extract

def run(text):
    df = pd.DataFrame({"company": ["A"], "quarter": ["2025-Q1"], "transcript": [text]})
    return extract(df)["A"]

def test_counts_and_sentences():
    r = run("Supply chain disruption hurt our lead time badly this quarter. Sales were great overall for everyone.")
    assert r["counts"]["supply_chain"] == [1]
    assert r["total_sentences"] == [2]
    assert "Supply chain" in r["sentences"]["supply_chain"][0][0]

def test_lemma_matching():
    r = run("Several regulators are investigating our lending practices across three different states.")
    assert r["counts"]["regulatory"] == [1]

def test_negation_is_ignored():
    r = run("We saw no supply chain disruption at all during the entire quarter this year.")
    assert sum(sum(v) for v in r["counts"].values()) == 0

def test_negation_in_later_sentence():
    r = run("Sales were fine today for all of our customers in every region. We saw no shortage at all of parts during the entire quarter.")
    assert sum(sum(v) for v in r["counts"].values()) == 0

def test_dataset_masks_keywords_and_labels():
    import pandas as pd
    from fault_lines.dataset import build_sentences
    df = pd.DataFrame({"company": ["A"], "quarter": ["2025-Q1"], "transcript": ["The recession is hurting demand for our products across many different markets this year."]})
    r = build_sentences(df).iloc[0]
    assert r.label == "macro" and "recession" not in r.masked and "KW" in r.masked
