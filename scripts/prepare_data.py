"""Turn the Kaggle Motley Fool pickle into data/transcripts.csv (10 companies, 8 latest quarters each).

Dataset: tpotterer/motley-fool-scraped-earnings-call-transcripts (cite it; license unknown).
Only open a pickle from a source you trust.
"""
import argparse
import pandas as pd

TICKERS = ["AAPL", "GOOGL", "AMZN", "TSLA", "DIS", "IBM", "INTC", "F", "GE", "JPM"]

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--pkl", default="data/motley-fool-data.pkl")
    a.add_argument("--out", default="data/transcripts.csv")
    a.add_argument("--quarters", type=int, default=8)
    p = a.parse_args()
    d = pd.read_pickle(p.pkl)
    d = d[d["ticker"].isin(TICKERS)].copy()
    d["q"] = d["q"].astype(str)
    d["n"] = d["transcript"].str.len()
    # the scrape has duplicates: keep the longest transcript per company and quarter
    d = d.sort_values("n", ascending=False).drop_duplicates(["ticker", "q"])
    d = d.sort_values(["ticker", "q"]).groupby("ticker").tail(p.quarters)
    d.rename(columns={"ticker": "company", "q": "quarter"})[["company", "quarter", "transcript"]].to_csv(p.out, index=False)
    print(f"wrote {p.out}: {d['ticker'].nunique()} companies, {len(d)} transcripts")

if __name__ == "__main__":
    main()
