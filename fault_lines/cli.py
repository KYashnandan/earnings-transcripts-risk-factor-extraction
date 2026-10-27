import argparse, glob, json
import pandas as pd
from .extract import extract

def guess_columns(df):
    """Pick company, quarter and transcript columns by name and content."""
    cols = list(df.columns)
    low = {c: str(c).lower() for c in cols}
    text = max((c for c in cols if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object),
               key=lambda c: df[c].astype(str).str.len().mean())
    def named(*keys, skip=()):
        for k in keys:
            for c in cols:
                if k in low[c] and c not in skip:
                    return c
    company = named("ticker", "symbol", "company", "name", skip=(text,))
    if company is None:
        company = next(c for c in cols if c != text and df[c].nunique() < len(df) / 2)
    quarter = named("quarter", "period", "date", skip=(text, company)) or next(c for c in cols if c not in (text, company))
    return company, quarter, text

def to_quarter(s):
    """Turn dates into 'YYYY-Qn'; leave labels like '2019-Q1' alone."""
    if s.astype(str).str.contains("Q", case=False).mean() > .5:
        return s.astype(str)
    d = pd.to_datetime(s, errors="coerce")
    return d.dt.to_period("Q").astype(str) if d.notna().mean() > .5 else s.astype(str)

def main():
    a = argparse.ArgumentParser(description="Extract risk sentences from earnings-call transcripts")
    a.add_argument("--csv", default="data/sample_transcripts.csv")
    a.add_argument("--kaggle", help="Kaggle dataset, e.g. ashwinm500/earnings-call-transcripts (downloads first CSV)")
    a.add_argument("--company-col"); a.add_argument("--quarter-col"); a.add_argument("--text-col")
    a.add_argument("--quarters", type=int, default=6)
    a.add_argument("--out", default="docs/data/risks.json")
    p = a.parse_args()
    if p.kaggle:
        import kagglehub
        p.csv = sorted(glob.glob(kagglehub.dataset_download(p.kaggle) + "/**/*.csv", recursive=True))[0]
    df = pd.read_csv(p.csv)
    g = guess_columns(df)
    c, q, t = p.company_col or g[0], p.quarter_col or g[1], p.text_col or g[2]
    print(f"{p.csv}: company={c!r} quarter={q!r} text={t!r}, {len(df)} rows")
    df[q] = to_quarter(df[q])
    res = extract(df, c, q, t, p.quarters)
    with open(p.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {p.out}: {len(res)} companies")

if __name__ == "__main__":
    main()
