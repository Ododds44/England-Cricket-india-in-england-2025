"""Refresh data/root_test_innings.csv from ESPNcricinfo Statsguru.

    python fetch_data.py            # Joe Root (player 303669)
    python fetch_data.py 253802     # any other player id

Statsguru sometimes rate-limits or blocks scripted requests. If this fails,
open the URL below in a browser, save the page as HTML and run
    python fetch_data.py --html saved_page.html
"""
import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

URL = ("https://stats.espncricinfo.com/ci/engine/player/{pid}.html"
       "?class=1;template=results;type=batting;view=innings")
OUT = Path(__file__).parent / "data" / "root_test_innings.csv"


def parse(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    raw = next(t for t in tables if {"Runs", "BF", "Pos", "Inns", "Ground"} <= set(t.columns))
    raw = raw[~raw["Runs"].astype(str).isin(["DNB", "TDNB", "absent", "sub"])]
    return pd.DataFrame({
        "date": pd.to_datetime(raw["Start Date"], format="%d %b %Y").dt.strftime("%Y-%m-%d"),
        "runs": raw["Runs"].astype(str).str.replace("*", "", regex=False).astype(int),
        "not_out": raw["Runs"].astype(str).str.contains(r"\*").astype(int),
        "balls": pd.to_numeric(raw["BF"], errors="coerce"),
        "position": pd.to_numeric(raw["Pos"], errors="coerce"),
        "match_innings": raw["Inns"].astype(int),
        "opposition": raw["Opposition"].str.replace(r"^v ", "", regex=True),
        "ground": raw["Ground"].str.replace(",", "", regex=False),
    })


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--html"]:
        html = Path(args[1]).read_text(encoding="utf-8")
    else:
        pid = args[0] if args else "303669"
        html = requests.get(URL.format(pid=pid), headers={"User-Agent": "Mozilla/5.0"},
                            timeout=30).text
    df = parse(html)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} innings, {df.runs.sum()} runs, {df.not_out.sum()} not outs -> {OUT}")
