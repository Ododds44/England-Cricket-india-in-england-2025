"""
Joe Root - Test innings run-distribution model
==============================================

Question: before an innings starts, what is the full probability distribution
of Joe Root's score - not just "how many runs on average", but P(10+), P(50+),
and P(100+)?

Method (in one paragraph)
-------------------------
Batting is modelled as survival: at each run r, Root either gets out
("hazard" h_r) or moves on. The hazard is held constant within run bands
(0, 1-4, 5-9, 10-19, ... 150-199, 200+) and estimated from his past innings,
with three choices that matter:
  1. Not-outs are treated as *censored*, not as dismissals. A 254* tells us he
     survived to 254; it does not tell us where he would have got out.
  2. Recent innings count more (exponential decay, half-life in years).
  3. Innings in similar conditions count more (home vs away; batting in the
     match's 1st/2nd innings vs 3rd/4th).
Each band's hazard is shrunk towards his whole-career hazard for that band, so
thin data can't produce silly numbers. The decay and condition weights are tuned
on 2016-2022 and evaluated out-of-sample on 2023-2026 (walk-forward: every
prediction only uses innings that had already happened).

Run:
    python root_model.py backtest          # walk-forward evaluation + charts
    python root_model.py series            # India 2025 pre-series forecast vs actual
    python root_model.py predict --home --first-innings --asof 2026-11-17
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data" / "root_test_innings.csv"
OUT = Path(__file__).parent / "output"

# Statsguru ground names for Tests in England & Wales
HOME_GROUNDS = {
    "Lord's", "The Oval", "Manchester", "Leeds", "Birmingham", "Nottingham",
    "Southampton", "Chester-le-Street", "Cardiff",
}

BAND_EDGES = [0, 1, 5, 10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200, 10_000]
MAX_RUNS = 400              # distribution is computed out to this score
THRESHOLDS = [10, 30, 50, 100]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["home"] = df["ground"].isin(HOME_GROUNDS)
    df["first_half"] = df["match_innings"] <= 2      # match innings 1-2 vs 3-4
    return df.sort_values(["date", "match_innings"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class Params:
    # defaults = the values chosen by tune() on 2016-2022
    half_life_years: float = 20.0  # weight halves every N years
    w_other_venue: float = 1.0     # weight on away innings when predicting home (and vice versa)
    w_other_phase: float = 1.0     # weight on 3rd/4th-inns data when predicting 1st/2nd (and vice versa)
    prior_strength: float = 80.0   # shrinkage towards career hazard, in "runs of exposure"


def _band_counts(runs, not_out, w):
    """Weighted dismissals and run-exposure in each band.

    Exposure = number of runs r at which he was 'at risk' of getting out.
    Out on R: at risk at 0..R (R+1 values). Not out on R: at risk at 0..R-1.
    """
    runs, not_out, w = map(np.asarray, (runs, not_out, w))
    last_at_risk = np.where(not_out == 1, runs - 1, runs)
    events, exposure = [], []
    for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        exp_i = np.clip(np.minimum(last_at_risk + 1, b) - a, 0, None)
        ev_i = ((not_out == 0) & (runs >= a) & (runs < b)).astype(float)
        exposure.append(np.sum(w * exp_i))
        events.append(np.sum(w * ev_i))
    return np.array(events), np.array(exposure)


def band_hazards(hist: pd.DataFrame, home: bool, first_half: bool,
                 asof: pd.Timestamp, p: Params) -> np.ndarray:
    # career prior (unweighted)
    e0, x0 = _band_counts(hist.runs, hist.not_out, np.ones(len(hist)))
    h0 = np.where(x0 > 0, e0 / np.maximum(x0, 1e-9), e0.sum() / x0.sum())

    age = (asof - hist.date).dt.days.to_numpy() / 365.25
    w = 0.5 ** (age / p.half_life_years)
    w = w * np.where(hist.home.to_numpy() == home, 1.0, p.w_other_venue)
    w = w * np.where(hist.first_half.to_numpy() == first_half, 1.0, p.w_other_phase)
    w = w / w.mean()                       # keep total weight ~ number of innings

    e, x = _band_counts(hist.runs, hist.not_out, w)
    return (e + p.prior_strength * h0) / (x + p.prior_strength)


def distribution(h_band: np.ndarray) -> np.ndarray:
    """P(score = k) for k = 0..MAX_RUNS (last cell holds the tail)."""
    h = np.empty(MAX_RUNS)
    for i, (a, b) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
        h[a:min(b, MAX_RUNS)] = h_band[i]
    survive_to = np.concatenate([[1.0], np.cumprod(1 - h)])   # P(score >= k)
    pmf = survive_to[:-1] * h
    return np.append(pmf, survive_to[-1])


def p_at_least(pmf: np.ndarray, k: int) -> float:
    return float(pmf[k:].sum())


@dataclass
class Forecast:
    pmf: np.ndarray

    @property
    def mean(self): return float(np.arange(len(self.pmf)) @ self.pmf)
    @property
    def median(self): return int(np.searchsorted(np.cumsum(self.pmf), 0.5))
    def p(self, k): return p_at_least(self.pmf, k)


def forecast(df: pd.DataFrame, asof, home: bool, first_half: bool,
             p: Params = Params()) -> Forecast:
    asof = pd.Timestamp(asof)
    hist = df[df.date < asof]
    return Forecast(distribution(band_hazards(hist, home, first_half, asof, p)))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def _logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def walk_forward(df, start, end, p: Params | None, k_list=THRESHOLDS):
    """Predict every innings in [start, end) using only earlier innings.

    p=None gives the naive baseline: raw share of past scores >= k, with
    not-outs counted as if he'd been dismissed (what a simple average implies).
    A not-out below k is excluded from that threshold's scoring - we never
    learn whether he'd have reached k.
    """
    rows = []
    test = df[(df.date >= start) & (df.date < end)]
    for _, r in test.iterrows():
        hist = df[df.date < r.date]
        if p is None:
            probs = {k: (hist.runs >= k).mean() for k in k_list}
        else:
            f = forecast(df, r.date, r.home, r.first_half, p)
            probs = {k: f.p(k) for k in k_list}
        for k in k_list:
            if r.not_out and r.runs < k:
                continue
            rows.append((r.date, k, probs[k], int(r.runs >= k)))
    return pd.DataFrame(rows, columns=["date", "k", "prob", "hit"])


def score(wf: pd.DataFrame) -> pd.DataFrame:
    return wf.groupby("k").apply(lambda g: pd.Series({
        "n": len(g), "hit_rate": g.hit.mean(), "avg_prob": g.prob.mean(),
        "log_loss": _logloss(g.prob.values, g.hit.values),
        "brier": float(np.mean((g.prob - g.hit) ** 2)),
    }))


def tune(df, start="2016-01-01", end="2023-01-01") -> Params:
    grid = itertools.product([2, 3, 5, 8, 12, 20], [0.4, 0.6, 0.8, 1.0],
                             [0.6, 0.8, 1.0], [20, 40, 80])
    best, best_ll = None, np.inf
    for hl, wv, wp, ps in grid:
        p = Params(hl, wv, wp, ps)
        ll = score(walk_forward(df, start, end, p)).log_loss.mean()
        if ll < best_ll:
            best, best_ll = p, ll
    return best


# --------------------------------------------------------------------------
# India 2025 series
# --------------------------------------------------------------------------
INDIA_2025 = ["Leeds", "Birmingham", "Lord's", "Manchester", "The Oval"]   # Headingley, Edgbaston, Lord's, Old Trafford, The Oval
SERIES_START = "2025-06-20"


def p_second_innings(df, asof) -> float:
    """Share of his past home Tests in which he batted twice."""
    h = df[(df.date < pd.Timestamp(asof)) & df.home]
    per_test = h.groupby("date").size()
    return float((per_test >= 2).mean())


def simulate_series(df, p: Params, n_tests=5, asof=SERIES_START, sims=50_000, seed=1):
    rng = np.random.default_rng(seed)
    f1 = forecast(df, asof, True, True, p)
    f2 = forecast(df, asof, True, False, p)
    p2 = p_second_innings(df, asof)
    ks = np.arange(len(f1.pmf))
    first = rng.choice(ks, size=(sims, n_tests), p=f1.pmf).sum(1)
    second = (rng.choice(ks, size=(sims, n_tests), p=f2.pmf) *
              (rng.random((sims, n_tests)) < p2)).sum(1)
    return f1, f2, p2, first + second


def old_model_forecast(df):
    """Re-create the original repo's approach on the same data, for comparison:
    random forest on (ground, innings, rolling-5 avg) using home Tests v India,
    random 80/20 split, PrevRuns fixed at its mean when forecasting."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    d = df[(df.opposition == "India") & df.home & (df.date < SERIES_START)].copy()
    d["Innings"] = d.match_innings          # Statsguru 'Inns' = match innings 1-4
    d["PrevRuns"] = d.runs.shift(1).rolling(5, 1).mean().fillna(d.runs.mean())
    le = LabelEncoder().fit(d.ground)
    d["g"] = le.transform(d.ground)
    X_tr, _, y_tr, _ = train_test_split(d[["g", "Innings", "PrevRuns"]], d.runs,
                                        test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_tr, y_tr)
    preds = []
    for v in INDIA_2025:
        for inn in (1, 2):
            X = pd.DataFrame({"g": [le.transform([v])[0]], "Innings": [inn],
                              "PrevRuns": [d.PrevRuns.mean()]})
            preds.append(rf.predict(X)[0])
    return len(d), float(np.sum(preds))


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
INK, MUTED, GRID, ACCENT, ACCENT2 = "#1f2328", "#6e7781", "#e6e8eb", "#2f6fdb", "#b4c7ec"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def chart_series(totals, actual, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    _style(ax)
    bins = np.arange(0, 1100, 25)
    ax.hist(totals, bins=bins, color=ACCENT, edgecolor="white", linewidth=1.5)
    ax.axvline(actual, color=INK, lw=2)
    pct = (totals < actual).mean()
    ax.annotate(f"Actual: {actual} runs\n({pct:.0%} of simulations were lower)",
                xy=(actual, ax.get_ylim()[1] * 0.9), xytext=(8, 0),
                textcoords="offset points", color=INK, fontsize=9, va="top")
    lo, hi = np.percentile(totals, [10, 90])
    ax.set_title(f"Root series runs v India 2025 - pre-series forecast  "
                 f"(median {np.median(totals):.0f}, 80% range {lo:.0f}-{hi:.0f})",
                 loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("Total runs in the five-Test series", color=INK)
    ax.set_ylabel("Simulations", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_survival(df, p: Params, asof, path):
    """P(score >= k): the model vs naively counting not-outs as dismissals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    asof = pd.Timestamp(asof)
    hist = df[df.date < asof]
    f = forecast(df, asof, True, True, p)
    ks = np.arange(0, 201)
    model = [f.p(k) for k in ks]
    naive = [(hist.runs >= k).mean() for k in ks]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    _style(ax)
    ax.plot(ks, naive, color=ACCENT2, lw=2, label="Naive: past scores, not-outs counted as out")
    ax.plot(ks, model, color=ACCENT, lw=2, label="Model: not-outs censored, bands shrunk to career")
    for k in (50, 100, 150):
        ax.plot([k], [f.p(k)], "o", ms=8, color=ACCENT, mec="white", mew=2)
        ax.annotate(f"{f.p(k):.0%}", (k, f.p(k)), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlim(0, 200); ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Score k", color=INK)
    ax.set_ylabel("P(score >= k)", color=INK)
    ax.set_title(f"Joe Root - chance of reaching each score in his next Test innings "
                 f"(as at {asof:%d %b %Y})", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def cmd_backtest(df):
    OUT.mkdir(exist_ok=True)
    print("Tuning on 2016-2022 (walk-forward)...")
    p = tune(df)
    print(f"  best params: {p}\n")
    hold = ("2023-01-01", "2027-01-01")
    model = score(walk_forward(df, *hold, p))
    naive = score(walk_forward(df, *hold, None))
    cmp = pd.DataFrame({
        "innings_scored": model.n.astype(int),
        "actual_hit_rate": model.hit_rate.round(3),
        "model_avg_prob": model.avg_prob.round(3),
        "naive_avg_prob": naive.avg_prob.round(3),
        "model_log_loss": model.log_loss.round(4),
        "naive_log_loss": naive.log_loss.round(4),
        "log_loss_improvement_%": ((1 - model.log_loss / naive.log_loss) * 100).round(1),
    })
    cmp.index = [f"{k}+" for k in cmp.index]
    n_hold = len(df[df.date >= hold[0]])
    print(f"Out-of-sample 2023-2026, {n_hold} innings (lower log loss = better).")
    print("Naive = share of all past scores >= k, with not-outs counted as dismissals.")
    print(cmp.to_string())
    cmp.to_csv(OUT / "backtest_2023_2026.csv")
    return p


def cmd_series(df, p):
    OUT.mkdir(exist_ok=True)
    f1, f2, p2, totals = simulate_series(df, p)
    actual = df[(df.opposition == "India") & (df.date >= SERIES_START) &
                (df.date < "2025-09-01")]
    print(f"\nIndia 2025 - forecast as at {SERIES_START} (using only earlier innings)")
    if abs(f1.mean - f2.mean) < 0.05:
        print("  (tuned weights treat 1st/2nd and 3rd/4th innings alike, so both rows match)")
    print(f"  Per innings, 1st/2nd match inns:  mean {f1.mean:.1f}  median {f1.median}"
          f"  P(50+) {f1.p(50):.0%}  P(100+) {f1.p(100):.0%}")
    print(f"  Per innings, 3rd/4th match inns:  mean {f2.mean:.1f}  median {f2.median}"
          f"  P(50+) {f2.p(50):.0%}  P(100+) {f2.p(100):.0%}")
    print(f"  P(bats twice in a Test) {p2:.0%}")
    lo, hi = np.percentile(totals, [10, 90])
    print(f"  Series total: median {np.median(totals):.0f}, mean {totals.mean():.0f}, "
          f"80% range {lo:.0f}-{hi:.0f}")
    hundreds_exp = 5 * f1.p(100) + 5 * p2 * f2.p(100)
    print(f"  Expected hundreds {hundreds_exp:.2f}")
    print(f"\n  Actual: {actual.runs.sum()} runs in {len(actual)} innings, "
          f"{(actual.runs >= 100).sum()} hundreds, {((actual.runs >= 50) & (actual.runs < 100)).sum()} fifties")
    print(f"  Actual total sits at the {(totals < actual.runs.sum()).mean():.0%} percentile of the forecast")
    n_old, old_total = old_model_forecast(df)
    print(f"\n  Original repo approach (RF on {n_old} home innings v India, venues corrected): "
          f"{old_total:.0f} runs, single number, no uncertainty")
    chart_series(totals, int(actual.runs.sum()), OUT / "india_2025_series_forecast.png")


def cmd_predict(df, p, home, first_half, asof):
    OUT.mkdir(exist_ok=True)
    chart_survival(df, p, asof, OUT / "root_score_curve.png")
    f = forecast(df, asof, home, first_half, p)
    where = "home" if home else "away"
    when = "1st/2nd" if first_half else "3rd/4th"
    print(f"\nRoot, {where} Test, {when} match innings, as at {asof}")
    print(f"  Expected runs {f.mean:.1f}, median {f.median}")
    print(f"  {'Score':>6} {'P(reaches it)':>14}")
    for k in [10, 20, 30, 40, 50, 75, 100, 150]:
        print(f"  {str(k) + '+':>6} {f.p(k):>14.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["backtest", "series", "predict", "all"])
    ap.add_argument("--home", action="store_true")
    ap.add_argument("--first-innings", action="store_true",
                    help="1st/2nd match innings (default: 3rd/4th)")
    ap.add_argument("--asof", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    a = ap.parse_args()
    data = load()
    params = cmd_backtest(data) if a.command in ("backtest", "series", "all") else Params()
    if a.command in ("series", "all"):
        cmd_series(data, params)
    if a.command in ("predict", "all"):
        cmd_predict(data, params, a.home, a.first_innings, a.asof)
