import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
import requests, re, warnings
from io import StringIO

warnings.filterwarnings('ignore')

# --------------------------------------------------
# Helper – fetch and return the first StatsGuru HTML table that
# contains ALL of the supplied key words in its flattened columns.
# --------------------------------------------------

def fetch_html_table(url: str, key_cols) -> pd.DataFrame:
    html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
    tables = pd.read_html(html, header=0)
    for tbl in tables:
        flat_cols = []
        for c in tbl.columns:
            if isinstance(c, tuple):
                flat_cols.append(' '.join([str(x) for x in c if x and str(x) != 'nan']).strip())
            else:
                flat_cols.append(str(c).strip())
        tbl.columns = flat_cols
        if all(any(k.lower() in col.lower() for col in flat_cols) for k in key_cols):
            return tbl
    raise ValueError(f"No table with columns {key_cols} found at {url}")

# --------------------------------------------------
# JoeRootAnalyzer
# --------------------------------------------------

class JoeRootAnalyzer:
    def __init__(self, innings_url: str, team_url: str):
        self.inn_url = innings_url
        self.team_url = team_url
        self.data = pd.DataFrame()
        self.encoders = {}
        self.run_rf = None
        self.win_lr = None

    # ---------- LOAD ----------
    def load(self):
        print('⬇️  Fetching Root innings table…')
        inn_tbl = fetch_html_table(self.inn_url, key_cols=['Date', 'Ground'])
        print('⬇️  Fetching England‑India team table…')
        team_tbl = fetch_html_table(self.team_url, key_cols=['Date', 'Result'])

        self.data = self._parse_innings(inn_tbl)
        self._merge_team(team_tbl)
        self._feature_engineer()
        print(f'✅ Combined records: {len(self.data)}')

    # ---------- PARSE INNINGS (handles two StatsGuru layouts) ----------
    def _parse_innings(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col   = next(c for c in df.columns if 'date' in c.lower())
        ground_col = next(c for c in df.columns if 'ground' in c.lower())

        # ── Layout A: one‑row‑per‑innings (has "Inns" + "Runs") ──
        if any('inn' in c.lower() for c in df.columns) and 'Runs' in df.columns:
            inns_col = next(c for c in df.columns if 'inn' in c.lower())
            parsed = pd.DataFrame({
                'Match date': pd.to_datetime(df[date_col], errors='coerce'),
                'Ground': df[ground_col].astype(str).str.split(',').str[0],
                'Innings': pd.to_numeric(df[inns_col], errors='coerce'),
                'Runs': pd.to_numeric(df['Runs'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')
            }).dropna(subset=['Match date', 'Ground', 'Innings', 'Runs'])
            return parsed

        # ── Layout B: Bat1 / Bat2 columns ──
        bat_cols = [c for c in df.columns if re.fullmatch(r'Bat\d', c)]
        records = []
        for _, row in df.iterrows():
            md = pd.to_datetime(row[date_col], errors='coerce')
            if pd.isna(md):
                continue
            grd = str(row[ground_col]).split(',')[0]
            for inn, bc in enumerate(bat_cols, 1):
                val = str(row[bc])
                if val in ['-', 'DNB', 'TDNB', 'absent', 'nan']:
                    continue
                m = re.search(r'(\d+)', val)
                if not m:
                    continue
                records.append({'Match date': md, 'Ground': grd, 'Innings': inn, 'Runs': int(m.group(1))})
        return pd.DataFrame(records)

    # ---------- MERGE TEAM RESULTS ----------
    def _merge_team(self, tm: pd.DataFrame):
        dcol = next((c for c in tm.columns if 'date' in c.lower()), None)
        gcol = next((c for c in tm.columns if 'ground' in c.lower() or 'venue' in c.lower()), None)
        rcol = next((c for c in tm.columns if 'result' in c.lower()), None)

        if not (dcol and gcol and rcol):
            print('⚠️  Team table missing date/ground/result cols – merge skipped')
            return

        tm = tm.rename(columns={dcol: 'Match date', gcol: 'Ground', rcol: 'Result'})
        tm['Match date'] = pd.to_datetime(tm['Match date'], errors='coerce')
        tm['Ground'] = tm['Ground'].astype(str).str.split(',').str[0]

        self.data = self.data.merge(tm[['Match date', 'Ground', 'Result']],
                                    on=['Match date', 'Ground'], how='left')
        self.data['Win'] = self.data['Result'].str.contains('won', case=False).fillna(False).astype(int)

    # ---------- FEATURES ----------
    def _feature_engineer(self):
        self.data.sort_values('Match date', inplace=True)
        self.data['PrevRuns'] = (
            self.data['Runs'].shift(1).rolling(5, 1).mean().fillna(self.data['Runs'].mean())
        )
        le = LabelEncoder()
        self.data['Ground_enc'] = le.fit_transform(self.data['Ground'])
        self.encoders['Ground'] = le

    # ---------- TRAIN ----------
    def train(self):
        X = self.data[['Ground_enc', 'Innings', 'PrevRuns']]
        y = self.data['Runs']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        self.run_rf = RandomForestRegressor(n_estimators=200, random_state=42)
        self.run_rf.fit(X_tr, y_tr)
        print('Run‑model R²:', r2_score(y_te, self.run_rf.predict(X_te)))

        if 'Win' in self.data.columns:
            self.data['PredRuns'] = self.run_rf.predict(X)
            Xw, yw = self.data[['PredRuns', 'Ground_enc', 'Innings']], self.data['Win']
            Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(Xw, yw, test_size=0.2, random_state=42)
            self.win_lr = LogisticRegression(max_iter=1000)
            self.win_lr.fit(Xw_tr, yw_tr)
            print('Win‑model acc:', accuracy_score(yw_te, self.win_lr.predict(Xw_te)))
        else:
            print('No Win column – win‑model skipped')

    # ---------- FORECAST SERIES ----------
    def forecast_series(self, venues):
        preds = []
        for v in venues:
            enc = self.encoders['Ground'].transform([v])[0]
            for inn in [1, 2]:
                X_pred = pd.DataFrame({'Ground_enc': [enc],
                                        'Innings': [inn],
                                        'PrevRuns': [self.data['PrevRuns'].mean()]})
                preds.append(self.run_rf.predict(X_pred)[0])
        total, avg = sum(preds), np.mean(preds)
        print(f'📈 2025 Series – Root total {total:.1f} runs | avg/inn {avg:.1f}')
        self._plot(venues, preds)

    def _plot(self, venues, preds):
        labels = [f'{v} Inn{i}' for v in venues for i in [1, 2]]
        plt.figure(figsize=(10, 4))
        plt.bar(labels, preds, color='#1f77b4')
        plt.xticks(rotation=45)
        plt.ylabel('Predicted Runs')
        plt.title('Joe Root predicted runs – 2025 England v India')
        plt.tight_layout()
        plt.show()

# ---------------- MAIN ----------------
if __name__ == '__main__':
    INN_URL = 'https://stats.espncricinfo
