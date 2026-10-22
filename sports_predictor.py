#!/usr/bin/env python3
"""
Leak-safe match-outcome predictor for Bangladeshi cricket and football
(decision trees -> random forest -> XGBoost), chronological features only.

  pip install numpy pandas scikit-learn xgboost joblib
  python sports_predictor.py --sport football --fetch            # men + women, fully separate models
  python sports_predictor.py --sport football --gender female    # one category only
  python sports_predictor.py --sport cricket  --fetch --formats ODI T20I Test
  python sports_predictor.py --sport football --gender male   --predict Bangladesh Nepal --fmt qualifier
  python sports_predictor.py --sport football --gender female --predict Bangladesh Nepal --fmt other

Data: cricket = Cricsheet JSON, football = martj42 men's + women's international results (GitHub).
Men and women are NEVER mixed: separate Elo pools, separate models, separate result tables. Each match's features use ONLY earlier matches.
"""
import argparse, json, pickle, urllib.request, zipfile, io, sys
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CRICSHEET = {"ODI": "odis_json.zip", "T20I": "t20s_json.zip", "Test": "tests_json.zip", "BPL": "bpl_json.zip"}
FOOTBALL_URL = {"male": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
                "female": "https://raw.githubusercontent.com/martj42/womens-international-results/master/results.csv"}
HOME_ADV = {"cricket": 0.0, "football": 100.0}


# ----------------------------------------------------------------------------- data
def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})).read()


def fetch(sport, root, formats):
    root.mkdir(parents=True, exist_ok=True)
    if sport == "football":
        for g, url in FOOTBALL_URL.items():
            print("downloading football", g)
            (root / f"results_{g}.csv").write_bytes(_get(url))
    else:
        for f in formats:
            print("downloading", f)
            zipfile.ZipFile(io.BytesIO(_get("https://cricsheet.org/downloads/" + CRICSHEET[f]))).extractall(root / f)


def load_cricket(root, formats, gender):
    rows = []
    for fmt in formats:
        for p in (root / fmt).glob("*.json"):
            try:
                info = json.loads(p.read_text())["info"]
            except Exception:
                continue
            if gender and info.get("gender") != gender:
                continue
            teams, w = info.get("teams", []), info.get("outcome", {}).get("winner")
            if len(teams) != 2 or w not in teams:          # skip ties / no results
                continue
            toss = info.get("toss", {})
            rows.append(dict(date=pd.to_datetime(info["dates"][0]), fmt=fmt, team1=teams[0], team2=teams[1],
                             s1=float(w == teams[0]), venue=info.get("venue", ""), neutral=True,
                             toss_winner=toss.get("winner"), toss_bat=toss.get("decision") == "bat", mid=p.stem, w=1.0))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "mid"]).reset_index(drop=True)


MAJOR = {"FIFA World Cup", "World Cup", "UEFA Euro", "Euro", "European Championship", "Copa América", "African Cup of Nations",
         "Africa Cup of Nations", "African Championship", "AFC Asian Cup", "AFC Championship", "CONCACAF Gold Cup",
         "CONCACAF Championship", "OFC Championship", "Oceania Nations Cup", "Olympic Games", "Confederations Cup"}
TIER_K = {"friendly": 0.7, "qualifier": 1.0, "regional": 1.0, "major": 1.3}


def tier(t):
    if t == "Friendly":
        return "friendly"
    if "qualif" in t.lower():
        return "qualifier"
    return "major" if t in MAJOR else "regional"      # regional = SAFF, Asian Games, nations leagues, cups...


def load_football(root, gender):
    d = pd.read_csv(root / f"results_{gender}.csv", parse_dates=["date"]).dropna(subset=["home_score", "away_score"])
    h, a = d.home_score.to_numpy(), d.away_score.to_numpy()
    gd = np.abs(h - a)
    margin = np.where(gd <= 1, 1.0, np.where(gd == 2, 1.5, (11 + gd) / 8))      # bigger wins move Elo more
    fm = d.tournament.map(tier)
    df = pd.DataFrame(dict(date=d.date, fmt=fm, team1=d.home_team, team2=d.away_team,
                           s1=np.where(h > a, 1.0, np.where(h == a, 0.5, 0.0)), venue=d.country,
                           neutral=d.neutral.astype(bool), toss_winner=None, toss_bat=False,
                           w=margin * fm.map(TIER_K).to_numpy()))
    df["mid"] = range(len(df))
    return df.sort_values(["date", "mid"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- state / features
class Tracker:
    """Running state. features() reads it, update() writes it. Always call them in that order."""

    def __init__(self, home_adv=0.0, k_scale=1.0):
        self.home_adv, self.k_scale = home_adv, k_scale
        self.elo, self.elo_f, self.n = {}, {}, {}
        self.form, self.h2h, self.venue = {}, {}, {}

    def _k(self, n):                             # big K for new teams, smaller as history grows
        return self.k_scale * (16 + 48 / (1 + n / 20))

    def _form(self, t, m):
        v = list(self.form.get(t, []))[-m:]
        return (sum(v) + 0.5 * (m - len(v))) / m

    def _h2h(self, a, b):
        key = tuple(sorted((a, b)))
        pts, g = self.h2h.get(key, (0.0, 0))
        pa = pts if a == key[0] else g - pts
        return (pa + 1) / (g + 2) - 0.5, g

    def _venue(self, t, v):
        pts, g = self.venue.get((t, v), (0.0, 0))
        return (pts + 1.5) / (g + 3) - 0.5

    def features(self, a, b, fmt, venue, neutral):
        ea, eb = self.elo.get(a, 1500.0), self.elo.get(b, 1500.0)
        fa, fb = self.elo_f.get((fmt, a), 1500.0), self.elo_f.get((fmt, b), 1500.0)
        h, hn = self._h2h(a, b)
        return {"elo_diff": ea - eb, "elo_avg": (ea + eb) / 2 - 1500, "home": 0.0 if neutral else 1.0,
                "elo_fmt_diff": fa - fb,
                "form5_diff": self._form(a, 5) - self._form(b, 5), "form10_diff": self._form(a, 10) - self._form(b, 10),
                "h2h": h, "h2h_n": hn, "venue_diff": self._venue(a, venue) - self._venue(b, venue),
                "exp_diff": np.log1p(self.n.get(a, 0)) - np.log1p(self.n.get(b, 0))}

    def update(self, a, b, fmt, venue, s, neutral, w=1.0):
        ea, eb = self.elo.get(a, 1500.0), self.elo.get(b, 1500.0)
        adv = 0.0 if neutral else self.home_adv
        exp = 1 / (1 + 10 ** (-(ea + adv - eb) / 400))
        ka, kb = w * self._k(self.n.get(a, 0)), w * self._k(self.n.get(b, 0))
        self.elo[a], self.elo[b] = ea + ka * (s - exp), eb - kb * (s - exp)
        fa, fb = self.elo_f.get((fmt, a), 1500.0), self.elo_f.get((fmt, b), 1500.0)
        expf = 1 / (1 + 10 ** (-(fa + adv - fb) / 400))
        self.elo_f[(fmt, a)], self.elo_f[(fmt, b)] = fa + ka * (s - expf), fb - kb * (s - expf)
        for t, r in ((a, s), (b, 1 - s)):
            self.form.setdefault(t, deque(maxlen=10)).append(r)
            self.n[t] = self.n.get(t, 0) + 1
            p, g = self.venue.get((t, venue), (0.0, 0))
            self.venue[(t, venue)] = (p + r, g + 1)
        key = tuple(sorted((a, b)))
        pts, g = self.h2h.get(key, (0.0, 0))
        self.h2h[key] = (pts + (s if a == key[0] else 1 - s), g + 1)


def build(df, tracker, fmts, use_toss=False, seed=0):
    """Chronological pass: record features BEFORE updating state with the result."""
    flips = (np.random.default_rng(seed).random(len(df)) < 0.5) & df.neutral.to_numpy()   # randomise A/B order
    recs, ys, meta = [], [], []
    for i, r in enumerate(df.itertuples(index=False)):
        a, b, s = r.team1, r.team2, r.s1
        if flips[i]:
            a, b, s = b, a, 1 - s
        f = tracker.features(a, b, r.fmt, r.venue, r.neutral)
        for m in fmts:
            f["fmt_" + m] = float(m == r.fmt)
        if use_toss:
            f["toss_a"] = 1.0 if r.toss_winner == a else (-1.0 if r.toss_winner == b else 0.0)
            f["toss_bat"] = float(bool(r.toss_bat))
        recs.append(f)
        ys.append(int(round(s * 2)))
        meta.append((r.date, a, b))
        tracker.update(a, b, r.fmt, r.venue, s, r.neutral, r.w)      # <- only now does the result enter the state
    return pd.DataFrame(recs), np.array(ys), pd.DataFrame(meta, columns=["date", "a", "b"])


def leak_test(df, fmts, use_toss, home_adv, n=1500):
    n = min(n, len(df))
    """Flip one result: features of that match and everything before must not change."""
    sub = df.iloc[:n].reset_index(drop=True)
    i = next(j for j in range(n // 2, n) if sub.s1[j] != 0.5)
    X1, _, _ = build(sub, Tracker(home_adv), fmts, use_toss)
    sub2 = sub.copy()
    sub2.loc[i, "s1"] = 1 - sub2.loc[i, "s1"]
    X2, _, _ = build(sub2, Tracker(home_adv), fmts, use_toss)
    assert X1.iloc[: i + 1].equals(X2.iloc[: i + 1]), "LEAK: a result changed its own / earlier features"
    assert not X1.iloc[i + 1:].equals(X2.iloc[i + 1:]), "state not updating"
    print(f"leak test passed (flipped match #{i}; its features and all earlier ones unchanged)")


# ----------------------------------------------------------------------------- modelling
def score(y, p, k):
    return dict(acc=accuracy_score(y, p.argmax(1)), logloss=log_loss(y, p, labels=list(range(k))),
                brier=((p - np.eye(k)[y]) ** 2).sum(1).mean(), n=len(y))


def make_xgb(**kw):
    from xgboost import XGBClassifier
    return XGBClassifier(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         reg_lambda=5, n_jobs=-1, random_state=0, **kw)


def split_masks(meta, t_from, v_start, te_start):
    tr = ((meta.date >= t_from) & (meta.date < v_start)).to_numpy()
    va = ((meta.date >= v_start) & (meta.date < te_start)).to_numpy()
    return tr, va, (meta.date >= te_start).to_numpy()


def tune_elo(df, fmts, sport, use_toss, dates):
    """Pick home-advantage and K-scale by validation log loss (never the test set)."""
    grid = [(h, k) for h in ((0.0, 50.0, 100.0) if sport == "football" else (0.0,)) for k in (1.0, 1.4, 2.0)]
    best = None
    for h, ks in grid:
        X, y_raw, meta = build(df, Tracker(h, ks), fmts, use_toss)
        classes = np.unique(y_raw)
        y = np.searchsorted(classes, y_raw)
        tr, va, _ = split_masks(meta, *dates)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
        ll = score(y[va], m.predict_proba(X[va]), len(classes))["logloss"]
        print(f"    home_adv={h:>5.0f} k_scale={ks}: val logloss {ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, h, ks)
    return best[1], best[2]


def run(args, gender, root):
    sport = args.sport
    df = load_football(root, gender) if sport == "football" else load_cricket(root, args.formats, gender)
    if df.empty:
        return print(f"\nno {gender} {sport} matches found in {root} - skipping")
    fmts = sorted(df.fmt.unique())
    d = dict(cricket=("2005-01-01", "2023-01-01", "2025-01-01"), football=("1995-01-01", "2021-01-01", "2024-01-01"))[sport]
    dates = tuple(pd.Timestamp(x or y) for x, y in zip((args.train_from, args.val_start, args.test_start), d))
    t_from, v_start, te_start = dates
    print(f"\n{'=' * 70}\n{sport.upper()} / {gender.upper()}: {len(df)} matches, {df.date.min().date()} -> "
          f"{df.date.max().date()}, categories={fmts}\n{'=' * 70}")

    base_ha = HOME_ADV[sport]
    leak_test(df, fmts, args.use_toss, base_ha)
    ha, ks = (base_ha, 1.0) if args.no_tune else tune_elo(df, fmts, sport, args.use_toss, dates)
    print(f"  Elo settings: home_adv={ha:.0f}, k_scale={ks}")
    tracker = Tracker(ha, ks)
    X, y_raw, meta = build(df, tracker, fmts, args.use_toss)
    classes = np.unique(y_raw)
    y, k = np.searchsorted(classes, y_raw), len(classes)
    tr, va, te = split_masks(meta, *dates)
    if tr.sum() < 50 or va.sum() < 20:
        return print(f"  not enough data: train={tr.sum()}, validation={va.sum()} matches. "
                     f"Adjust --train-from / --val-start / --test-start (data ends {df.date.max().date()}).")
    foc = ((meta.a == args.focus) | (meta.b == args.focus)).to_numpy()
    print(f"  train {tr.sum()} | val {va.sum()} | test {te.sum()} (focus '{args.focus}' in test: {(te & foc).sum()}, "
          f"in all data: {foc.sum()})")

    cols = list(X.columns)
    models = {"elo_only (LR)": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), ["elo_diff", "home"]),
              "logreg": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), cols),
              "random_forest": (RandomForestClassifier(500, min_samples_leaf=15, n_jobs=-1, random_state=0), cols)}
    best = None
    for depth in (2, 3, 4):                                          # tune on VALIDATION only, early-stopped
        m = make_xgb(n_estimators=1500, max_depth=depth, early_stopping_rounds=50,
                     eval_metric="mlogloss" if k > 2 else "logloss")
        m.fit(X[tr][cols], y[tr], eval_set=[(X[va][cols], y[va])], verbose=False)
        ll = score(y[va], m.predict_proba(X[va][cols]), k)["logloss"]
        print(f"  xgb depth={depth}: val logloss {ll:.4f} at {m.best_iteration + 1} trees")
        if best is None or ll < best[0]:
            best = (ll, depth, m.best_iteration + 1, m)
    models["xgboost"] = (best[3], cols)
    for name, (m, c) in models.items():
        if name != "xgboost":
            m.fit(X[tr][c], y[tr])

    rows = []
    for name, (m, c) in models.items():
        for split, mask in (("val", va), ("test", te), (f"test:{args.focus}", te & foc)):
            if mask.sum():
                rows.append(dict(model=name, split=split, **score(y[mask], m.predict_proba(X[mask][c]), k)))
    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    print("  (brier is summed over classes; classes =", [float(c) / 2 for c in classes], "= score for team A)")
    imp = pd.Series(best[3].feature_importances_, cols).sort_values(ascending=False)
    print("\n  xgboost feature importance:\n" + imp.round(3).head(8).to_string())

    allm = (meta.date >= t_from).to_numpy()          # deployment model: refit on everything since train_from
    final = make_xgb(n_estimators=best[2], max_depth=best[1]).fit(X[allm][cols], y[allm])
    path = Path(f"model_{sport}_{gender}.pkl")
    pickle.dump(dict(model=final, tracker=tracker, cols=cols, classes=classes, fmts=fmts,
                     use_toss=args.use_toss, sport=sport), open(path, "wb"))
    print(f"\n  saved {path} (ratings current through {df.date.max().date()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", choices=["cricket", "football"], default="football")
    ap.add_argument("--data", default=None, help="data folder (default data/<sport>)")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--formats", nargs="+", default=["ODI", "T20I", "Test"], help="cricket formats")
    ap.add_argument("--gender", choices=["male", "female", "both"], default="both",
                    help="men and women are always modelled separately")
    ap.add_argument("--focus", default="Bangladesh", help="report a separate score for this team")
    ap.add_argument("--train-from"); ap.add_argument("--val-start"); ap.add_argument("--test-start")
    ap.add_argument("--no-tune", action="store_true", help="skip Elo home-advantage / K-scale search")
    ap.add_argument("--use-toss", action="store_true", help="cricket: post-toss model (toss is known pre-match)")
    ap.add_argument("--predict", nargs=2, metavar=("TEAM_A", "TEAM_B"))
    ap.add_argument("--fmt", default=None, help="football: friendly|qualifier|regional|major; cricket: ODI|T20I|Test")
    ap.add_argument("--venue", default="")
    ap.add_argument("--home", action="store_true", help="football: TEAM_A is the home side (else neutral)")
    ap.add_argument("--toss-winner"); ap.add_argument("--toss-bat", action="store_true")
    args = ap.parse_args()

    if args.predict:
        if args.gender == "both":
            sys.exit("with --predict choose --gender male or --gender female")
        return predict(args, Path(f"model_{args.sport}_{args.gender}.pkl"))
    root = Path(args.data or f"data/{args.sport}")
    if args.fetch:
        fetch(args.sport, root, args.formats)
    for g in (["male", "female"] if args.gender == "both" else [args.gender]):
        run(args, g, root)


def predict(args, model_path):
    B = pickle.load(open(model_path, "rb"))
    tr, A, Bt = B["tracker"], *args.predict
    for t in (A, Bt):
        if t not in tr.n:
            sys.exit(f"unknown team '{t}' - check spelling against the dataset")
    fmt = args.fmt or B["fmts"][0]
    neutral = not args.home

    def probs(a, b):
        f = tr.features(a, b, fmt, args.venue, neutral)
        for m in B["fmts"]:
            f["fmt_" + m] = float(m == fmt)
        if B["use_toss"]:
            f["toss_a"] = 1.0 if args.toss_winner == a else (-1.0 if args.toss_winner == b else 0.0)
            f["toss_bat"] = float(args.toss_bat)
        p = dict(zip(B["classes"], B["model"].predict_proba(pd.DataFrame([f])[B["cols"]])[0]))
        return p.get(2, 0.0), p.get(1, 0.0), p.get(0, 0.0)          # win, draw, loss for first team

    w, dr, l = probs(A, Bt)
    if neutral:                                                      # average both orientations
        w2, d2, l2 = probs(Bt, A)
        w, dr, l = (w + l2) / 2, (dr + d2) / 2, (l + w2) / 2
    print(f"{A} vs {Bt} [{fmt}]: {A} win {w:.1%}" + (f" | draw {dr:.1%}" if dr else "") + f" | {Bt} win {l:.1%}")
    print(f"Elo: {A} {tr.elo[A]:.0f} | {Bt} {tr.elo[Bt]:.0f}")


if __name__ == "__main__":
    main()
