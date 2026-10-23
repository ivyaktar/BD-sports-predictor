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


def load_venue_hosts(path=None):
    """Load reviewed venue/city mappings without guessing missing hosts."""
    import csv

    path = (
        Path(path) if path is not None
        else Path(__file__).resolve().parent
        / "cricket_venue_hosts_expanded.csv"
    )

    hosts = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "venue", "city", "host_team", "venue_country",
            "mapping_source", "review_status",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Venue mapping file has missing columns.")

        for row in reader:
            status = row["review_status"].strip()
            if status == "unknown":
                continue
            if status != "verified":
                raise ValueError(f"Unexpected review status: {status!r}")

            key = (row["venue"].strip(), row["city"].strip())
            host = row["host_team"].strip()

            if not all((
                key[0], host,
                row["venue_country"].strip(),
                row["mapping_source"].strip(),
            )):
                raise ValueError(f"Incomplete verified mapping: {key}")

            if key in hosts and hosts[key] != host:
                raise ValueError(f"Conflicting venue hosts: {key}")

            hosts[key] = host

    return hosts


def load_cricket(root, formats, gender):
    venue_hosts = load_venue_hosts()
    rows = []
    for fmt in formats:
        for p in (root / fmt).glob("*.json"):
            try:
                info = json.loads(p.read_text())["info"]
            except Exception:
                continue
            if gender and info.get("gender") != gender:
                continue
            teams = info.get("teams", [])
            outcome = info.get("outcome", {})
            if len(teams) != 2:
                continue

            winner = (
                outcome.get("winner")
                or outcome.get("eliminator")
                or outcome.get("bowl_out")
            )
            if winner in teams:
                s1 = float(winner == teams[0])
            elif fmt == "Test" and outcome.get("result") == "draw":
                s1 = 0.5
            else:
                # Exclude no-results and unresolved ties.
                continue
            venue = info.get("venue", "")
            city = info.get("city", "")
            host = venue_hosts.get((venue.strip(), city.strip()))

            home_known = (
                host is not None
                and info.get("team_type") == "international"
            )
            neutral = not home_known or host not in teams

            # Existing tracker expects a non-neutral home team first.
            if not neutral and teams[1] == host:
                teams = [teams[1], teams[0]]
                s1 = 1.0 - s1

            toss = info.get("toss", {})
            rows.append(dict(date=pd.to_datetime(info["dates"][0]),
                             result_date=pd.to_datetime(max(info["dates"])),
                             fmt=fmt, team1=teams[0], team2=teams[1],
                             s1=s1, venue=venue, neutral=neutral,
                             home_known=home_known,
                             host_team=host or "", city=city,
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
    df["result_date"] = df["date"]
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
    """Build start-time features; apply only earlier-day completed results."""
    import heapq

    if not df.date.is_monotonic_increasing:
        raise ValueError("Matches must be sorted by start date.")

    flips = (
        np.random.default_rng(seed).random(len(df)) < 0.5
    ) & df.neutral.to_numpy()

    pending = []
    recs, ys, meta = [], [], []

    for i, r in enumerate(df.itertuples(index=False)):
        started = pd.Timestamp(r.date).normalize()
        finished = pd.Timestamp(
            getattr(r, "result_date", r.date)
        ).normalize()

        if pd.isna(started) or pd.isna(finished) or finished < started:
            raise ValueError(f"Invalid match dates at row {i}.")

        # Same-day results remain unavailable until the following day.
        while pending and pending[0][0] < started:
            _, _, update_args = heapq.heappop(pending)
            tracker.update(*update_args)

        a, b, s = r.team1, r.team2, r.s1
        if flips[i]:
            a, b, s = b, a, 1 - s

        f = tracker.features(a, b, r.fmt, r.venue, r.neutral)
        f["home_known"] = float(getattr(r, "home_known", True))
        for m in fmts:
            f["fmt_" + m] = float(m == r.fmt)

        if use_toss:
            f["toss_a"] = (
                1.0 if r.toss_winner == a
                else -1.0 if r.toss_winner == b
                else 0.0
            )
            f["toss_bat"] = float(bool(r.toss_bat))

        recs.append(f)
        ys.append(int(round(s * 2)))
        meta.append((r.date, a, b, finished))

        update_args = (a, b, r.fmt, r.venue, s, r.neutral, r.w)
        heapq.heappush(pending, (finished, i, update_args))

    # Prepare the final tracker for predictions after this dataset ends.
    while pending:
        _, _, update_args = heapq.heappop(pending)
        tracker.update(*update_args)

    return (
        pd.DataFrame(recs),
        np.array(ys),
        pd.DataFrame(meta, columns=["date", "a", "b", "result_date"]),
    )


def leak_test(df, fmts, use_toss, home_adv, n=1500):
    """Check that changing a result cannot change its own or earlier features."""
    sub = df.iloc[:min(n, len(df))].reset_index(drop=True)

    if len(sub) < 2:
        print("Leak check skipped: need at least two matches.")
        return

    # Find a match with at least one participant appearing again later.
    latest_start = {}
    candidates = []
    for j in range(len(sub) - 1, -1, -1):
        teams = {sub.at[j, "team1"], sub.at[j, "team2"]}
        available = pd.Timestamp(
            sub.at[j, "result_date"]
            if "result_date" in sub.columns else sub.at[j, "date"]
        ).normalize()
        if any(
            latest_start.get(team, pd.Timestamp.min) > available
            for team in teams
        ):
            candidates.append(j)
        for team in teams:
            latest_start[team] = max(
                latest_start.get(team, pd.Timestamp.min),
                pd.Timestamp(sub.at[j, "date"]).normalize(),
            )

    if not candidates:
        print("Leak check skipped: no repeated teams in the sample.")
        return

    i = min(candidates, key=lambda j: abs(j - len(sub) // 2))
    changed = sub.copy()
    original = changed.at[i, "s1"]
    changed.at[i, "s1"] = 1.0 if original == 0.5 else 1.0 - original

    before, _, _ = build(sub, Tracker(home_adv), fmts, use_toss)
    after, _, _ = build(changed, Tracker(home_adv), fmts, use_toss)

    if not before.iloc[:i + 1].equals(after.iloc[:i + 1]):
        raise AssertionError(
            "LEAK: a result changed its own or earlier features."
        )

    if before.iloc[i + 1:].equals(after.iloc[i + 1:]):
        raise AssertionError(
            "Changing a result did not affect later repeated-team features."
        )

    print(
        f"Leak check passed: changed match #{i}; "
        "its own and earlier features stayed unchanged."
    )


# ----------------------------------------------------------------------------- modelling
def match_probabilities(model, features):
    """Use the same neutral-match symmetry rule everywhere."""
    probabilities = model.predict_proba(features)

    if not np.array_equal(
        model.classes_, np.arange(probabilities.shape[1])
    ):
        raise ValueError("Unexpected classifier class order.")

    neutral = features["home"].eq(0.0).to_numpy()
    if neutral.any():
        swapped = features.loc[neutral].copy()

        for column in swapped.columns:
            if column.endswith("_diff") or column in {"h2h", "toss_a"}:
                swapped[column] = -swapped[column]

        reversed_probabilities = model.predict_proba(swapped)[:, ::-1]
        probabilities[neutral] = (
            probabilities[neutral] + reversed_probabilities
        ) / 2.0

    return probabilities


def score(y, p, k):
    return dict(acc=accuracy_score(y, p.argmax(1)), logloss=log_loss(y, p, labels=list(range(k))),
                brier=((p - np.eye(k)[y]) ** 2).sum(1).mean(), n=len(y))


def make_xgb(**kw):
    from xgboost import XGBClassifier
    return XGBClassifier(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         reg_lambda=5, n_jobs=-1, random_state=0, **kw)


def split_masks(meta, t_from, v_start, te_start):
    started = pd.to_datetime(meta["date"]).dt.normalize()
    finished = pd.to_datetime(meta["result_date"]).dt.normalize()

    if started.isna().any() or finished.isna().any():
        raise ValueError("Missing match dates.")
    if (finished < started).any():
        raise ValueError("A result date precedes its match start.")

    tr = (
        (started >= t_from)
        & (started < v_start)
        & (finished < v_start)
    ).to_numpy()

    va = (
        (started >= v_start)
        & (started < te_start)
        & (finished < te_start)
    ).to_numpy()

    te = (started >= te_start).to_numpy()
    return tr, va, te


def tune_elo(df, fmts, sport, use_toss, dates):
    """Pick home-advantage and K-scale by validation log loss (never the test set)."""
    grid = [(h, k) for h in (0.0, 50.0, 100.0) for k in (1.0, 1.4, 2.0)]
    best = None
    for h, ks in grid:
        X, y_raw, meta = build(df, Tracker(h, ks), fmts, use_toss)
        classes = np.unique(y_raw)
        y = np.searchsorted(classes, y_raw)
        tr, va, _ = split_masks(meta, *dates)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
        ll = score(y[va], match_probabilities(m, X[va]), len(classes))["logloss"]
        print(f"    home_adv={h:>5.0f} k_scale={ks}: val logloss {ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, h, ks)
    return best[1], best[2]


def run(args, gender, root):
    sport = args.sport
    df = load_football(root, gender) if sport == "football" else load_cricket(root, args.formats, gender)
    if sport == "cricket" and not df.empty:
        wanted = ["Test"] if args.cricket_task == "test" else ["ODI", "T20I"]
        df = df[df.fmt.isin(wanted)].reset_index(drop=True)
    if df.empty:
        return print(f"\nno {gender} {sport} matches found in {root} - skipping")
    fmts = sorted(df.fmt.unique())
    d = dict(cricket=("2005-01-01", "2023-01-01", "2025-01-01"), football=("1995-01-01", "2021-01-01", "2024-01-01"))[sport]
    dates = tuple(pd.Timestamp(x or y) for x, y in zip((args.train_from, args.val_start, args.test_start), d))
    t_from, v_start, te_start = dates
    print(f"\n{'=' * 70}\n{sport.upper()} / {gender.upper()}: {len(df)} matches, {df.date.min().date()} -> "
          f"{df.date.max().date()}, categories={fmts}\n{'=' * 70}")

    # Validate partitions before leak checks, tuning or model fitting.
    if not (t_from < v_start < te_start):
        raise SystemExit(
            "Dates must satisfy: train-from < val-start < test-start"
        )

    raw_tr, raw_va, raw_te = split_masks(df, *dates)
    train_rows = df[raw_tr]
    val_rows = df[raw_va]
    test_rows = df[raw_te]

    if len(train_rows) < 50 or len(val_rows) < 20:
        print(
            f"Skipping: insufficient data. Train={len(train_rows)}, "
            f"validation={len(val_rows)}, test={len(test_rows)}. "
            "Need at least 50 training and 20 validation matches."
        )
        return

    if test_rows.empty:
        print("Skipping: no test matches after the selected test-start date.")
        return

    expected = (
        {0.0, 1.0}
        if sport == "cricket" and args.cricket_task == "limited"
        else {0.0, 0.5, 1.0}
    )
    observed = set(train_rows.s1.unique())
    missing = expected - observed
    if missing:
        print(
            f"Skipping: training data lacks required outcome scores: "
            f"{sorted(missing)}"
        )
        return

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
        ll = score(y[va], match_probabilities(m, X[va][cols]), k)["logloss"]
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
                rows.append(dict(model=name, split=split, **score(y[mask], match_probabilities(m, X[mask][c]), k)))
    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    print("  (brier is summed over classes; classes =", [float(c) / 2 for c in classes], "= score for team A)")
    imp = pd.Series(best[3].feature_importances_, cols).sort_values(ascending=False)
    print("\n  xgboost feature importance:\n" + imp.round(3).head(8).to_string())

    # Select using validation only, never test performance.
    validation_losses = {}
    for name, (candidate, candidate_cols) in models.items():
        probabilities = match_probabilities(
            candidate, X[va][candidate_cols]
        )
        validation_losses[name] = score(
            y[va], probabilities, k
        )["logloss"]

    selected_name = min(validation_losses, key=validation_losses.get)
    selected_model, selected_cols = models[selected_name]
    selected_cols = list(selected_cols)

    print(
        f"\\nSelected: {selected_name}; validation log loss "
        f"{validation_losses[selected_name]:.4f}"
    )

    # Deployment refit is separate from held-out evaluation.
    allm = (meta.date >= t_from).to_numpy()

    if selected_name == "xgboost":
        final = make_xgb(
            n_estimators=best[2], max_depth=best[1]
        )
    else:
        from sklearn.base import clone
        final = clone(selected_model)

    final.fit(X[allm][selected_cols], y[allm])

    cutoff = pd.Timestamp(df["result_date"].max()).date().isoformat()
    path = model_path_for(args, gender)

    bundle = dict(
        model=final,
        tracker=tracker,
        cols=selected_cols,
        classes=classes,
        fmts=fmts,
        use_toss=args.use_toss,
        sport=sport,
        gender=gender,
        cricket_task=getattr(args, "cricket_task", None)
            if sport == "cricket" else None,
        model_name=selected_name,
        validation_logloss=validation_losses[selected_name],
        candidate_validation_logloss=validation_losses,
        result_cutoff=cutoff,
        schema_version=3,
        venue_hosts=load_venue_hosts() if sport == "cricket" else {},
    )

    with path.open("wb") as handle:
        pickle.dump(bundle, handle)

    print(f"Saved {path}; completed results through {cutoff}.")


def model_path_for(args, gender):
    if args.sport == "cricket":
        timing = "posttoss" if args.use_toss else "pretoss"
        return Path(
            f"model_cricket_{gender}_{args.cricket_task}_{timing}.pkl"
        )
    return Path(f"model_football_{gender}.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", choices=["cricket", "football"], default="football")
    ap.add_argument("--data", default=None, help="data folder (default data/<sport>)")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument(
        "--cricket-task",
        choices=["limited", "test"],
        default="limited",
        help="limited = ODI/T20I; test = Test cricket",
    )
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
    ap.add_argument("--city", default="")
    ap.add_argument("--home", action="store_true", help="football: TEAM_A is the home side (else neutral)")
    ap.add_argument("--toss-winner")
    ap.add_argument("--toss-decision", choices=["bat", "field"])
    args = ap.parse_args()

    if args.sport == "football" and (
        args.use_toss or args.toss_winner or args.toss_decision
    ):
        ap.error("Toss options apply only to cricket.")

    if args.sport == "cricket" and args.home:
        ap.error(
            "Cricket --home is unavailable until home/away training "
            "features are implemented."
        )

    if args.sport == "cricket" and args.predict:
        if args.use_toss:
            if args.toss_winner not in args.predict:
                ap.error("--toss-winner must name one of the two teams.")
            if args.toss_decision is None:
                ap.error("--use-toss requires --toss-decision bat or field.")
        elif args.toss_winner or args.toss_decision:
            ap.error("Toss details require --use-toss.")

    if args.predict:
        if args.gender == "both":
            sys.exit("with --predict choose --gender male or --gender female")
        return predict(args, model_path_for(args, args.gender))
    root = Path(args.data or f"data/{args.sport}")
    if args.fetch:
        fetch(args.sport, root, args.formats)
    for g in (["male", "female"] if args.gender == "both" else [args.gender]):
        run(args, g, root)


def prediction_context(bundle, team_a, team_b, venue="", city="", home=False):
    """Return ordered teams, neutral, location-known and reversal flag."""
    if bundle["sport"] == "football":
        return team_a, team_b, not home, True, False

    # Older models were trained without the new location features.
    if "venue_hosts" not in bundle:
        return team_a, team_b, True, False, False

    hosts = bundle["venue_hosts"]
    venue, city = venue.strip(), city.strip()

    if city:
        host = hosts.get((venue, city))
    else:
        candidates = {
            host for (name, _), host in hosts.items()
            if name == venue
        }
        host = next(iter(candidates)) if len(candidates) == 1 else None

    if host is None:
        return team_a, team_b, True, False, False
    if host == team_a:
        return team_a, team_b, False, True, False
    if host == team_b:
        return team_b, team_a, False, True, True

    return team_a, team_b, True, True, False


def predict(args, model_path):
    if not model_path.is_file():
        raise SystemExit(
            f"Model not found: {model_path}. Train this task first."
        )
    with model_path.open("rb") as handle:
        B = pickle.load(handle)
    tr, A, Bt = B["tracker"], *args.predict
    for t in (A, Bt):
        if t not in tr.n:
            sys.exit(f"unknown team '{t}' - check spelling against the dataset")
    if A == Bt:
        raise SystemExit("Choose two different teams.")

    if args.fmt is None:
        raise SystemExit(
            "Specify --fmt. Supported formats: " + ", ".join(B["fmts"])
        )

    if args.fmt not in B["fmts"]:
        raise SystemExit(
            f"Unsupported format: {args.fmt}. Choose from: "
            + ", ".join(B["fmts"])
        )

    fmt = args.fmt
    ordered_a, ordered_b, neutral, home_known, reversed_order = (
        prediction_context(
            B, A, Bt, args.venue, args.city, args.home
        )
    )

    def probs(a, b):
        f = tr.features(a, b, fmt, args.venue, neutral)
        f["home_known"] = float(home_known)
        for m in B["fmts"]:
            f["fmt_" + m] = float(m == fmt)
        if B["use_toss"]:
            f["toss_a"] = 1.0 if args.toss_winner == a else (-1.0 if args.toss_winner == b else 0.0)
            f["toss_bat"] = float(args.toss_decision == "bat")
        p = dict(zip(B["classes"], match_probabilities(B["model"], pd.DataFrame([f])[B["cols"]])[0]))
        return p.get(2, 0.0), p.get(1, 0.0), p.get(0, 0.0)          # win, draw, loss for first team

    w, dr, l = probs(ordered_a, ordered_b)
    if reversed_order:
        w, l = l, w
    print(f"{A} vs {Bt} [{fmt}]: {A} win {w:.1%}" + (f" | draw {dr:.1%}" if dr else "") + f" | {Bt} win {l:.1%}")
    print(f"Elo: {A} {tr.elo[A]:.0f} | {Bt} {tr.elo[Bt]:.0f}")

    if B["sport"] == "football":
        location = f"{A} home" if args.home else "neutral (CLI default)"
    elif "venue_hosts" not in B:
        location = "unknown; legacy model needs retraining for location support"
    elif not home_known:
        location = "unknown; venue/city is absent or unmapped"
    elif neutral:
        location = "verified neutral local conditions"
    else:
        location = f"{ordered_a} local home conditions"

    print(f"Location: {location}")
    print(f"Completed results through: {B.get('result_cutoff', 'unrecorded')}")


if __name__ == "__main__":
    main()
