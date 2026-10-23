# Bangladesh Sports Match Predictor

Predict Bangladesh senior national-team outcomes using chronological
Elo features, logistic regression, random forests and XGBoost.

Inspired by these videos and the creator's Elo-leakage correction:
- https://www.youtube.com/watch?v=LkJpNLIaeVk
- https://www.youtube.com/watch?v=N4JDlSTMOck

## Supported tasks

- Men's and women's football: win/draw/loss, including extra time
  but excluding penalty shootouts.
- Men's and women's ODI/T20I: eventual winner, including super overs
  and bowl-outs. No-results and unresolved ties are excluded.
- Men's Tests: win/draw/loss, retaining draws.
- Women's Tests: insufficient data; only 24 matches are available.

Genders and cricket tasks have separate models and rating pools.
Limited-overs probabilities are conditional on a decisive result.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

On macOS, XGBoost may also require: brew install libomp

Run commands from the project folder.

## Training

    python sports_predictor.py --sport cricket --gender male --cricket-task limited
    python sports_predictor.py --sport cricket --gender female --cricket-task limited
    python sports_predictor.py --sport cricket --gender male --cricket-task test
    python sports_predictor.py --sport football --gender male
    python sports_predictor.py --sport football --gender female

Add --fetch to download source data. Downloads do not guarantee complete
coverage through today. Cricket requires cricket_venue_hosts_expanded.csv.

Model families are selected by validation log loss. The selected model is
then refitted on available history for deployment. Reported test results
belong to the held-out evaluation model, not the full-history refit.

## Predictions

Cricket example:

    python sports_predictor.py --sport cricket --gender male --cricket-task limited --predict Bangladesh India --fmt ODI --venue "Shere Bangla National Stadium" --city Mirpur

For women's cricket, use --gender female.
For men's Tests, use --cricket-task test --fmt Test.

Football example:

    python sports_predictor.py --sport football --gender female --predict Bangladesh Nepal --fmt regional

Football formats: friendly, qualifier, regional, major.
Use --home only for football when the first team is the home side.
Football --venue represents the country recorded in the source.

Cricket home conditions come from reviewed venue/city mappings.
Unmapped locations remain unknown. Each new cricket model stores its
mapping snapshot. These labels describe local conditions, not the
designated host of an overseas series.

The five saved models are pre-toss models. Post-toss models require
separate training with --use-toss. Their predictions also require
--toss-winner TEAM and --toss-decision bat or field.

## Chronological safeguards

- Features precede the corresponding result update.
- Only results completed before the prediction day enter rating state.
- Same-day matches cannot use one another's results.
- Training and validation labels must finish before their fitting cutoff.
- Neutral evaluation and prediction use the same team-order averaging.
- Model selection uses validation, not test performance.
- Invalid formats, identical teams and incomplete toss inputs are rejected.

Timing, boundary, feature and saved-model checks passed during repair.
These checks do not prove the absence of every possible leakage path.

## Recorded retrospective results

Repair runs: 23 October 2026.

| Task | Selected model | Test matches | Accuracy | Log loss |
|---|---|---:|---:|---:|
| Men's ODI/T20I with location | Logistic regression | 1211 | 71.02% | 0.5474 |
| Women's ODI/T20I with location | Logistic regression | 918 | 76.91% | 0.4864 |
| Men's Tests with location | XGBoost | 60 | 65.00% | 0.8615 |
| Men's football | XGBoost | 3034 | 60.51% | 0.8641 |
| Women's football | XGBoost | 1938 | 66.77% | 0.7862 |

Cricket tests start in 2025; football tests start in 2024.

The men's Test validation log loss worsened from 0.9386 to 0.9439
after location changes. Location support has not demonstrated a
validation improvement for that task.

Bangladesh-only test samples are small: 55 men's limited-overs matches,
48 women's limited-overs matches, 11 men's Tests, 21 men's football
matches and 20 women's football matches.

These test results have been inspected during development. Evaluate
future completed matches prospectively with frozen modelling rules.

## Saved models

- model_cricket_male_limited_pretoss.pkl
- model_cricket_female_limited_pretoss.pkl
- model_cricket_male_test_pretoss.pkl
- model_football_male.pkl
- model_football_female.pkl

Completed-result cutoffs:
- Limited-overs cricket: 14 October 2026.
- Men's Tests: 13 October 2026.
- Men's football: 6 October 2026.
- Women's football: 13 October 2026.

A .pkl file stores model data; it is not a terminal command.
Load only trusted model files through the Python script.

## Data and limitations

Cricket: https://cricsheet.org/downloads/
Identifiers: https://cricsheet.org/register/
Coverage: https://cricsheet.org/coverage/
Licensing: https://cricsheet.org/licensing/
Men's football: https://github.com/martj42/international_results
Women's football: https://github.com/martj42/womens-international-results

Cricsheet has coverage gaps and withheld matches. Women's football
coverage is incomplete. This project does not contain every match
ever played or guarantee complete coverage through today.

The partial location mapping has 36 verified venue/city combinations,
covering 413 of 854 Bangladesh records and 1558 of 9943 total records
in the venue audit. Those totals include outcomes excluded from training.

Cricsheet offers BPL data, but club prediction is not implemented here.
Do not mix club wins into national-team rating pools.

Remaining improvements: expand verified venues, add historical player
features, assess calibration and rolling validation, and preserve
immutable source snapshots. Retain source attribution and license notices.
No particular accuracy increase is guaranteed.
