# Bangladesh Sports Match Predictor

This project predicts the outcomes of Bangladesh senior national-team matches. It uses Elo ratings built in chronological order, plus logistic regression, random forest and XGBoost models.

It was inspired by two videos, including the creator's correction of an Elo data-leakage problem:
- https://www.youtube.com/watch?v=LkJpNLIaeVk
- https://www.youtube.com/watch?v=N4JDlSTMOck

## What it can predict

- **Men's and women's football:** win, draw or loss. Extra time counts, but penalty shootouts don't.
- **Men's and women's ODI/T20I cricket:** the eventual winner, including super overs and bowl-outs. No-results and unresolved ties are left out.
- **Men's Tests:** win, draw or loss. Draws are kept.
- **Women's Tests:** not supported, because only 24 matches are available, which isn't enough data.

Each gender and each cricket task has its own model and its own rating pool. For limited-overs cricket, probabilities assume the match produces a decisive result.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

On macOS, XGBoost may also need libomp: `brew install libomp`

Run all commands from the project folder.

## Training

    python sports_predictor.py --sport cricket --gender male --cricket-task limited
    python sports_predictor.py --sport cricket --gender female --cricket-task limited
    python sports_predictor.py --sport cricket --gender male --cricket-task test
    python sports_predictor.py --sport football --gender male
    python sports_predictor.py --sport football --gender female

Add `--fetch` to download the source data. Keep in mind that a download doesn't guarantee complete coverage up to today. Cricket also needs the file `cricket_venue_hosts_expanded.csv`.

The model family is chosen by validation log loss. The chosen model is then refitted on all available history for deployment. The test results reported below come from the held-out evaluation model, not from this full-history refit.

## Making predictions

Cricket example:

    python sports_predictor.py --sport cricket --gender male --cricket-task limited --predict Bangladesh India --fmt ODI --venue "Shere Bangla National Stadium" --city Mirpur

For women's cricket, use `--gender female`. For men's Tests, use `--cricket-task test --fmt Test`.

Football example:

    python sports_predictor.py --sport football --gender female --predict Bangladesh Nepal --fmt regional

The football formats are friendly, qualifier, regional and major. Use `--home` only for football, and only when the first team is the home side. In football, `--venue` means the country recorded in the source data.

Cricket home conditions come from reviewed venue and city mappings. Any location that isn't mapped stays unknown, and each new cricket model saves a snapshot of the mapping it used. These labels describe local conditions, not which team officially hosted an overseas series.

The five saved models are all pre-toss models. Post-toss models have to be trained separately with `--use-toss`, and predicting with them also requires `--toss-winner TEAM` and `--toss-decision bat` or `field`.

## Safeguards against leakage

- Features are computed before the result they correspond to is added to the ratings.
- Only results completed before the prediction day feed into the rating state.
- Matches played on the same day can't use each other's results.
- Training and validation labels must be finished before the cutoff used for fitting.
- Evaluation and prediction on neutral ground use the same team-order averaging.
- Models are selected on validation performance, never on test performance.
- Invalid formats, identical teams and incomplete toss inputs are rejected.

Checks on timing, boundaries, features and saved models passed during the repair. They don't prove that every possible leakage path has been closed.

## Recorded retrospective results

These come from the repair runs on 23 October 2026.

| Task | Selected model | Test matches | Accuracy | Log loss |
|---|---|---:|---:|---:|
| Men's ODI/T20I with location | Logistic regression | 1211 | 71.02% | 0.5474 |
| Women's ODI/T20I with location | Logistic regression | 918 | 76.91% | 0.4864 |
| Men's Tests with location | XGBoost | 60 | 65.00% | 0.8615 |
| Men's football | XGBoost | 3034 | 60.51% | 0.8641 |
| Women's football | XGBoost | 1938 | 66.77% | 0.7862 |

Cricket test periods begin in 2025, and football test periods begin in 2024.

For men's Tests, validation log loss got slightly worse after the location changes, going from 0.9386 to 0.9439. So location support hasn't been shown to improve validation results for that task.

The Bangladesh-only test samples are small: 55 men's limited-overs matches, 48 women's limited-overs matches, 11 men's Tests, 21 men's football matches and 20 women's football matches.

These test results were looked at during development. To judge the models fairly from here on, evaluate them on future completed matches, with the modelling rules frozen.

## Saved models

- model_cricket_male_limited_pretoss.pkl
- model_cricket_female_limited_pretoss.pkl
- model_cricket_male_test_pretoss.pkl
- model_football_male.pkl
- model_football_female.pkl

Each model includes results completed up to these cutoffs:
- Limited-overs cricket: 14 October 2026
- Men's Tests: 13 October 2026
- Men's football: 6 October 2026
- Women's football: 13 October 2026

A .pkl file holds model data and isn't something you run in the terminal. Only load model files you trust, and load them through the Python script.

## Data and limitations

- Cricket: https://cricsheet.org/downloads/
- Identifiers: https://cricsheet.org/register/
- Coverage: https://cricsheet.org/coverage/
- Licensing: https://cricsheet.org/licensing/
- Men's football: https://github.com/martj42/international_results
- Women's football: https://github.com/martj42/womens-international-results

Cricsheet has coverage gaps and withheld matches, and women's football coverage is incomplete. This project doesn't contain every match ever played, and it doesn't guarantee complete coverage through today.

The location mapping is only partial. It has 36 verified venue/city combinations, covering 413 of 854 Bangladesh records and 1558 of 9943 total records in the venue audit. Those totals include outcomes that were excluded from training.

Cricsheet also provides BPL data, but club prediction isn't implemented here. Don't mix club wins into the national-team rating pools.

Remaining improvements:
- Expand the set of verified venues.
- Add historical player features.
- Assess calibration and rolling validation.
- Keep immutable snapshots of the source data.
- Retain source attribution and license notices.

No particular increase in accuracy is guaranteed.

## Acknowledgements

Inspired by two tennis-prediction videos, including the creator's correction about Elo data leakage. Data comes from Cricsheet and martj42.