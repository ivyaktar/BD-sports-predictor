# Bangladesh Sports Match Predictor

Predicts match outcomes for **Bangladesh football (men and women)** and **cricket (ODI, T20I, Test)** using Elo ratings and tree-based models (random forest, XGBoost). **Men and women are modelled completely separately**: separate Elo pools, separate tuned settings, separate models and result tables.

## Why this project is leak-safe
Many tutorials get inflated accuracy by accidentally using information from *after* a match (e.g. post-match Elo). Here:
- Every feature for a match is computed only from **earlier** matches, then the state is updated with the result.
- Train / validation / test are split **by date**, never randomly.
- A built-in leak test runs on every execution: it flips one match result and checks that its features and all earlier ones do not change.

## Data
| Sport | Source |
|---|---|
| Cricket | [Cricsheet](https://cricsheet.org) ball-by-ball JSON |
| Football (men) | [martj42/international_results](https://github.com/martj42/international_results) |
| Football (women) | [martj42/womens-international-results](https://github.com/martj42/womens-international-results) |

Data is downloaded by the script (`--fetch`) and is not stored in this repo.

## Features
Overall Elo, Elo by competition tier (friendly / qualifier / regional / major), last 5/10 form, head-to-head, venue record, experience, home advantage, optional toss (cricket). Elo updates are weighted by competition importance and goal margin; the Elo K-scale and home advantage are tuned on the validation set only.

## Setup
```bash
git clone https://github.com/ivyaktar/bd-sports-predictor.git
cd bd-sports-predictor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
# train + evaluate (downloads data first time)
python sports_predictor.py --sport cricket --fetch --formats ODI T20I Test
python sports_predictor.py --sport football --fetch                 # men + women, separate models
python sports_predictor.py --sport football --gender female         # one category only

# predict a match (choose the category)
python sports_predictor.py --sport football --gender male   --predict Bangladesh Nepal --fmt regional
python sports_predictor.py --sport football --gender female --predict Bangladesh Nepal --fmt regional
python sports_predictor.py --sport cricket  --gender male   --predict Bangladesh India --fmt ODI
```

## Results (football, tested on matches from 2024 onward; 3 classes: win / draw / loss)
| Category | Test matches | Elo only (acc / log loss) | XGBoost (acc / log loss) |
|---|---|---|---|
| Men | 3,034 | 60.2% / 0.869 | 60.5% / 0.864 |
| Women | 1,938 | 65.6% / 0.800 | 66.3% / 0.791 |

Tree models add only a small improvement over Elo. Bangladesh-only test sets are tiny (21 men's, 20 women's matches) so they are not reported as a headline number. Cricket results: add yours after running.

## Limitations
International (senior national team) matches only; no club, youth or Bangladesh Premier League data (no open dataset found). The women's dataset is incomplete (many friendlies are missing). No player-level or squad data, no pitch/weather. Small samples for Bangladesh-only evaluation.

## Acknowledgements
Inspired by two tennis-prediction videos (including the creator's correction about Elo data leakage). Data by Cricsheet and martj42.
# BD-sports-predictor
