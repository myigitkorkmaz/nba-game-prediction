# Predicting NBA Game Outcomes with Machine Learning

CPS493-01 Machine Learning, Fall 2026 · Mehmet Yigit Korkmaz

Predict whether the home team wins an NBA regular-season game using only information available before tip-off.

I wanted to make clear that I used CLAUDE(OPUS 5.5) for the coding part of this project. I have deep understanding of how code and implemented algorithims work and managed the whole development process by myself such as prompting for;

    *Picking to apply logistic regression.
    *Picking to apply decision trees.
    *Implementing Dean Olivers Statistics method.
    *Adding bubble games. 
    *Using 2010–2024 data instead of the proposal’s 2004–2020 Kaggle set so the test seasons include the most recent NBA.
    *Three baselines.The models have to beat more than just the easiest guess (home is favorite %55).
    *Using FiveThirtyEight’s elo method as training/testing data.

Please let me know if you think certain part of this code needs to be hardcodede by myself so I can fix it before other report. Thank you!
    

## Data

`data/raw/regular_season_totals_2010_2024.csv`: team box scores for every
regular-season game from 2010-11 to 2023-24 (33,316 team-games = 16,658 games),
originally pulled from stats.nba.com. Source:
https://github.com/NocturneBear/NBA-Data-2010-2024

## Pipeline

```bash
pip install -r requirements.txt
python src/clean.py      # step 1: clean + pair home/away rows -> data/processed/games.csv
python src/features.py   # step 2: pre-game features             -> data/processed/features.csv
python src/train.py      # step 3: tune, train, test, figures     -> results/
```

Runs in about 2 minutes on a laptop (`train.py` does the hyperparameter search).

| Step | What it does |
| --- | --- |
| `clean.py` | Checks duplicates, missing values and ties; finds the home team from the matchup string; joins both teams into one row per game; flags overtime and the 88 games of the 2020 Orlando bubble. |
| `features.py` | Builds 19 pre-game features: home-minus-away Elo, season win % and net rating, last-10-game form (win %, margin, offensive/defensive/net rating, pace, four factors), rest days, games in the last 7 days, back-to-backs. Every rolling stat is shifted one game so a game never sees its own result. |
| `train.py` | Train on 2010-11 to 2018-19, test on 2019-20 to 2023-24. Tunes hyperparameters with forward-chaining validation (validate on 2016-17, 2017-18, 2018-19). Reports accuracy, log loss, Brier score and AUC against three baselines, per-season results, home-court rates, permutation feature importance and an ablation. |

## Preliminary results (test seasons 2019-20 to 2023-24, 5,050 games)

| Model | Accuracy | Log loss | AUC |
| --- | --- | --- | --- |
| Home team always wins | 55.2% | | |
| Better record wins | 62.8% | | |
| Elo favorite wins | 63.9% | 0.640 | 0.694 |
| Logistic regression | 64.7% | 0.627 | 0.701 |
| Random forest | 64.8% | 0.632 | 0.695 |

All outputs are in `results/` (CSV tables, `metrics.json`, figures).
