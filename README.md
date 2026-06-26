# WC2026 Predictor — Sequence Model for International Football Outcomes

A deep learning project predicting FIFA World Cup match outcomes, built around testable hypotheses about how international football has changed since the 2006-2010 era — and evaluated live against the actual 2026 tournament as it unfolds.

## Why This Project Exists

Most football prediction projects on Kaggle either (a) throw all historical data at a model with no era-awareness, or (b) use simple tabular features with no temporal structure. This project does neither. It is built around four specific, falsifiable hypotheses about the modern game, each translated into a concrete modelling decision:

| Hypothesis | Modelling decision |
|---|---|
| Pre-2014 data (2006, 2010 eras) is less relevant to today's game | Recency-weighted training — older matches contribute less to loss |
| The pool of competitive nations has changed (e.g. Italy hasn't qualified since 2014) | Dynamic team-availability handling — no hardcoded fixed team list |
| Team strength trajectories matter more than static snapshots | Elo/ranking history fed as a *sequence*, not a single number |
| The "winner's curse" (defending champions underperforming) has weakened since 2018 | Explicit `is_defending_champion` × `years_since_won` interaction feature, empirically tested |

## Critical Methodology Note: No Data Leakage

**Training data: 1990–2022 World Cups only.** The 2026 tournament is used purely as a held-out, real-time evaluation set:

1. **Pre-tournament predictions** are generated using only data available before the first ball was kicked (squad strength, qualifying form, Elo entering the tournament).
2. **Retrospective scoring** then compares those locked-in predictions against actual results as they happen — since the tournament is ~40% complete, we get immediate, real feedback on calibration.

This mirrors the walk-forward validation methodology used in the author's other quantitative research projects: a model is only as trustworthy as its performance on data it never saw during training.

## Data Sources (all free, Kaggle)

- `martj42/international-football-results-from-1872-to-2017` — full match history, 1872–2026
- `afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings` — Elo ratings 1901–2026 for all 48 qualified teams
- `cashncarry/fifaworldranking` — official FIFA rankings, 1992–2024
- `mominullptr/fifa-world-cup-2026-dataset` — live 2026 match results, updated daily

## Architecture

A sequence model (LSTM) processes each team's recent form (last 10–15 matches as W/D/L) and Elo trajectory (last 5 years), concatenated with static features (current Elo, FIFA ranking, head-to-head record, home/neutral venue, defending-champion flags), feeding into dense layers ending in a 3-way softmax: home win / draw / away win.

This is a genuine sequence-modelling choice, not a tabular MLP in disguise — form and momentum are inherently temporal, and treating them as a sequence rather than hand-engineered summary statistics (e.g. "win rate last 10 games") lets the model learn what patterns of recent results actually matter.

## Validation Strategy

Walk-forward by tournament:
- Train on all data before WC2014 → test on WC2014
- Train on all data before WC2018 (including WC2014) → test on WC2018
- Train on all data before WC2022 (including WC2018) → test on WC2022
- Train on all data through WC2022 → generate WC2026 pre-tournament predictions → score against real results as they resolve

Evaluated on **log loss and Brier score**, not accuracy — the goal is calibrated probabilities ("quantify uncertainty"), not just picking winners.

## Project Structure

```
wc2026/
├── notebooks/
│   └── 01_wc2026_predictor.ipynb   # Main Colab notebook, run top to bottom
├── src/
│   ├── data_loader.py               # Kaggle dataset download + merge
│   ├── features.py                  # Feature engineering, the four hypotheses
│   ├── model.py                     # LSTM sequence model architecture
│   └── evaluate.py                  # Walk-forward validation, calibration plots
├── data/                            # Downloaded datasets land here (gitignored)
└── README.md
```

## Honest Limitations

- Squad-level data (injuries, player form) is not included — this is team-level only
- Elo and FIFA rankings disagree at times; we use Elo as primary (more predictive per academic literature) and FIFA ranking as a secondary feature
- The "winner's curse weakening" hypothesis has a sample size of essentially 2 data points (France 2018, Argentina 2022) — treated as an exploratory finding, not a strong claim
- Live 2026 data depends on third-party dataset update frequency, not official FIFA feeds
