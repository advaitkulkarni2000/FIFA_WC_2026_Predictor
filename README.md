# WC2026 Predictor — Sequence Model for International Football Outcomes

A deep learning project predicting FIFA World Cup match outcomes, built around testable hypotheses about how international football has changed since the 2006-2010 era — validated through walk-forward testing, then evaluated live against the actual 2026 tournament as it unfolds, and finally used to run a 1,000-iteration Monte Carlo simulation of the full 48-team bracket.

## Why This Project Exists

Most football prediction projects on Kaggle either (a) throw all historical data at a model with no era-awareness, or (b) use simple tabular features with no temporal structure. This project does neither. It is built around five specific, falsifiable hypotheses about the modern game, each translated into a concrete modelling decision:

| Hypothesis | Modelling decision |
|---|---|
| Pre-2014 data (2006, 2010 eras) is less relevant to today's game | Recency-weighted training — older matches contribute less to loss |
| The pool of competitive nations has changed (e.g. Italy hasn't qualified since 2014) | Dynamic team-availability handling — no hardcoded fixed team list |
| Team strength trajectories matter more than static snapshots | Elo/ranking history fed as a *sequence*, not a single number |
| The "winner's curse" (defending champions underperforming) has weakened since 2018 | Explicit `is_defending_champion` x `is_modern_era` interaction feature, empirically tested |
| Recent continental championship success is a current-strength signal the model is otherwise blind to (e.g. Argentina's Copa America 2024 win, ahead of WC2026) | Explicit `won_continental_title_recently` feature, computed purely from pre-tournament historical record across all confederations -- not specific to any one team |

## Critical Methodology Note: No Data Leakage

**Training data: 1990-2022, expanded to all competitive internationals (qualifiers, continental championships, World Cups), not World Cup matches alone.** The 2026 tournament is used purely as a held-out, real-time evaluation set:

1. **Pre-tournament predictions** are generated using only data available before the first ball was kicked (squad strength, qualifying form, Elo entering the tournament, recent continental titles).
2. **Retrospective scoring** then compares those locked-in predictions against actual results as they happen -- since the tournament is ~40% complete, we get immediate, real feedback on calibration.
3. **Monte Carlo tournament simulation** (1,000 full-tournament runs) generates calibrated probability estimates for every team's path through the bracket, rather than presenting one random simulated outcome as "the prediction."

This mirrors the walk-forward validation methodology used in the author's other quantitative research projects: a model is only as trustworthy as its performance on data it never saw during training.

## Data Sources (all free, Kaggle)

- `martj42/international-football-results-from-1872-to-2017` -- full match history, 1872-2026 (includes a real `neutral` venue flag used directly as a feature -- see Bugs Found, below)
- `afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings` -- Elo ratings 1901-2026 for all 48 qualified teams
- `cashncarry/fifaworldranking` -- official FIFA rankings, 1992-2024 (ships as multiple dated CSVs, concatenated during loading)
- `mominullptr/fifa-world-cup-2026-dataset` -- live 2026 match results, updated daily. Fully relational: match and team identities are resolved via foreign-key joins against `teams.csv` and `tournament_stages.csv`, not flat columns.

## Architecture

A sequence model (LSTM) processes each team's recent form (last 12 matches as W/D/L) and Elo trajectory (last 5 years), concatenated with 9 static features (Elo differential, neutral-venue flag, defending-champion flags with modern-era interaction, recent continental title flags) feeding into dense layers ending in a 3-way softmax: home win / draw / away win.

This is a genuine sequence-modelling choice, not a tabular MLP in disguise -- form and momentum are inherently temporal, and treating them as a sequence rather than hand-engineered summary statistics (e.g. "win rate last 10 games") lets the model learn what patterns of recent results actually matter. Both team-form and Elo-trajectory encoders use **shared weights** across home/away teams -- a team's recent form should be evaluated identically regardless of which side of the fixture list it's listed on, a symmetry enforced by design rather than left to chance.

## Validation Strategy

Walk-forward by tournament, trained on the expanded competitive-match dataset, tested on World-Cup-only matches:
- Train on all competitive matches before WC2014 -> test on WC2014
- Train on all competitive matches before WC2018 -> test on WC2018
- Train on all competitive matches before WC2022 -> test on WC2022
- Train on all competitive matches through 2022 (28,343 matches) -> generate WC2026 pre-tournament predictions -> score against real results as they resolve -> run 1,000-iteration Monte Carlo bracket simulation

Evaluated on **log loss and Brier score**, not accuracy -- the goal is calibrated probabilities ("quantify uncertainty"), not just picking winners.

## Results

**Walk-forward validation** (trained on competitive internationals, tested on held-out World Cup matches):

| Tournament | N Matches | Log Loss | Mean Brier | Accuracy |
|---|---|---|---|---|
| 2014 | 64 | 1.0045 | 0.1988 | 0.594 |
| 2018 | 64 | 1.0122 | 0.2032 | 0.562 |
| 2022 | 64 | 0.9837 | 0.1935 | 0.531 |
| **Average** | | **1.0001** | **0.1985** | **0.562** |

Naive baseline (constant historical base rates): log loss ~ 1.02-1.05. The model beats this baseline on every individual fold, not just on average -- a modest (~3% relative) but consistent improvement. Football has a high irreducible randomness component; this magnitude of edge is consistent with published benchmarks for match-outcome prediction, not a sign of an under-performing model.

**Expanding training data from World-Cup-only (~850 matches) to all competitive internationals (28,343 matches)** was the single highest-leverage fix in this project -- see Bugs Found below for what that replaced.

**Monte Carlo tournament simulation** (1,000 full-bracket simulations, pre-tournament features only):

| Team | P(Champion) | P(Final) | P(Semis) | P(Quarters) |
|---|---|---|---|---|
| Spain | 12.9% | 18.4% | 30.2% | 51.6% |
| France | 9.8% | 17.8% | 29.4% | 41.7% |
| Argentina | 9.8% | 18.3% | 25.7% | 35.8% |
| Brazil | 7.8% | 14.7% | 22.9% | 47.5% |
| Netherlands | 6.0% | 8.7% | 20.6% | 33.9% |

No team exceeds a 13% championship probability -- the top 3 favourites combine for only ~33% of simulated outcomes. This spread reflects genuine model uncertainty rather than false confidence, appropriate for a 48-team, heavily single-elimination tournament.

## Bugs Found During Development (and why they're documented, not hidden)

Three real bugs were found and fixed during this project, through systematic testing rather than by accident. Documenting them is part of the honest-evaluation philosophy this project is built on:

1. **`is_neutral_venue` was a silent hardcoded constant.** The feature-assembly function defaulted to `True` for every match, and every call site relied on that default rather than passing the real value -- meaning the model never saw genuine venue variation across ~28,000 training matches, despite the underlying dataset containing a real `neutral` column with a real True/False split (roughly 70/30). Fixed by removing the default entirely (forcing every call site to pass a real value) and wiring in the dataset's actual `neutral` column. Re-validation showed this fix had a negligible effect on aggregate log loss -- informative in itself: it indicates the model's strength-based features (Elo, form) dominate the smaller home-advantage signal, not that the fix was wrong.

2. **2026 host-nation home advantage was not modelled.** Mexico, the USA, and Canada (this tournament's three co-hosts) genuinely play their group-stage matches in their own countries -- real home advantage -- but knockout-round matches for all teams, hosts included, move to shared/neutral venues per FIFA's confirmed 2026 format. The initial prediction pipeline marked every 2026 fixture as neutral, erasing this real signal specifically where it mattered most. Fixed with an explicit `is_host_nation_home_match()` check, gated on both host status and group-stage timing.

3. **Round of 32 bracket construction silently dropped 4 teams and duplicated others.** An early version of the bracket-building logic mismatched which qualifying-team pool fed which matchup slot, resulting in only 14 of the required 16 Round-of-32 matchups and several 4th-place (eliminated) teams incorrectly appearing in the knockout stage. Caught via an assertion-driven test (verifying exactly 32 unique teams across exactly 16 matchups) before any real simulation was run on it -- not discovered after the fact.

## Bracket Simulation -- Documented Scope Decisions

Real FIFA Round-of-32 pairing logic depends on which 8 of 12 possible third-placed teams qualify, with 495 officially defined pairing combinations (Annex C of the tournament regulations). Fully encoding this was judged disproportionate to the project's purpose. Instead: group winners are paired against qualifying third-place teams (8 matchups), remaining group winners pair against each other (2 matchups), and all runners-up pair against each other (6 matchups) -- preserving the correct *structure* (16 matchups, 32 unique teams, no team appearing twice) without the exact administrative pairing table.

Group-stage tiebreaks use points -> goal-difference proxy -> FIFA ranking. Real FIFA tiebreaks also include a disciplinary/conduct-score criterion, which is not modelled -- a deliberate, documented simplification, not an oversight.

Knockout matches cannot end in a draw; the model's predicted draw probability is redistributed proportionally between the two teams based on their relative win probabilities, rather than treated as a true coin flip.

## Project Structure

```
wc2026/
├── notebooks/
│   └── 01_wc2026_predictor.ipynb   # Main Colab notebook, run top to bottom
├── src/
│   ├── data_loader.py               # Kaggle dataset download + merge, relational joins
│   ├── features.py                  # Feature engineering, all five hypotheses
│   ├── model.py                     # LSTM sequence model architecture
│   ├── evaluate.py                  # Walk-forward validation, calibration plots
│   └── bracket.py                   # Group stage, qualification, knockout bracket, Monte Carlo
├── results/
│   ├── monte_carlo_results.csv
│   ├── predictions_df.csv
│   └── walk_forward_summary.txt
├── data/                            # Downloaded datasets land here (gitignored)
└── README.md
```

## Honest Limitations

- Squad-level data (injuries, player availability, current-form changes within the tournament) is not included -- this is team-level only
- Elo and FIFA rankings disagree at times; Elo is used as the primary strength signal (more predictive per academic literature), FIFA ranking as secondary
- The "winner's curse weakening" hypothesis has a sample size of essentially 2 data points (France 2018, Argentina 2022) at the time of writing -- included as a tested, exploratory feature, not presented as a confirmed finding
- Round-of-32 pairing logic is a documented structural simplification of FIFA's actual 495-combination rule (see above)
- Group-stage tiebreaks omit the disciplinary/conduct-score criterion
- Live 2026 data depends on third-party dataset update frequency, not official FIFA feeds
