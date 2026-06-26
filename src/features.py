"""
Feature Engineering — The Four Hypotheses
--------------------------------------------
Each function here translates one of the project's specific hypotheses
into a concrete, testable feature. This is the heart of what makes this
project more than "download Kaggle data, fit a model."

Hypothesis 1: Pre-2014 data is less relevant -> recency weighting
Hypothesis 2: Team pool changes over time -> dynamic availability, no hardcoding
Hypothesis 3: Trajectory matters more than snapshot -> Elo/form as sequences
Hypothesis 4: Winner's curse has weakened since 2018 -> explicit interaction feature
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime


# ── HYPOTHESIS 1: Recency weighting ────────────────────────────────────────

def compute_recency_weight(
    match_date: pd.Series,
    reference_date: pd.Timestamp,
    half_life_years: float = 8.0,
) -> pd.Series:
    """
    Exponential decay weight: matches from `half_life_years` ago get half
    the weight of a match today. This is the concrete, defensible version
    of "2006/2010 are less relevant" — not a hard cutoff, but a smooth decay
    so the model isn't crudely discarding information, just discounting it.

    Args:
        match_date: Series of match dates
        reference_date: the "today" anchor (e.g. start of WC being predicted)
        half_life_years: years for weight to halve

    Returns:
        Series of weights in (0, 1], 1.0 for matches on reference_date
    """
    days_ago = (reference_date - match_date).dt.days
    years_ago = days_ago / 365.25
    decay_rate = np.log(2) / half_life_years
    weights = np.exp(-decay_rate * years_ago.clip(lower=0))
    return weights


# ── HYPOTHESIS 2: Dynamic team availability ────────────────────────────────

def get_active_teams_at_tournament(
    matches_long: pd.DataFrame,
    tournament_year: int,
    lookback_years: int = 4,
) -> set:
    """
    Returns the set of teams that have actually played international
    matches in the lookback window before a given tournament.

    This deliberately avoids any hardcoded team list. Italy not qualifying
    since 2014 isn't special-cased — it simply won't appear in the relevant
    "qualified teams" set for tournaments after 2014, because we derive
    participation from the data, not from a fixed roster the model assumes
    is constant.
    """
    cutoff_start = pd.Timestamp(f"{tournament_year - lookback_years}-01-01")
    cutoff_end = pd.Timestamp(f"{tournament_year}-01-01")
    window = matches_long[
        (matches_long["date"] >= cutoff_start) & (matches_long["date"] < cutoff_end)
    ]
    return set(window["team"].unique())


# ── HYPOTHESIS 3: Trajectory as sequence ───────────────────────────────────

def build_recent_form_sequence(
    matches_long: pd.DataFrame,
    team: str,
    as_of_date: pd.Timestamp,
    n_matches: int = 12,
) -> list[int]:
    """
    Build a fixed-length sequence encoding a team's last n_matches results
    BEFORE as_of_date, as integers: Win=2, Draw=1, Loss=0.

    Padded with -1 (a distinct "no data" token) if fewer than n_matches
    exist — the model's embedding layer will learn to treat -1 specially,
    rather than us silently injecting a fake "average" result.

    This is what feeds the LSTM — actual sequential structure, not a single
    "win rate last 10 games" scalar that throws away the ORDER of results.
    """
    team_history = matches_long[
        (matches_long["team"] == team) & (matches_long["date"] < as_of_date)
    ].sort_values("date", ascending=False).head(n_matches)

    result_map = {"W": 2, "D": 1, "L": 0}
    seq = [result_map[r] for r in team_history["result"]]
    seq = seq[::-1]  # chronological order (oldest of the window first)

    # Pad on the left with -1 if not enough history
    if len(seq) < n_matches:
        seq = [-1] * (n_matches - len(seq)) + seq

    return seq


def build_elo_trajectory(
    elo_df: pd.DataFrame,
    team: str,
    as_of_year: int,
    n_years: int = 5,
) -> list[float]:
    """
    Build a sequence of a team's Elo rating for each of the last n_years,
    ending the year BEFORE as_of_year (no leakage of the current year).

    Returns ratings in chronological order. Missing years are forward-filled
    from the nearest available prior year, since Elo doesn't reset to zero
    when data is sparse for newer/smaller federations.
    """
    team_elo = elo_df[
        (elo_df["team"] == team) & (elo_df["year"] < as_of_year)
    ].sort_values("year")

    years_wanted = list(range(as_of_year - n_years, as_of_year))
    ratings = []
    last_known = 1500.0  # neutral default Elo if NO history exists at all

    for y in years_wanted:
        row = team_elo[team_elo["year"] == y]
        if len(row) > 0:
            last_known = float(row.iloc[-1]["rating"])
        ratings.append(last_known)

    return ratings


# ── HYPOTHESIS 4: Winner's curse interaction ───────────────────────────────

def compute_defending_champion_features(
    team: str,
    tournament_year: int,
    past_winners: dict[int, str],
) -> dict:
    """
    Returns features capturing the "defending champion" effect and whether
    it has weakened in recent eras.

    past_winners: {year: winning_team_name}, e.g. {2018: 'France', 2022: 'Argentina'}

    is_defending_champion: 1 if this team won the immediately preceding WC
    years_since_any_title: years since this team's most recent WC win (any year)
    is_modern_era: 1 if tournament_year >= 2018 — lets the model learn a
                   SEPARATE defending-champion effect for the modern era,
                   rather than assuming the historical "winner's curse"
                   pattern still holds unchanged.
    """
    prior_years = [y for y in past_winners if y < tournament_year]

    if not prior_years:
        # No World Cup before this one in our records (e.g. this tournament
        # IS the earliest one in past_winners) -- there is no "defending
        # champion" concept yet. Return neutral/zero values rather than
        # crashing or silently dropping the match.
        is_defending_champion = 0
    else:
        previous_wc_year = max(prior_years)
        is_defending_champion = int(past_winners.get(previous_wc_year) == team)

    team_titles = [y for y, w in past_winners.items() if w == team and y < tournament_year]
    years_since_any_title = (
        tournament_year - max(team_titles) if team_titles else 999  # "never won"
    )

    return {
        "is_defending_champion": is_defending_champion,
        "years_since_any_title": years_since_any_title,
        "is_modern_era": int(tournament_year >= 2018),
        # Explicit interaction term — lets a linear-ish layer pick up
        # "defending champion AND modern era" as a distinct effect from
        # "defending champion" alone.
        "defending_champion_x_modern_era": is_defending_champion * int(tournament_year >= 2018),
    }


# ── HYPOTHESIS 5: Recent continental title as a strength signal ───────────
# Rationale: the model's training data (WC history, Elo, FIFA rankings) has
# NO visibility into continental championships. But these are the most
# recent, highest-stakes competitive evidence available before a World Cup --
# squads change meaningfully between WC cycles, and a team winning their
# continental title 1-2 years before a WC is a real, current strength signal
# the model would otherwise be blind to. This generalises across confederations
# (not just "Argentina specifically") -- any team, any era, recent continental
# success gets the same flag.

CONTINENTAL_CHAMPIONS = {
    # (confederation, year): winning_team
    ("CONMEBOL", 2019): "Brazil",
    ("CONMEBOL", 2021): "Argentina",
    ("CONMEBOL", 2024): "Argentina",
    ("UEFA", 2016): "Portugal",
    ("UEFA", 2020): "Italy",
    ("UEFA", 2024): "Spain",
    ("CAF", 2019): "Algeria",
    ("CAF", 2021): "Senegal",       # 2021 AFCON, played Jan 2022
    ("CAF", 2023): "Ivory Coast",   # played Jan-Feb 2024
    ("AFC", 2019): "Qatar",
    ("AFC", 2023): "Qatar",         # played Jan-Feb 2024
    ("CONCACAF", 2021): "United States",  # 2021 Gold Cup
    ("CONCACAF", 2023): "Mexico",
    ("CONCACAF", 2025): "Mexico",
}


def compute_recent_continental_title_feature(
    team: str,
    tournament_year: int,
    continental_champions: dict = CONTINENTAL_CHAMPIONS,
    recency_years: int = 2,
) -> dict:
    """
    Returns whether `team` won ANY continental championship within
    `recency_years` years before the World Cup, plus years-since for a
    smoother continuous signal.

    This is computed PURELY from historical, pre-tournament record --
    e.g. Copa America 2024 result, known well before WC2026 started.
    No in-tournament 2026 information is used here. Critical for keeping
    the "no leakage" guarantee intact everywhere else in this project.
    """
    titles_for_team = [
        year for (_, year), winner in continental_champions.items()
        if winner == team and year < tournament_year
    ]

    if not titles_for_team:
        return {
            "won_continental_title_recently": 0,
            "years_since_continental_title": 999,
        }

    most_recent = max(titles_for_team)
    years_since = tournament_year - most_recent

    return {
        "won_continental_title_recently": int(years_since <= recency_years),
        "years_since_continental_title": years_since,
    }


# ── Static feature assembly ────────────────────────────────────────────────

def assemble_match_features(
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    matches_long: pd.DataFrame,
    elo_df: pd.DataFrame,
    fifa_rank_df: pd.DataFrame,
    past_winners: dict[int, str],
    is_neutral_venue: bool,
) -> dict:
    """
    Assemble the full feature dict for one match, combining all four
    hypothesis-driven feature groups. This is what gets fed to the model
    for a single training example (or a single prediction at inference time).
    """
    tournament_year = match_date.year

    home_form_seq = build_recent_form_sequence(matches_long, home_team, match_date)
    away_form_seq = build_recent_form_sequence(matches_long, away_team, match_date)

    home_elo_traj = build_elo_trajectory(elo_df, home_team, tournament_year)
    away_elo_traj = build_elo_trajectory(elo_df, away_team, tournament_year)

    home_champ_features = compute_defending_champion_features(
        home_team, tournament_year, past_winners
    )
    away_champ_features = compute_defending_champion_features(
        away_team, tournament_year, past_winners
    )

    home_continental_features = compute_recent_continental_title_feature(
        home_team, tournament_year
    )
    away_continental_features = compute_recent_continental_title_feature(
        away_team, tournament_year
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": match_date,
        "home_form_sequence": home_form_seq,
        "away_form_sequence": away_form_seq,
        "home_elo_trajectory": home_elo_traj,
        "away_elo_trajectory": away_elo_traj,
        "elo_diff": home_elo_traj[-1] - away_elo_traj[-1],
        "is_neutral_venue": int(is_neutral_venue),
        **{f"home_{k}": v for k, v in home_champ_features.items()},
        **{f"away_{k}": v for k, v in away_champ_features.items()},
        **{f"home_{k}": v for k, v in home_continental_features.items()},
        **{f"away_{k}": v for k, v in away_continental_features.items()},
    }
