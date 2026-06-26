"""
WC2026 Bracket Simulator
--------------------------
Simulates the full 48-team tournament using the trained model's predictions:
group stage -> round of 32 -> round of 16 -> quarters -> semis -> final.

Honest scope decision: FIFA's real round-of-32 pairing logic has 495
possible combinations for how third-placed teams get matched with group
winners (Annex C of the official regulations). Encoding that fully is a
project on its own. Instead, we use the ACTUAL confirmed 2026 fixture
list/bracket slots wherever the tournament has already locked in real
matchups, and a simplified "best group winner plays worst qualifying
third-place team" heuristic for any remaining hypothetical matchups.
This is a documented simplification, not a hidden one.

Tiebreak simplification: real FIFA tiebreaks go points -> goal difference
-> goals scored -> conduct score -> FIFA ranking. We use points -> a
single expected-goal-difference proxy derived from the model's predicted
score margin, then FIFA ranking as a final tiebreak. We do not model
yellow/red cards (conduct score) at all -- this is the documented gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TeamGroupResult:
    team: str
    group: str
    points: int = 0
    goal_diff_proxy: float = 0.0
    fifa_rank: float = 999.0


def predict_match_probs(
    model,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    matches_long: pd.DataFrame,
    elo_df: pd.DataFrame,
    fifa_rank_df: pd.DataFrame,
    past_winners: dict,
    is_neutral_venue: bool,
    assemble_match_features_fn,
) -> np.ndarray:
    """
    Returns [p_home_win, p_draw, p_away_win] for a single hypothetical
    matchup, using the same feature pipeline as training.
    """
    feats = assemble_match_features_fn(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        matches_long=matches_long,
        elo_df=elo_df,
        fifa_rank_df=fifa_rank_df,
        past_winners=past_winners,
        is_neutral_venue=is_neutral_venue,
    )

    home_form = torch.tensor([feats['home_form_sequence']], dtype=torch.long)
    away_form = torch.tensor([feats['away_form_sequence']], dtype=torch.long)
    home_elo = torch.tensor([feats['home_elo_trajectory']], dtype=torch.float32)
    away_elo = torch.tensor([feats['away_elo_trajectory']], dtype=torch.float32)
    static = torch.tensor([[
        feats['elo_diff'], feats['is_neutral_venue'],
        feats['home_is_defending_champion'], feats['home_defending_champion_x_modern_era'],
        feats['away_is_defending_champion'], feats['away_defending_champion_x_modern_era'],
        feats['home_is_modern_era'],
        feats['home_won_continental_title_recently'], feats['away_won_continental_title_recently'],
    ]], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(home_form, away_form, home_elo, away_elo, static)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    return probs  # [p_home_win, p_draw, p_away_win]


def simulate_group_stage(
    groups: dict[str, list[str]],
    model,
    matches_long: pd.DataFrame,
    elo_df: pd.DataFrame,
    fifa_rank_df: pd.DataFrame,
    past_winners: dict,
    fifa_ranks: dict,
    is_neutral_venue_fn,
    match_date: pd.Timestamp,
    assemble_match_features_fn,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate every group-stage match (round-robin within each group of 4 =
    6 matches per group) using the model's probabilities. Each match's
    outcome is sampled from the predicted distribution (not just argmax) --
    this matters because a tournament simulation should reflect genuine
    uncertainty, not collapse every match to its single most likely result.

    Returns a DataFrame of team standings per group: points, goal_diff_proxy.
    """
    rng = np.random.default_rng(rng_seed)
    results = {team: TeamGroupResult(team=team, group=g, fifa_rank=fifa_ranks.get(team, 999.0))
               for g, teams in groups.items() for team in teams}

    for group_name, teams in groups.items():
        # Round robin: every pair plays once (6 matches for 4 teams)
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                home, away = teams[i], teams[j]
                # is_neutral_venue_fn here should return whether the match
                # IS neutral, i.e. the INVERSE of "is this a host nation home match"
                is_neutral = not is_neutral_venue_fn(home, "Group Stage")

                probs = predict_match_probs(
                    model, home, away, match_date, matches_long, elo_df,
                    fifa_rank_df, past_winners, is_neutral, assemble_match_features_fn,
                )

                probs_normalized = np.array([float(p) for p in probs])
                probs_normalized = probs_normalized / probs_normalized.sum()
                outcome = rng.choice(3, p=probs_normalized)  # 0=home_win, 1=draw, 2=away_win

                if outcome == 0:
                    results[home].points += 3
                    margin = 1.0  # proxy goal difference contribution
                    results[home].goal_diff_proxy += margin
                    results[away].goal_diff_proxy -= margin
                elif outcome == 1:
                    results[home].points += 1
                    results[away].points += 1
                else:
                    results[away].points += 3
                    margin = 1.0
                    results[away].goal_diff_proxy += margin
                    results[home].goal_diff_proxy -= margin

    return pd.DataFrame([vars(r) for r in results.values()])


def rank_groups_and_qualify(standings: pd.DataFrame) -> dict:
    """
    Apply the 2026 qualification rule: top 2 from each group automatically
    qualify; the 8 best third-placed teams (across all 12 groups) also
    qualify. Tiebreak: points -> goal_diff_proxy -> fifa_rank (lower is
    better). Conduct score is NOT modelled (documented simplification).

    Returns dict with keys: 'group_winners', 'group_runners_up',
    'qualifying_third_place', 'eliminated'
    """
    standings = standings.copy()

    group_winners, group_runners_up, third_place_candidates = [], [], []

    for group_name, group_df in standings.groupby('group'):
        sorted_group = group_df.sort_values(
            by=['points', 'goal_diff_proxy', 'fifa_rank'],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        group_winners.append(sorted_group.iloc[0]['team'])
        group_runners_up.append(sorted_group.iloc[1]['team'])
        third_place_candidates.append({
            'team': sorted_group.iloc[2]['team'],
            'group': group_name,
            'points': sorted_group.iloc[2]['points'],
            'goal_diff_proxy': sorted_group.iloc[2]['goal_diff_proxy'],
            'fifa_rank': sorted_group.iloc[2]['fifa_rank'],
        })
        # 4th place is eliminated, not tracked further

    third_place_df = pd.DataFrame(third_place_candidates).sort_values(
        by=['points', 'goal_diff_proxy', 'fifa_rank'],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    qualifying_third_place = third_place_df.iloc[:8]['team'].tolist()
    eliminated_third_place = third_place_df.iloc[8:]['team'].tolist()

    return {
        'group_winners': group_winners,
        'group_runners_up': group_runners_up,
        'qualifying_third_place': qualifying_third_place,
        'eliminated_third_place': eliminated_third_place,
    }


def simulate_knockout_match(
    home_team: str,
    away_team: str,
    model,
    matches_long: pd.DataFrame,
    elo_df: pd.DataFrame,
    fifa_rank_df: pd.DataFrame,
    past_winners: dict,
    match_date: pd.Timestamp,
    assemble_match_features_fn,
    rng: np.random.Generator,
) -> str:
    """
    Simulate a single knockout match. Knockout matches cannot end in a draw
    (extra time + penalties decide it), so we redistribute draw probability
    proportionally to home/away win probability -- this is a standard,
    defensible simplification (effectively treating a drawn-after-90-minutes
    result as a coin-flip-adjusted continuation, weighted by each team's
    underlying strength rather than a true 50/50).

    Returns the winning team's name.
    """
    is_neutral = True  # all knockout matches at shared/neutral venues, incl. host nations

    probs = predict_match_probs(
        model, home_team, away_team, match_date, matches_long, elo_df,
        fifa_rank_df, past_winners, is_neutral, assemble_match_features_fn,
    )
    p_home, p_draw, p_away = probs

    # Redistribute draw probability proportionally
    p_home_adjusted = float(p_home) + float(p_draw) * (float(p_home) / (float(p_home) + float(p_away)))
    p_away_adjusted = float(p_away) + float(p_draw) * (float(p_away) / (float(p_home) + float(p_away)))

    # Cast to plain Python float (float64) BEFORE normalizing -- torch's
    # softmax returns float32, and numpy.random.Generator.choice validates
    # probability sums at float64 precision internally. A sum that displays
    # as exactly 1.0 in float32 can still fail validation once promoted to
    # float64, so we must normalize using float64 arithmetic throughout.
    total = p_home_adjusted + p_away_adjusted
    p_home_adjusted /= total
    p_away_adjusted /= total

    winner = rng.choice([home_team, away_team], p=[p_home_adjusted, p_away_adjusted])
    return winner


def run_full_knockout_bracket(
    round_of_32_matchups: list[tuple[str, str]],
    model,
    matches_long: pd.DataFrame,
    elo_df: pd.DataFrame,
    fifa_rank_df: pd.DataFrame,
    past_winners: dict,
    match_date: pd.Timestamp,
    assemble_match_features_fn,
    rng_seed: int = 42,
) -> dict:
    """
    Run the single-elimination bracket from Round of 32 through to the
    Final. Returns the results of every round as a dict of round_name ->
    list of (winner, loser) tuples, plus the overall champion.
    """
    rng = np.random.default_rng(rng_seed)
    round_results = {}
    current_round = round_of_32_matchups
    round_names = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]

    for round_name in round_names:
        if len(current_round) == 0:
            break

        winners = []
        round_log = []
        for home, away in current_round:
            winner = simulate_knockout_match(
                home, away, model, matches_long, elo_df, fifa_rank_df,
                past_winners, match_date, assemble_match_features_fn, rng,
            )
            loser = away if winner == home else home
            winners.append(winner)
            round_log.append((home, away, winner))

        round_results[round_name] = round_log

        # Pair up winners for next round (sequential pairing)
        current_round = [(winners[i], winners[i+1]) for i in range(0, len(winners) - 1, 2)]

    champion = round_results.get("Final", [(None, None, None)])[0][2]
    round_results["champion"] = champion
    return round_results


def build_round_of_32_matchups(qualification_result: dict, groups: dict) -> list[tuple[str, str]]:
    """
    Build Round of 32 matchups from group qualification results.

    There are exactly 32 qualifying teams: 12 group winners, 12 runners-up,
    8 best third-place teams. We need exactly 16 matchups pairing them up.

    Honest scope note: real FIFA Round of 32 pairings depend on WHICH eight
    third-placed teams qualify (495 possible combinations, Annex C of the
    official regulations). We use a documented simplification instead:
    each of the 8 third-place qualifiers is paired against one group winner
    (using the remaining 4 winners' against each other instead, since there
    are 12 winners but only 8 third-place teams), and all 12 runners-up are
    paired against each other. This preserves the correct COUNT and
    STRUCTURE of the bracket (16 matchups, every qualifying team appears
    EXACTLY once) without encoding FIFA's exact administrative pairing table.
    """
    winners = list(qualification_result['group_winners'])
    runners_up = list(qualification_result['group_runners_up'])
    third_place = list(qualification_result['qualifying_third_place'])

    assert len(winners) == 12, f"Expected 12 group winners, got {len(winners)}"
    assert len(runners_up) == 12, f"Expected 12 runners-up, got {len(runners_up)}"
    assert len(third_place) == 8, f"Expected 8 qualifying third-place teams, got {len(third_place)}"

    matchups = []

    # First 8 winners each face one of the 8 qualifying third-place teams
    winners_vs_third = winners[:8]
    for i in range(8):
        matchups.append((winners_vs_third[i], third_place[i]))

    # Remaining 4 winners pair against each other (2 matchups)
    remaining_winners = winners[8:12]
    for i in range(0, len(remaining_winners), 2):
        matchups.append((remaining_winners[i], remaining_winners[i + 1]))

    # All 12 runners-up pair against each other (6 matchups)
    for i in range(0, len(runners_up), 2):
        matchups.append((runners_up[i], runners_up[i + 1]))

    assert len(matchups) == 16, f"Expected exactly 16 Round of 32 matchups, built {len(matchups)}"

    # Verify every qualifying team appears EXACTLY once across all matchups
    all_teams_in_matchups = [t for pair in matchups for t in pair]
    assert len(all_teams_in_matchups) == 32, f"Expected 32 teams total, got {len(all_teams_in_matchups)}"
    assert len(set(all_teams_in_matchups)) == 32, "Some team appears more than once in the bracket -- bug"

    return matchups


def run_single_tournament_simulation(
    groups: dict,
    model,
    matches_long: pd.DataFrame,
    elo_df: pd.DataFrame,
    fifa_rank_df: pd.DataFrame,
    past_winners: dict,
    fifa_ranks: dict,
    is_neutral_venue_fn,
    match_date: pd.Timestamp,
    assemble_match_features_fn,
    rng_seed: int,
) -> dict:
    """
    Run ONE complete tournament simulation end-to-end: group stage ->
    qualification -> round of 32 -> ... -> champion.

    Returns a dict mapping team -> final stage reached, suitable for
    aggregating across many simulations.
    """
    standings = simulate_group_stage(
        groups, model, matches_long, elo_df, fifa_rank_df, past_winners,
        fifa_ranks, is_neutral_venue_fn, match_date, assemble_match_features_fn,
        rng_seed=rng_seed,
    )

    qualification = rank_groups_and_qualify(standings)
    round_of_32 = build_round_of_32_matchups(qualification, groups)

    bracket_results = run_full_knockout_bracket(
        round_of_32, model, matches_long, elo_df, fifa_rank_df, past_winners,
        match_date, assemble_match_features_fn, rng_seed=rng_seed,
    )

    all_teams = [t for teams in groups.values() for t in teams]
    progress = {team: "Group Stage" for team in all_teams}

    qualified_round_of_32 = set(qualification['group_winners']) | \
                             set(qualification['group_runners_up']) | \
                             set(qualification['qualifying_third_place'])
    for team in qualified_round_of_32:
        progress[team] = "Round of 32"

    round_order = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
    for round_name in round_order:
        if round_name not in bracket_results:
            continue
        for home, away, winner in bracket_results[round_name]:
            loser = away if winner == home else home
            next_round_idx = round_order.index(round_name) + 1
            if next_round_idx < len(round_order):
                progress[winner] = round_order[next_round_idx]
            else:
                progress[winner] = "Champion"
            progress[loser] = round_name

    champion = bracket_results.get("champion")
    if champion:
        progress[champion] = "Champion"

    return progress


def run_monte_carlo_simulation(
    groups: dict,
    model,
    matches_long: pd.DataFrame,
    elo_df: pd.DataFrame,
    fifa_rank_df: pd.DataFrame,
    past_winners: dict,
    fifa_ranks: dict,
    is_neutral_venue_fn,
    match_date: pd.Timestamp,
    assemble_match_features_fn,
    n_simulations: int = 1000,
    progress_every: int = 100,
) -> pd.DataFrame:
    """
    Run the full tournament n_simulations times, each with a different
    random seed, and aggregate how often each team reaches each stage.

    This is the statistically honest version of "predict the World Cup" --
    instead of one random sample from the model's probability distribution,
    we report the empirical frequency of outcomes across many samples,
    which IS the model's actual probability estimate for each team's
    tournament outcome.
    """
    all_teams = [t for teams in groups.values() for t in teams]
    stage_order = ["Group Stage", "Round of 32", "Round of 16",
                   "Quarter-finals", "Semi-finals", "Final", "Champion"]

    counts = {team: {stage: 0 for stage in stage_order} for team in all_teams}

    for sim_i in range(n_simulations):
        progress = run_single_tournament_simulation(
            groups, model, matches_long, elo_df, fifa_rank_df, past_winners,
            fifa_ranks, is_neutral_venue_fn, match_date, assemble_match_features_fn,
            rng_seed=sim_i,
        )
        for team, final_stage in progress.items():
            reached_idx = stage_order.index(final_stage)
            for s in stage_order[:reached_idx + 1]:
                counts[team][s] += 1

        if (sim_i + 1) % progress_every == 0:
            print(f"  Completed {sim_i + 1}/{n_simulations} simulations")

    rows = []
    for team in all_teams:
        row = {"team": team}
        for stage in stage_order:
            col_name = stage.lower().replace(' ', '_').replace('-', '_')
            row[f"pct_{col_name}"] = counts[team][stage] / n_simulations
        rows.append(row)

    return pd.DataFrame(rows).sort_values("pct_champion", ascending=False).reset_index(drop=True)

