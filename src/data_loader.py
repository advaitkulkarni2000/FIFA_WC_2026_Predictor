"""
Data Loader — WC2026 Predictor
--------------------------------
Downloads and merges three Kaggle datasets:
1. Match history (1872-2026)
2. Elo ratings (1901-2026)
3. FIFA official rankings (1992-2024)

Run via kagglehub (no manual API key juggling needed in Colab —
kagglehub handles auth via the Colab secrets UI or kaggle.json).
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_datasets():
    """
    Download all required datasets via kagglehub.
    In Colab, this will prompt for Kaggle credentials on first run
    (kagglehub.login() or upload kaggle.json).
    """
    import kagglehub

    print("Downloading match history...")
    matches_path = kagglehub.dataset_download(
        "martj42/international-football-results-from-1872-to-2017"
    )

    print("Downloading Elo ratings...")
    elo_path = kagglehub.dataset_download(
        "afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings"
    )

    print("Downloading FIFA rankings...")
    fifa_rank_path = kagglehub.dataset_download(
        "cashncarry/fifaworldranking"
    )

    print("Downloading live 2026 results...")
    wc2026_path = kagglehub.dataset_download(
        "mominullptr/fifa-world-cup-2026-dataset"
    )

    return {
        "matches": matches_path,
        "elo": elo_path,
        "fifa_rank": fifa_rank_path,
        "wc2026": wc2026_path,
    }


def load_match_history(matches_path: str) -> pd.DataFrame:
    """
    Load the core match history dataset.
    Real columns confirmed: date, home_team, away_team, home_score,
                             away_score, tournament, city, country, neutral
    """
    csv_path = Path(matches_path) / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected results.csv in {matches_path}, found: "
            f"{[c.name for c in Path(matches_path).glob('*.csv')]}"
        )

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    df["outcome"] = np.where(
        df["home_score"] > df["away_score"], "home_win",
        np.where(df["home_score"] < df["away_score"], "away_win", "draw")
    )

    return df


def load_elo_ratings(elo_path: str) -> pd.DataFrame:
    """
    Load historical Elo ratings for WC2026-qualified teams.
    Real columns confirmed: year, snapshot_date, country, rank, country_code,
                             rating, rank_max, rating_max, ... confederation, is_host
    One row per (country, year) snapshot.

    Renames 'country' -> 'team' so downstream code (features.py) has one
    consistent name across all datasets.
    """
    csv_path = Path(elo_path) / "elo_ratings_wc2026.csv"
    if not csv_path.exists():
        csvs = list(Path(elo_path).glob("*.csv"))
        csv_path = csvs[0]

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"country": "team"})
    return df


def load_fifa_rankings(rank_path: str) -> pd.DataFrame:
    """
    Load official FIFA rankings, 1992-2024.
    Real columns confirmed: rank, country_full, country_abrv, total_points,
                             previous_points, rank_change, confederation, rank_date

    IMPORTANT: this dataset ships as MULTIPLE separate dated CSVs
    (one per ranking-update date), not a single combined file. We must
    concatenate all of them, not just read the first one found.

    Renames 'country_full' -> 'team' for consistency with other datasets.
    """
    csvs = list(Path(rank_path).glob("fifa_ranking-*.csv"))
    if not csvs:
        csvs = list(Path(rank_path).glob("*.csv"))

    dfs = [pd.read_csv(c) for c in csvs]
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={"country_full": "team"})
    df["rank_date"] = pd.to_datetime(df["rank_date"])
    df = df.drop_duplicates(subset=["team", "rank_date"])
    return df


def load_wc2026_live(wc2026_path: str) -> pd.DataFrame:
    """
    Load live, daily-updated 2026 World Cup match data.

    IMPORTANT: this dataset is FULLY RELATIONAL, not a flat file.
    matches.csv only contains home_team_id / away_team_id (integer foreign
    keys) and stage_id -- NOT team names or stage names. Both must be
    joined in from teams.csv and tournament_stages.csv respectively.

    Real matches.csv columns confirmed: match_id, date, kickoff_time_utc,
        stage_id, venue_id, home_team_id, away_team_id, home_score,
        away_score, status, home_xg, away_xg, referee_id,
        player_of_the_match_id

    Returns a DataFrame with resolved home_team, away_team, and stage_name
    columns, matching the schema used everywhere else in this project.
    """
    matches_path = Path(wc2026_path) / "matches.csv"
    teams_path = Path(wc2026_path) / "teams.csv"
    stages_path = Path(wc2026_path) / "tournament_stages.csv"

    for p, label in [(matches_path, "matches.csv"), (teams_path, "teams.csv"),
                      (stages_path, "tournament_stages.csv")]:
        if not p.exists():
            raise FileNotFoundError(
                f"Expected {label} in {wc2026_path}, found: "
                f"{[c.name for c in Path(wc2026_path).glob('*.csv')]}"
            )

    matches = pd.read_csv(matches_path)
    teams = pd.read_csv(teams_path)
    stages = pd.read_csv(stages_path)

    team_id_col = "team_id" if "team_id" in teams.columns else teams.columns[0]
    team_name_col = "team_name" if "team_name" in teams.columns else (
        [c for c in teams.columns if "name" in c.lower()][0]
    )
    teams_lookup = teams[[team_id_col, team_name_col]].rename(
        columns={team_id_col: "_id", team_name_col: "_name"}
    )

    matches = matches.merge(
        teams_lookup.rename(columns={"_id": "home_team_id", "_name": "home_team"}),
        on="home_team_id", how="left",
    )
    matches = matches.merge(
        teams_lookup.rename(columns={"_id": "away_team_id", "_name": "away_team"}),
        on="away_team_id", how="left",
    )

    # Resolve stage_id -> readable stage_name (e.g. "Group Stage", "Round of 16")
    stage_id_col = "stage_id" if "stage_id" in stages.columns else stages.columns[0]
    stage_name_col = "stage_name" if "stage_name" in stages.columns else (
        [c for c in stages.columns if "name" in c.lower()][0]
    )
    stages_lookup = stages[[stage_id_col, stage_name_col]].rename(
        columns={stage_id_col: "stage_id", stage_name_col: "stage_name"}
    )
    matches = matches.merge(stages_lookup, on="stage_id", how="left")

    matches["date"] = pd.to_datetime(matches["date"])
    return matches


# WC2026 host nations -- only these teams get genuine home advantage, and
# ONLY for their group-stage matches. From kickoff through the knockout
# rounds, ALL matches (including ones host nations might play in) move to
# shared/neutral venues per FIFA's 2026 format.
WC2026_HOST_NATIONS = {"Mexico", "USA", "United States", "Canada"}


def is_host_nation_home_match(home_team: str, stage_name: str) -> bool:
    """
    Returns True if this is a genuine home-advantage match for a 2026
    host nation (i.e. NOT neutral venue). Per FIFA's confirmed format,
    each host nation plays all three of its group-stage matches in its
    own country; everything else (including host nations in the knockout
    rounds) is at a shared/neutral venue.
    """
    is_host = home_team in WC2026_HOST_NATIONS
    is_group_stage = "group" in str(stage_name).lower()
    return is_host and is_group_stage


def build_team_match_long_format(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level rows (one row per match, home+away columns) into
    team-match long format (one row per team per match) — needed to build
    a clean rolling "recent form" sequence per team.

    Each match produces two rows: one from the home team's perspective,
    one from the away team's perspective.
    """
    home_rows = matches_df.copy()
    home_rows["team"] = home_rows["home_team"]
    home_rows["opponent"] = home_rows["away_team"]
    home_rows["goals_for"] = home_rows["home_score"]
    home_rows["goals_against"] = home_rows["away_score"]
    home_rows["is_home"] = 1
    home_rows["result"] = np.where(
        home_rows["goals_for"] > home_rows["goals_against"], "W",
        np.where(home_rows["goals_for"] < home_rows["goals_against"], "L", "D")
    )

    away_rows = matches_df.copy()
    away_rows["team"] = away_rows["away_team"]
    away_rows["opponent"] = away_rows["home_team"]
    away_rows["goals_for"] = away_rows["away_score"]
    away_rows["goals_against"] = away_rows["home_score"]
    away_rows["is_home"] = 0
    away_rows["result"] = np.where(
        away_rows["goals_for"] > away_rows["goals_against"], "W",
        np.where(away_rows["goals_for"] < away_rows["goals_against"], "L", "D")
    )

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)
    return long_df


if __name__ == "__main__":
    paths = download_datasets()
    print("\nDownloaded to:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    matches = load_match_history(paths["matches"])
    print(f"\nMatch history: {len(matches):,} rows, {matches['year'].min()}-{matches['year'].max()}")

    elo = load_elo_ratings(paths["elo"])
    print(f"Elo ratings: {len(elo):,} rows")

    fifa_rank = load_fifa_rankings(paths["fifa_rank"])
    print(f"FIFA rankings: {len(fifa_rank):,} rows")

    wc2026 = load_wc2026_live(paths["wc2026"])
    print(f"WC2026 live data: {len(wc2026):,} rows")
