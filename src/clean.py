"""Step 1 - Data cleaning and extraction.

Reads the raw team-game box scores (one row per team per game) and produces
one clean row per game with the home team's and away team's box score side by
side.

Input : data/raw/regular_season_totals_2010_2024.csv
Output: data/processed/games.csv, data/processed/cleaning_report.txt
"""
from pathlib import Path

import pandas as pd

#managing files 
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "regular_season_totals_2010_2024.csv"
OUT_DIR = ROOT / "data" / "processed"

# Box-score columns we keep (the *_RANK columns are league ranks of the raw
# numbers and add nothing for modelling, so they are dropped).
BOX_COLS = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB",
            "REB", "AST", "TOV", "STL", "BLK", "PF", "PTS"]

# The 2020 "bubble": after the COVID stoppage the 2019-20 season restarted at
# Walt Disney World on 30 July 2020 with no fans and every game at one site.
BUBBLE_START = pd.Timestamp("2020-07-30")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = []

    raw = pd.read_csv(RAW, dtype={"GAME_ID": str})
    log.append(f"raw rows (team-games): {len(raw):,}")
    log.append(f"raw columns: {raw.shape[1]}")

    df = raw[["SEASON_YEAR", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID",
              "GAME_DATE", "MATCHUP", "WL", "MIN"] + BOX_COLS].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()

    # --- quality checks -------------------------------------------------
    dups = df.duplicated(["GAME_ID", "TEAM_ID"]).sum()
    log.append(f"duplicate team-game rows removed: {dups}")
    df = df.drop_duplicates(["GAME_ID", "TEAM_ID"])

    missing = df[["WL", "PTS", "FGA", "FTA", "TOV", "OREB"]].isna().sum().sum()
    log.append(f"missing values in key columns: {missing}")
    df = df.dropna(subset=["WL", "PTS", "FGA", "FTA", "TOV", "OREB"])

    counts = df.groupby("GAME_ID")["TEAM_ID"].transform("size")
    log.append(f"team-games whose opponent row is missing (dropped): {(counts != 2).sum()}")
    df = df[counts == 2]

    # home team = the row whose matchup reads "XXX vs. YYY"
    df["IS_HOME"] = df["MATCHUP"].str.contains(" vs. ").astype(int)
    bad_home = (df.groupby("GAME_ID")["IS_HOME"].sum() != 1).sum()
    log.append(f"games without exactly one home team: {bad_home}")

    # --- pair home and away rows into one row per game ---------------------
    home = df[df["IS_HOME"] == 1].set_index("GAME_ID")
    away = df[df["IS_HOME"] == 0].set_index("GAME_ID")
    keep = ["TEAM_ID", "TEAM_ABBREVIATION", "MIN"] + BOX_COLS
    games = home[["SEASON_YEAR", "GAME_DATE"]].join(
        home[keep].add_prefix("HOME_")).join(away[keep].add_prefix("AWAY_"), how="inner")
    games = games.reset_index()

    games["HOME_WIN"] = (games["HOME_PTS"] > games["AWAY_PTS"]).astype(int)
    ties = (games["HOME_PTS"] == games["AWAY_PTS"]).sum()
    log.append(f"tied final scores (should be 0): {ties}")
    games["OVERTIME"] = (games["HOME_MIN"] > 48.5).astype(int)
    games["BUBBLE"] = ((games["SEASON_YEAR"] == "2019-20") &
                       (games["GAME_DATE"] >= BUBBLE_START)).astype(int)
    games["SEASON"] = games["SEASON_YEAR"].str[:4].astype(int) + 1   # 2019-20 -> 2020

    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    games.to_csv(OUT_DIR / "games.csv", index=False)

    log.append(f"clean games: {len(games):,}")
    log.append(f"seasons: {games['SEASON_YEAR'].min()} to {games['SEASON_YEAR'].max()}")
    log.append("games per season:\n" + games.groupby("SEASON_YEAR").size().to_string())
    log.append(f"bubble games: {games['BUBBLE'].sum()}")
    log.append(f"overall home win rate: {games['HOME_WIN'].mean():.3f}")

    text = "\n".join(log)
    (OUT_DIR / "cleaning_report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
