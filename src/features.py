"""Step 2 - Pre-game feature extraction.

Every feature for a game is computed only from games that finished BEFORE
that game's date (each team's history is shifted by one game), so the model
never sees the result it is asked to predict.

Input : data/processed/games.csv
Output: data/processed/features.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

WINDOW = 10            # "recent form" window, in games
MIN_GAMES = 10         # both teams must have played this many games this season

# Elo settings (FiveThirtyEight-style)
ELO_START, ELO_MEAN, ELO_K, ELO_HOME, ELO_CARRY = 1500.0, 1505.0, 20.0, 100.0, 0.75


def team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with the team's and its opponent's stats."""
    sides = []
    for me, opp, home in (("HOME", "AWAY", 1), ("AWAY", "HOME", 0)):
        t = pd.DataFrame({
            "GAME_ID": games["GAME_ID"], "GAME_DATE": games["GAME_DATE"],
            "SEASON": games["SEASON"], "TEAM_ID": games[f"{me}_TEAM_ID"],
            "IS_HOME": home,
        })
        for c in ["PTS", "FGM", "FGA", "FG3M", "FTA", "OREB", "DREB", "TOV", "MIN"]:
            t[c] = games[f"{me}_{c}"]
            t["OPP_" + c] = games[f"{opp}_{c}"]
        sides.append(t)
    log = pd.concat(sides, ignore_index=True)

    # possessions (standard estimate), averaged over both teams
    poss = lambda p: log[p + "FGA"] - log[p + "OREB"] + log[p + "TOV"] + 0.44 * log[p + "FTA"]
    log["POSS"] = (poss("") + poss("OPP_")) / 2
    log["WIN"] = (log["PTS"] > log["OPP_PTS"]).astype(int)
    log["MARGIN"] = log["PTS"] - log["OPP_PTS"]
    log["ORTG"] = 100 * log["PTS"] / log["POSS"]
    log["DRTG"] = 100 * log["OPP_PTS"] / log["POSS"]
    log["NET_RTG"] = log["ORTG"] - log["DRTG"]
    log["PACE"] = 48 * log["POSS"] / log["MIN"]   # MIN = game length in minutes
    """ Dean Oliver's four factors (offense) 
        Dean Oliver identified what he called the "Four Factors of Basketball Success":
        Shooting (40%) Turnovers (25%)   Rebounding (20%)  Free Throws (15%)
        The number in parentheses is the approximate weight Mr. Oliver assigned each factor. 
        Shooting is the most important factor, followed by turnovers, rebounding, and free throws.  """
    log["EFG"] = (log["FGM"] + 0.5 * log["FG3M"]) / log["FGA"]
    log["TOV_PCT"] = log["TOV"] / log["POSS"]
    log["OREB_PCT"] = log["OREB"] / (log["OREB"] + log["OPP_DREB"])
    log["FT_RATE"] = log["FTA"] / log["FGA"]
    # defensive side: what the team allows
    log["OPP_EFG"] = (log["OPP_FGM"] + 0.5 * log["OPP_FG3M"]) / log["OPP_FGA"]
    log["DREB_PCT"] = log["DREB"] / (log["DREB"] + log["OPP_OREB"])
    return log.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def rolling_pregame(log: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date and last-N averages, shifted so they use past games only."""
    g = log.groupby(["TEAM_ID", "SEASON"], sort=False)
    out = log[["GAME_ID", "TEAM_ID", "IS_HOME"]].copy()
    out["GAMES_PLAYED"] = g.cumcount()                       # games before this one
    out["SEASON_WIN_PCT"] = g["WIN"].transform(lambda s: s.shift().expanding().mean())
    out["SEASON_NET_RTG"] = g["NET_RTG"].transform(lambda s: s.shift().expanding().mean())

    recent = ["WIN", "MARGIN", "ORTG", "DRTG", "NET_RTG", "PACE",
              "EFG", "TOV_PCT", "OREB_PCT", "FT_RATE", "OPP_EFG", "DREB_PCT"]
    for c in recent:
        name = "L10_WIN_PCT" if c == "WIN" else f"L10_{c}"
        out[name] = g[c].transform(lambda s: s.shift().rolling(WINDOW, min_periods=3).mean())

    # rest: days since this team's previous game (season opener -> 7, i.e. "rested")
    prev = g["GAME_DATE"].shift()
    out["REST_DAYS"] = (log["GAME_DATE"] - prev).dt.days.fillna(7).clip(upper=7)
    out["BACK_TO_BACK"] = (out["REST_DAYS"] == 1).astype(int)
    # games in the last 7 days (schedule density), not counting today
    def games_last_7(dates: pd.Series) -> pd.Series:
        d = dates.values.astype("datetime64[D]").astype(np.int64)
        return pd.Series([((d[:i] >= d[i] - 7) & (d[:i] < d[i])).sum() for i in range(len(d))],
                         index=dates.index)
    out["GAMES_LAST_7D"] = g["GAME_DATE"].transform(games_last_7)
    return out

#   Idea: every team has a strength number. 
def elo_ratings(games: pd.DataFrame) -> pd.DataFrame:
    """Pre-game Elo for both teams, updated game by game in date order."""
    elo, season_of = {}, {}
    pre_home, pre_away = [], []
    for row in games.itertuples(index=False):
        for t in (row.HOME_TEAM_ID, row.AWAY_TEAM_ID):
            if t not in elo:
                elo[t], season_of[t] = ELO_START, row.SEASON
            elif season_of[t] != row.SEASON:               # new season: regress to mean
                elo[t] = ELO_CARRY * elo[t] + (1 - ELO_CARRY) * ELO_MEAN
                season_of[t] = row.SEASON
        h, a = elo[row.HOME_TEAM_ID], elo[row.AWAY_TEAM_ID]
        pre_home.append(h)
        pre_away.append(a)
        hfa = 0.0 if row.BUBBLE else ELO_HOME
        exp_home = 1 / (1 + 10 ** (-(h + hfa - a) / 400))
        margin = row.HOME_PTS - row.AWAY_PTS
        win_diff = (h + hfa - a) if margin > 0 else (a - h - hfa)
        mult = (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * win_diff)
        shift = ELO_K * mult * ((margin > 0) - exp_home)
        elo[row.HOME_TEAM_ID] = h + shift
        elo[row.AWAY_TEAM_ID] = a - shift
    return pd.DataFrame({"GAME_ID": games["GAME_ID"], "HOME_ELO": pre_home, "AWAY_ELO": pre_away})


def main() -> None:
    games = pd.read_csv(PROC / "games.csv", dtype={"GAME_ID": str}, parse_dates=["GAME_DATE"])
    log = team_game_log(games)
    pre = rolling_pregame(log)

    feat_cols = [c for c in pre.columns if c not in ("GAME_ID", "TEAM_ID", "IS_HOME")]
    home = pre[pre["IS_HOME"] == 1].set_index("GAME_ID")[feat_cols].add_prefix("HOME_")
    away = pre[pre["IS_HOME"] == 0].set_index("GAME_ID")[feat_cols].add_prefix("AWAY_")

    base = games[["GAME_ID", "GAME_DATE", "SEASON", "SEASON_YEAR", "HOME_TEAM_ABBREVIATION",
                  "AWAY_TEAM_ABBREVIATION", "HOME_PTS", "AWAY_PTS", "HOME_WIN", "BUBBLE"]]
    df = base.join(home, on="GAME_ID").join(away, on="GAME_ID")
    df = df.merge(elo_ratings(games), on="GAME_ID")

    # model inputs = home-minus-away differences (plus each side's rest)
    diff = ["SEASON_WIN_PCT", "SEASON_NET_RTG", "L10_WIN_PCT", "L10_MARGIN", "L10_ORTG",
            "L10_DRTG", "L10_NET_RTG", "L10_PACE", "L10_EFG", "L10_TOV_PCT", "L10_OREB_PCT",
            "L10_FT_RATE", "L10_OPP_EFG", "L10_DREB_PCT", "REST_DAYS", "GAMES_LAST_7D"]
    for c in diff:
        df[f"DIFF_{c}"] = df[f"HOME_{c}"] - df[f"AWAY_{c}"]
    df["DIFF_ELO"] = df["HOME_ELO"] - df["AWAY_ELO"]

    before = len(df)
    df["ENOUGH_HISTORY"] = ((df["HOME_GAMES_PLAYED"] >= MIN_GAMES) &
                            (df["AWAY_GAMES_PLAYED"] >= MIN_GAMES)).astype(int)
    df.to_csv(PROC / "features.csv", index=False)
    print(f"games: {before:,}; with >= {MIN_GAMES} games of history for both teams: "
          f"{df['ENOUGH_HISTORY'].sum():,}")


if __name__ == "__main__":
    main()
