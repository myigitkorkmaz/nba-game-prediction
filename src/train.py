"""Step 3 - Train, tune and test the models.

Split by time: train on the 2010-11 to 2018-19 seasons, test on 2019-20 to
2023-24. Hyperparameters are chosen with forward-chaining validation inside the
training seasons only (train on seasons <= k, validate on season k+1).

Input : data/processed/features.csv
Output: results/*.csv, results/metrics.json, results/figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = RES / "figures" 
SEED = 42 #random seed for every run

#divided the play by play data in half here, half for testing and the other for training
TRAIN_SEASONS = range(2011, 2020)    # 2010-11 .. 2018-19
TEST_SEASONS = range(2020, 2025)     # 2019-20 .. 2023-24
VALID_SEASONS = (2017, 2018, 2019)   # forward-chaining folds inside the training seasons

#data features used for model training and prediction
FEATURES = [
    "DIFF_ELO", "DIFF_SEASON_WIN_PCT", "DIFF_SEASON_NET_RTG",
    "DIFF_L10_WIN_PCT", "DIFF_L10_MARGIN", "DIFF_L10_ORTG", "DIFF_L10_DRTG",
    "DIFF_L10_NET_RTG", "DIFF_L10_PACE", "DIFF_L10_EFG", "DIFF_L10_TOV_PCT",
    "DIFF_L10_OREB_PCT", "DIFF_L10_FT_RATE", "DIFF_L10_OPP_EFG", "DIFF_L10_DREB_PCT",
    "DIFF_REST_DAYS", "DIFF_GAMES_LAST_7D", "HOME_BACK_TO_BACK", "AWAY_BACK_TO_BACK",
]
TARGET = "HOME_WIN" #prediction, 1 = home team won

# Logistic regression: C = regularization strength (smaller C = simpler model)
LR_GRID = [{"C": c} for c in (0.001, 0.01, 0.1, 1.0, 10.0)]

# Random forest: tree depth x min games per leaf x share of features per split = 18 combos
RF_GRID = [{"max_depth": d, "min_samples_leaf": l, "max_features": f}
           for d in (4, 6, 10) for l in (20, 50, 100) for f in ("sqrt", 0.5)]


def make_lr(**p):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **p))


def make_rf(**p):
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, **p)


def forward_chain_score(make, params, train):
    """Mean validation log loss over the forward-chaining folds."""
    losses = []
    for v in VALID_SEASONS:
        fit, val = train[train.SEASON < v], train[train.SEASON == v]
        m = make(**params).fit(fit[FEATURES], fit[TARGET])
        losses.append(log_loss(val[TARGET], m.predict_proba(val[FEATURES])[:, 1]))
    return float(np.mean(losses))

"""Try every setting in the grid and return the one with the lowest score."""
def tune(name, make, grid, train):
    scores = [(forward_chain_score(make, p, train), p) for p in grid]
    rows = [{"model": name, **p, "val_log_loss": round(s, 4)} for s, p in scores]
    best = min(scores, key=lambda t: t[0])[1]
    return best, rows

 """This one compute the test metrics from true results (y) and predicted home-win chances (prob)."""
def evaluate(y, prob):
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    half = 1.96 * np.sqrt(acc * (1 - acc) / len(y))
    out = {"accuracy": acc, "acc_ci_low": acc - half, "acc_ci_high": acc + half, "n": len(y)}
    if len(np.unique(prob)) > 2:      # probabilistic models only
        out.update(log_loss=log_loss(y, prob), brier=brier_score_loss(y, prob),
                   auc=roc_auc_score(y, prob))
    return out


def main() -> None:
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    #load and parse by season
    allg = pd.read_csv(PROC / "features.csv", parse_dates=["GAME_DATE"])
    df = allg[allg.ENOUGH_HISTORY == 1].copy()
    train = df[df.SEASON.isin(TRAIN_SEASONS)]
    test = df[df.SEASON.isin(TEST_SEASONS)].copy()
    print(f"train games: {len(train):,}  test games: {len(test):,}  features: {len(FEATURES)}")

    # ---- tuning -----------------------------------------------------------
    lr_best, lr_rows = tune("logistic_regression", make_lr, LR_GRID, train)
    rf_best, rf_rows = tune("random_forest", make_rf, RF_GRID, train)
    pd.DataFrame(lr_rows + rf_rows).to_csv(RES / "tuning.csv", index=False)
    print("best LR:", lr_best, " best RF:", rf_best)

    lr = make_lr(**lr_best).fit(train[FEATURES], train[TARGET])
    rf = make_rf(**rf_best).fit(train[FEATURES], train[TARGET])
    test["P_LR"] = lr.predict_proba(test[FEATURES])[:, 1]
    test["P_RF"] = rf.predict_proba(test[FEATURES])[:, 1]

    # ---- baselines ---------------------------------------------------------
    test["P_HOME"] = 1.0
    better = np.sign(test.DIFF_SEASON_WIN_PCT)
    test["P_RECORD"] = np.where(better == 0, 1.0, (better > 0).astype(float))
    test["P_ELO"] = 1 / (1 + 10 ** (-(test.DIFF_ELO + np.where(test.BUBBLE == 1, 0, 100)) / 400))

    models = {
        "Baseline: home team always wins": "P_HOME",
        "Baseline: better record wins": "P_RECORD",
        "Baseline: Elo favorite wins": "P_ELO",
        "Logistic regression": "P_LR",
        "Random forest": "P_RF",
    }
    y = test[TARGET].values
    metrics = {}
    for name, col in models.items():
        p = test[col].values
        m = evaluate(y, p)
        if col in ("P_HOME", "P_RECORD"):
            m = {k: v for k, v in m.items() if k in ("accuracy", "acc_ci_low", "acc_ci_high", "n")}
        metrics[name] = m
    summary = pd.DataFrame(metrics).T
    summary.to_csv(RES / "test_metrics.csv")
    print(summary.round(4))

    # train accuracy, to check for over-fitting
    fit_check = {
        "Logistic regression": accuracy_score(train[TARGET], lr.predict(train[FEATURES])),
        "Random forest": accuracy_score(train[TARGET], rf.predict(train[FEATURES])),
    }

    # ---- per-season results and home court ----------------------------------
    test["SEASON_GROUP"] = np.where(test.BUBBLE == 1, "2019-20 bubble", test.SEASON_YEAR)
    per_season = test.groupby("SEASON_GROUP").apply(lambda g: pd.Series({
        "games": len(g),
        "home_win_rate": g[TARGET].mean(),
        "acc_home_always": g[TARGET].mean(),
        "acc_elo": ((g.P_ELO >= .5) == g[TARGET]).mean(),
        "acc_lr": ((g.P_LR >= .5) == g[TARGET]).mean(),
        "acc_rf": ((g.P_RF >= .5) == g[TARGET]).mean(),
        "mean_pred_home_prob_lr": g.P_LR.mean(),
    }), include_groups=False)
    per_season.to_csv(RES / "per_season_test.csv")
    print(per_season.round(3))

    #home court advantage
    allg["GROUP"] = np.where(allg.BUBBLE == 1, "2019-20 bubble", allg.SEASON_YEAR)
    home_court = allg.groupby("GROUP").agg(games=(TARGET, "size"), home_win_rate=(TARGET, "mean"),
                                           avg_home_margin=("HOME_PTS", "mean"))
    home_court["avg_home_margin"] = (allg.groupby("GROUP")["HOME_PTS"].mean()
                                     - allg.groupby("GROUP")["AWAY_PTS"].mean())
    home_court.to_csv(RES / "home_court_by_season.csv")
    print(home_court.round(3))

    # ---- feature importance --------------------------------------------------
    coefs = pd.Series(lr[-1].coef_[0], index=FEATURES)
    perm = permutation_importance(rf, test[FEATURES], y, scoring="neg_log_loss",
                                  n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"lr_std_coef": coefs,
                        "rf_perm_importance": perm.importances_mean,
                        "rf_perm_std": perm.importances_std}).sort_values("rf_perm_importance",
                                                                          ascending=False)
    imp.to_csv(RES / "feature_importance.csv")
    print(imp.round(4))

    # small ablation: logistic regression on Elo alone vs. all features
    abl = {}
    for name, cols in {"Elo difference only": ["DIFF_ELO"],
                       "Box-score form only (no Elo, no record)": [f for f in FEATURES if
                            f not in ("DIFF_ELO", "DIFF_SEASON_WIN_PCT", "DIFF_SEASON_NET_RTG")],
                       "All 19 features": FEATURES}.items():
        m = make_lr(**lr_best).fit(train[cols], train[TARGET])
        p = m.predict_proba(test[cols])[:, 1]
        abl[name] = evaluate(y, p)
    pd.DataFrame(abl).T.to_csv(RES / "ablation.csv")
    print(pd.DataFrame(abl).T.round(4))

    json.dump({"train_games": len(train), "test_games": len(test), "features": FEATURES,
               "best_lr": lr_best, "best_rf": rf_best, "train_accuracy": fit_check,
               "test": metrics, "ablation": abl}, open(RES / "metrics.json", "w"), indent=2,
              default=float)
    test[["GAME_ID", "GAME_DATE", "SEASON_YEAR", "HOME_TEAM_ABBREVIATION",
          "AWAY_TEAM_ABBREVIATION", TARGET, "BUBBLE", "P_ELO", "P_LR", "P_RF"]].to_csv(
        RES / "test_predictions.csv", index=False)

    # ---- figures ---------------------------------------------------------------
    order = list(models)[::-1]
    accs = [metrics[k]["accuracy"] for k in order]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(order, accs, color=["#2b6cb0" if "Base" not in k else "#a0aec0" for k in order])
    for i, a in enumerate(accs):
        ax.text(a + .003, i, f"{a:.1%}", va="center")
    ax.set_xlim(.5, .72)
    ax.set_xlabel(f"Test accuracy, 2019-20 to 2023-24 ({len(test):,} games)")
    ax.set_title("Both models beat 'home team always wins'")
    fig.tight_layout(); fig.savefig(FIG / "test_accuracy.png", dpi=150); plt.close(fig)

    hc = home_court.reset_index()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(hc.GROUP, hc.home_win_rate, color=["#c53030" if "bubble" in g else "#4a5568" for g in hc.GROUP])
    ax.axhline(.5, color="k", lw=.8, ls="--")
    ax.set_ylim(.4, .65); ax.set_ylabel("Home win rate")
    ax.set_title("Home-court advantage by season (bubble games in red)")
    plt.xticks(rotation=60, ha="right")
    fig.tight_layout(); fig.savefig(FIG / "home_court.png", dpi=150); plt.close(fig)

    top = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top.index, top.rf_perm_importance, xerr=top.rf_perm_std, color="#2b6cb0")
    ax.set_xlabel("Increase in test log loss when the feature is shuffled")
    ax.set_title("Random forest: most useful features")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png", dpi=150); plt.close(fig)

    # calibration
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    bins = np.linspace(0, 1, 11)
    for col, lab in (("P_LR", "Logistic regression"), ("P_RF", "Random forest")):
        b = pd.cut(test[col], bins)
        c = test.groupby(b, observed=True).agg(p=(col, "mean"), y=(TARGET, "mean"))
        ax.plot(c.p, c.y, marker="o", label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_xlabel("Predicted home win probability"); ax.set_ylabel("Actual home win rate")
    ax.legend(); ax.set_title("Calibration on test seasons")
    fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
