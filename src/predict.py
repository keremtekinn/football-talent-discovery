"""Train final models on matured FIFA 15-22 cohorts and score FC 26 youth prospects."""

import numpy as np
import pandas as pd

from src.features import compute_player_weights
from src.models import get_xgb_shap, make_ensemble, make_xgb


def fit_and_predict_fc26(df, feature_cols, cfg):
    cohort = cfg["cohort"]
    seed = cfg["evaluation"]["random_state"]
    metrics_dir = cfg["resolved_paths"]["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    youth_mask = (df["age"] <= cohort["max_scout_age"]) & (df["overall"] <= cohort["max_scout_overall"])
    fc26_df = df[youth_mask & (df["fifa_version"] == cohort["predict_version"])].copy().reset_index(drop=True)
    fc26_youth_pids = set(
        df.loc[(df["fifa_version"] == cohort["predict_version"]) & (df["age"] <= cohort["max_scout_age"]), "player_id"]
    )

    # FIFA 15-22 rows all have a full 4-year observation window inside the dataset (up to FC 26)
    train_df = df[
        youth_mask
        & (df["fifa_version"] >= cohort["min_version"])
        & (df["fifa_version"] <= cohort["final_train_max_version"])
        & (~df["player_id"].isin(fc26_youth_pids))
    ].copy().reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["target_elite"].to_numpy(dtype=int)
    w_train = compute_player_weights(train_df)
    X_fc26 = fc26_df[feature_cols].to_numpy(dtype=float)

    xgb_model = make_xgb(random_state=seed)
    xgb_model.fit(X_train, y_train, sample_weight=w_train)

    ensemble_model = make_ensemble(random_state=seed)
    ensemble_model.fit(X_train, y_train, sample_weight=w_train)

    xgb_probs = xgb_model.predict_proba(X_fc26)[:, 1]
    ens_probs = ensemble_model.predict_proba(X_fc26)[:, 1]

    shap_exp = get_xgb_shap(xgb_model, fc26_df, feature_cols)

    top_drivers = []
    for i in range(len(fc26_df)):
        sv = shap_exp.values[i]
        top_idx = np.argsort(sv)[::-1][:3]
        top_drivers.append(", ".join(f"{feature_cols[j]} ({sv[j]:+.2f})" for j in top_idx))

    pred_df = pd.DataFrame({
        "player_id": fc26_df["player_id"].astype(int),
        "player_name": fc26_df["canonical_name"],
        "pos_group": fc26_df["pos_group"],
        "primary_position": fc26_df["primary_position"],
        "age": fc26_df["age"].astype(int),
        "current_ovr": fc26_df["overall"].astype(int),
        "club": fc26_df["club_name"],
        "league": fc26_df["league_name"],
        "nationality": fc26_df["nationality_name"],
        "years_observed": fc26_df["years_observed"].astype(int),
        "xgb_prob": np.round(xgb_probs, 4),
        "ensemble_prob": np.round(ens_probs, 4),
        "top_shap_drivers": top_drivers,
    })
    pred_df = pred_df.sort_values("ensemble_prob", ascending=False).reset_index(drop=True)
    pred_df.insert(0, "overall_rank", np.arange(1, len(pred_df) + 1))

    top10_frames = []
    for pos in ("ATT", "MID", "DEF"):
        sub = pred_df[pred_df["pos_group"] == pos].head(10).copy()
        sub.insert(1, "pos_rank", np.arange(1, len(sub) + 1))
        top10_frames.append(sub)
    top10_pos_df = pd.concat(top10_frames, ignore_index=True)

    pred_df.to_csv(metrics_dir / "fc26_predictions_all.csv", index=False)
    top10_pos_df.to_csv(metrics_dir / "fc26_top10_by_position.csv", index=False)
    return pred_df, top10_pos_df, shap_exp, fc26_df
