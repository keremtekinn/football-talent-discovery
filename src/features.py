"""Point-in-time feature engineering, fixed-horizon labeling, and temporal splits."""

import numpy as np
import pandas as pd


def build_features(panel, cfg):
    df = panel.sort_values(["player_id", "fifa_version"]).reset_index(drop=True).copy()
    skill_cols = cfg["features"]["skill_cols"]
    traj_cols = cfg["features"]["trajectory_cols"]
    peer_z_cols = cfg["features"]["peer_z_cols"]

    df["pos_ATT"] = (df["pos_group"] == "ATT").astype(int)
    df["pos_MID"] = (df["pos_group"] == "MID").astype(int)
    df["pos_DEF"] = (df["pos_group"] == "DEF").astype(int)
    df["is_left_foot"] = (df["preferred_foot"] == "Left").astype(int)

    # Leave-one-out club & league strength within the same edition (excludes player's own rating)
    club_sum = df.groupby(["fifa_version", "club_name"])["overall"].transform("sum")
    club_cnt = df.groupby(["fifa_version", "club_name"])["overall"].transform("count")
    df["club_teammate_avg_ovr"] = np.where(club_cnt > 1, (club_sum - df["overall"]) / (club_cnt - 1), np.nan)
    df["club_teammate_avg_ovr"] = df["club_teammate_avg_ovr"].fillna(
        df.groupby("fifa_version")["overall"].transform("median")
    )

    lg_sum = df.groupby(["fifa_version", "league_name"])["overall"].transform("sum")
    lg_cnt = df.groupby(["fifa_version", "league_name"])["overall"].transform("count")
    df["league_peer_avg_ovr"] = np.where(lg_cnt > 1, (lg_sum - df["overall"]) / (lg_cnt - 1), np.nan)
    df["league_peer_avg_ovr"] = df["league_peer_avg_ovr"].fillna(
        df.groupby("fifa_version")["overall"].transform("median")
    )

    # Summary stats from the raw skills only (overall not used)
    skill_matrix = df[skill_cols].to_numpy(dtype=float)
    df["mean_all_skills"] = np.nanmean(skill_matrix, axis=1)
    df["top5_skill_mean"] = np.mean(np.sort(np.nan_to_num(skill_matrix, nan=0.0), axis=1)[:, -5:], axis=1)
    df["skill_std"] = np.nanstd(skill_matrix, axis=1)

    # Peer z-scores within (edition, position, age) cohort
    age_bucket = df["age"].clip(17, 25)
    peer_z_features = []
    for col in peer_z_cols:
        grp = df.groupby(["fifa_version", "pos_group", age_bucket])[col]
        mu = grp.transform("mean")
        sd = grp.transform("std").replace(0, np.nan).fillna(1.0)
        z_col = f"{col}_peer_z"
        df[z_col] = (df[col] - mu) / sd
        peer_z_features.append(z_col)

    # Past-only growth rates and 1-year deltas per player
    g = df.groupby("player_id")
    first_ver = g["fifa_version"].transform("first")
    df["years_observed"] = df["fifa_version"] - first_ver
    df["has_history"] = (df["years_observed"] > 0).astype(int)

    prev_ver = g["fifa_version"].shift(1)
    is_consecutive = (df["fifa_version"] - prev_ver) == 1

    traj_features = ["years_observed", "has_history"]
    for col in traj_cols:
        first_val = g[col].transform("first")
        prev_val = g[col].shift(1)

        rate_col = f"{col}_growth_rate"
        yoy_col = f"{col}_yoy_delta"

        df[rate_col] = np.where(
            df["years_observed"] > 0,
            (df[col] - first_val) / df["years_observed"].replace(0, np.nan),
            0.0,
        )
        df[yoy_col] = np.where(is_consecutive, df[col] - prev_val, 0.0)
        traj_features.extend([rate_col, yoy_col])

    demo_cols = [
        "age",
        "height_cm",
        "weight_kg",
        "weak_foot",
        "skill_moves",
        "is_left_foot",
        "pos_ATT",
        "pos_MID",
        "pos_DEF",
        "mean_all_skills",
        "top5_skill_mean",
        "skill_std",
    ]
    context_cols = ["club_teammate_avg_ovr", "league_peer_avg_ovr"]

    feature_groups = {
        "raw_skills": list(skill_cols),
        "demographics": demo_cols,
        "trajectories": traj_features,
        "peer_z_scores": peer_z_features,
        "club_league_context": context_cols,
        "all": demo_cols + list(skill_cols) + traj_features + peer_z_features + context_cols,
    }

    assert "overall" not in feature_groups["all"]
    return df, feature_groups


def add_horizon_labels(df, horizon_years=4, elite_threshold=82):
    out = df.copy()
    ovr_lookup = out.set_index(["player_id", "fifa_version"])["overall"].to_dict()

    pids = out["player_id"].to_numpy()
    vers = out["fifa_version"].to_numpy()
    max_future = np.full(len(out), np.nan, dtype=float)
    peak_version = np.full(len(out), np.nan, dtype=float)

    for i in range(len(out)):
        pid = pids[i]
        v = vers[i]
        best_ovr = -1.0
        best_v = np.nan
        for step in range(1, horizon_years + 1):
            val = ovr_lookup.get((pid, v + step))
            if val is not None and val > best_ovr:
                best_ovr = float(val)
                best_v = float(v + step)
        if best_ovr >= 0.0:
            max_future[i] = best_ovr
            peak_version[i] = best_v

    out["future_max_ovr"] = max_future
    out["future_peak_version"] = peak_version
    out["has_future_obs"] = out["future_max_ovr"].notna().astype(int)
    out["target_elite"] = (out["future_max_ovr"] >= elite_threshold).astype(int)
    return out


def compute_player_weights(train_df):
    """Inverse appearance frequency weights so repeated player-seasons sum to equal weight per player."""
    counts = train_df.groupby("player_id")["player_id"].transform("count").to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / weights.mean()


def make_temporal_split(df, cutoff_version, cfg):
    """Split youth cohort at cutoff_version with no future labels and no player overlap."""
    cohort = cfg["cohort"]
    youth_mask = (df["age"] <= cohort["max_scout_age"]) & (df["overall"] <= cohort["max_scout_overall"])

    test_df = df[youth_mask & (df["fifa_version"] == cutoff_version)].copy()
    cutoff_youth_pids = set(
        df.loc[(df["fifa_version"] == cutoff_version) & (df["age"] <= cohort["max_scout_age"]), "player_id"]
    )

    # Only include training rows whose forward horizon finishes on or before cutoff_version
    max_train_ver = cutoff_version - cohort["horizon_years"]
    train_df = df[
        youth_mask
        & (df["fifa_version"] >= cohort["min_version"])
        & (df["fifa_version"] <= max_train_ver)
        & (~df["player_id"].isin(cutoff_youth_pids))
    ].copy()

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
