"""Rolling-origin backtesting, model comparison, hard-cohort evaluation, and feature ablation."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.features import compute_player_weights, make_temporal_split
from src.models import get_models, make_xgb


def compute_metrics(y_true, y_prob, pos_groups, current_ovr, train_base_rate, k_list, k_per_pos, is_prob=True):
    base_rate = float(np.mean(y_true))
    total_pos = int(np.sum(y_true))
    order = np.argsort(-y_prob, kind="stable")

    brier = float(brier_score_loss(y_true, y_prob)) if is_prob else np.nan
    brier_ref = float(brier_score_loss(y_true, np.full_like(y_true, train_base_rate, dtype=float)))
    bss = float(1.0 - (brier / max(brier_ref, 1e-9))) if is_prob else np.nan

    # Hard cohort: serious prospects already rated 70-75 OVR
    hard_mask = current_ovr >= 70
    if hard_mask.sum() > 0 and y_true[hard_mask].sum() > 0:
        hard_auc = float(roc_auc_score(y_true[hard_mask], y_prob[hard_mask]))
        hard_pr_auc = float(average_precision_score(y_true[hard_mask], y_prob[hard_mask]))
    else:
        hard_auc = np.nan
        hard_pr_auc = np.nan

    metrics = {
        "n_test": int(len(y_true)),
        "n_positives": total_pos,
        "base_rate": base_rate,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "hard_roc_auc": hard_auc,
        "hard_pr_auc": hard_pr_auc,
        "brier_score": brier,
        "brier_ref": brier_ref,
        "brier_skill_score": bss,
    }

    for k in k_list:
        hits = int(y_true[order[:k]].sum())
        prec = hits / k
        rec = hits / max(1, total_pos)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        lift = prec / max(base_rate, 1e-9)
        metrics[f"hits_at_{k}"] = hits
        metrics[f"precision_at_{k}"] = prec
        metrics[f"recall_at_{k}"] = rec
        metrics[f"f1_at_{k}"] = f1
        metrics[f"lift_at_{k}"] = lift

    pos_hits = 0
    pos_total = 0
    for pos in ("ATT", "MID", "DEF"):
        mask = pos_groups == pos
        if mask.sum() == 0:
            continue
        sub_y = y_true[mask]
        sub_p = y_prob[mask]
        top_n = min(k_per_pos, len(sub_y))
        pos_hits += int(sub_y[np.argsort(-sub_p, kind="stable")[:top_n]].sum())
        pos_total += top_n

    pos_prec = pos_hits / max(1, pos_total)
    metrics["pos_top_k_hits"] = pos_hits
    metrics["pos_top_k_precision"] = pos_prec
    metrics["pos_top_k_lift"] = pos_prec / max(base_rate, 1e-9)
    return metrics


def run_rolling_backtest(df, feature_cols, cfg):
    cutoffs = cfg["cohort"]["rolling_cutoffs"]
    horizon = cfg["cohort"]["horizon_years"]
    k_list = cfg["evaluation"]["top_k_overall"]
    k_per_pos = cfg["evaluation"]["top_k_per_position"]
    seed = cfg["evaluation"]["random_state"]
    metrics_dir = cfg["resolved_paths"]["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    detailed_rows = []
    discovery_rows = []

    for cutoff in cutoffs:
        train_df, test_df = make_temporal_split(df, cutoff, cfg)

        assert set(train_df["player_id"]).isdisjoint(set(test_df["player_id"]))
        assert train_df["fifa_version"].max() + horizon <= cutoff

        X_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["target_elite"].to_numpy(dtype=int)
        w_train = compute_player_weights(train_df)
        train_base_rate = float(np.mean(y_train))

        X_test = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df["target_elite"].to_numpy(dtype=int)
        pos_test = test_df["pos_group"].to_numpy()
        ovr_test = test_df["overall"].to_numpy(dtype=float)
        age_test = test_df["age"].to_numpy(dtype=float)
        attrition_rate = float(1.0 - test_df["has_future_obs"].mean())

        # Domain heuristic baseline: rank by current OVR (tie-breaking by younger age)
        ovr_rank_score = (ovr_test + 0.01 * (25.0 - age_test)) / 100.0
        m_heur = compute_metrics(
            y_test, ovr_rank_score, pos_test, ovr_test, train_base_rate, k_list, k_per_pos, is_prob=False
        )
        m_heur.update({
            "cutoff_year": 2000 + cutoff,
            "eval_horizon_end": 2000 + cutoff + horizon,
            "n_train": int(len(train_df)),
            "n_train_unique_players": int(train_df["player_id"].nunique()),
            "n_train_pos": int(y_train.sum()),
            "test_attrition_rate_4y": round(attrition_rate, 4),
            "model": "Current OVR Heuristic",
        })
        detailed_rows.append(m_heur)

        for model_name, model in get_models(random_state=seed).items():
            model.fit(X_train, y_train, sample_weight=w_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            m = compute_metrics(
                y_test, y_prob, pos_test, ovr_test, train_base_rate, k_list, k_per_pos, is_prob=True
            )
            m.update({
                "cutoff_year": 2000 + cutoff,
                "eval_horizon_end": 2000 + cutoff + horizon,
                "n_train": int(len(train_df)),
                "n_train_unique_players": int(train_df["player_id"].nunique()),
                "n_train_pos": int(y_train.sum()),
                "test_attrition_rate_4y": round(attrition_rate, 4),
                "model": model_name,
            })
            detailed_rows.append(m)

            if model_name in ("Ensemble (XGB+LR+RF)", "XGBoost"):
                order = np.argsort(-y_prob, kind="stable")[:15]
                for rank, idx in enumerate(order, start=1):
                    row = test_df.iloc[idx]
                    discovery_rows.append({
                        "cutoff_year": 2000 + cutoff,
                        "model": model_name,
                        "rank": rank,
                        "player_id": int(row["player_id"]),
                        "player_name": row["canonical_name"],
                        "pos_group": row["pos_group"],
                        "age": int(row["age"]),
                        "club": row["club_name"],
                        "scout_ovr": int(row["overall"]),
                        "future_max_ovr_4y": int(row["future_max_ovr"]) if pd.notna(row["future_max_ovr"]) else None,
                        "hit_82_plus": int(row["target_elite"]),
                        "pred_prob": round(float(y_prob[idx]), 4),
                    })

    detailed_df = pd.DataFrame(detailed_rows)
    discoveries_df = pd.DataFrame(discovery_rows)

    agg_cols = [
        "roc_auc",
        "pr_auc",
        "hard_roc_auc",
        "hard_pr_auc",
        "brier_score",
        "brier_skill_score",
        "precision_at_15",
        "recall_at_15",
        "f1_at_15",
        "lift_at_15",
        "precision_at_30",
        "lift_at_30",
        "pos_top_k_precision",
        "pos_top_k_lift",
    ]
    summary_df = detailed_df.groupby("model", as_index=False)[agg_cols].agg(["mean", "std"])
    summary_df.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in summary_df.columns]
    summary_df = summary_df.sort_values("hard_roc_auc_mean", ascending=False).reset_index(drop=True)

    detailed_df.to_csv(metrics_dir / "rolling_backtest_detailed.csv", index=False)
    summary_df.to_csv(metrics_dir / "model_benchmark_summary.csv", index=False)
    discoveries_df.to_csv(metrics_dir / "backtest_top_discoveries.csv", index=False)
    return detailed_df, summary_df, discoveries_df


def run_ablation(df, feature_groups, cfg):
    cutoffs = cfg["cohort"]["rolling_cutoffs"]
    seed = cfg["evaluation"]["random_state"]
    metrics_dir = cfg["resolved_paths"]["metrics_dir"]

    stages = [
        ("1. Raw Skills (34)", feature_groups["raw_skills"]),
        ("2. + Age & Physicals", feature_groups["raw_skills"] + feature_groups["demographics"]),
        (
            "3. + Trajectories",
            feature_groups["raw_skills"] + feature_groups["demographics"] + feature_groups["trajectories"],
        ),
        (
            "4. + Peer Z-Scores",
            feature_groups["raw_skills"]
            + feature_groups["demographics"]
            + feature_groups["trajectories"]
            + feature_groups["peer_z_scores"],
        ),
        ("5. + Club & League Context", feature_groups["all"]),
    ]

    rows = []
    for stage_name, cols in stages:
        aucs, pr_aucs, hard_aucs, p15s, lifts = [], [], [], [], []
        for cutoff in cutoffs:
            train_df, test_df = make_temporal_split(df, cutoff, cfg)
            w_train = compute_player_weights(train_df)
            model = make_xgb(random_state=seed)
            model.fit(
                train_df[cols].to_numpy(dtype=float),
                train_df["target_elite"].to_numpy(dtype=int),
                sample_weight=w_train,
            )
            probs = model.predict_proba(test_df[cols].to_numpy(dtype=float))[:, 1]
            y_test = test_df["target_elite"].to_numpy(dtype=int)
            ovr_test = test_df["overall"].to_numpy(dtype=float)
            hard_mask = ovr_test >= 70

            aucs.append(roc_auc_score(y_test, probs))
            pr_aucs.append(average_precision_score(y_test, probs))
            hard_aucs.append(roc_auc_score(y_test[hard_mask], probs[hard_mask]))
            p15 = float(y_test[np.argsort(-probs, kind="stable")[:15]].mean())
            p15s.append(p15)
            lifts.append(p15 / max(float(y_test.mean()), 1e-9))

        rows.append({
            "stage": stage_name,
            "n_features": len(cols),
            "mean_roc_auc": float(np.mean(aucs)),
            "std_roc_auc": float(np.std(aucs)),
            "mean_pr_auc": float(np.mean(pr_aucs)),
            "std_pr_auc": float(np.std(pr_aucs)),
            "mean_hard_roc_auc": float(np.mean(hard_aucs)),
            "mean_precision_at_15": float(np.mean(p15s)),
            "mean_lift_at_15": float(np.mean(lifts)),
        })

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(metrics_dir / "ablation_study.csv", index=False)
    return ablation_df
