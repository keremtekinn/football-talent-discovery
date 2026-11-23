"""Run the full preprocessing, backtesting, FC 26 scoring, and plotting pipeline."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.evaluate import run_ablation, run_rolling_backtest
from src.features import add_horizon_labels, build_features
from src.predict import fit_and_predict_fc26
from src.preprocess import build_panel_dataset
from src.visualize import save_all_figures


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Football talent scouting pipeline")
    parser.add_argument("--skip-preprocess", action="store_true", help="Reuse processed panel_data.csv")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel_path = cfg["resolved_paths"]["panel_path"]

    if args.skip_preprocess and panel_path.exists():
        print(f"Loading panel from {panel_path}")
        panel = pd.read_csv(panel_path, low_memory=False)
    else:
        print("Building 12-year outfield panel (FIFA 15 -> FC 26)...")
        panel, audit = build_panel_dataset(cfg)
        print(
            f"Panel ready: {audit['clean_panel_rows']:,} rows, "
            f"{audit['unique_outfield_players']:,} unique players "
            f"(v24->v25: {audit['linkage_rates_pct']['v24_to_v25']}%, "
            f"v25->v26: {audit['linkage_rates_pct']['v25_to_v26']}%)"
        )

    print("Building features and 4-year horizon labels...")
    featured_df, feature_groups = build_features(panel, cfg)
    labeled_df = add_horizon_labels(
        featured_df,
        horizon_years=cfg["cohort"]["horizon_years"],
        elite_threshold=cfg["cohort"]["elite_threshold"],
    )

    print("Running rolling backtests (2020-2022) and feature ablation...")
    _, summary_df, _ = run_rolling_backtest(labeled_df, feature_groups["all"], cfg)
    ablation_df = run_ablation(labeled_df, feature_groups, cfg)

    cols_show = [
        "model",
        "roc_auc_mean",
        "hard_roc_auc_mean",
        "pr_auc_mean",
        "brier_skill_score_mean",
        "precision_at_15_mean",
        "lift_at_15_mean",
    ]
    print("\nBenchmark summary (mean over 2020-2022 cutoffs):")
    print(summary_df[cols_show].to_string(index=False))

    print("\nFeature ablation (XGBoost):")
    print(
        ablation_df[["stage", "n_features", "mean_roc_auc", "mean_hard_roc_auc", "mean_pr_auc", "mean_lift_at_15"]].to_string(
            index=False
        )
    )

    print("\nFitting final models on FIFA 15-22 and scoring FC 26 youth cohort...")
    pred_df, _, shap_exp, fc26_df = fit_and_predict_fc26(labeled_df, feature_groups["all"], cfg)
    print(f"Scored {len(pred_df):,} FC 26 prospects. Top 10 overall:")
    print(
        pred_df[
            ["overall_rank", "player_name", "pos_group", "age", "current_ovr", "club", "xgb_prob", "ensemble_prob"]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nSaving figures...")
    fig_paths = save_all_figures(panel, summary_df, ablation_df, shap_exp, fc26_df, pred_df, cfg)
    for p in fig_paths:
        print(f"  {p.name}")
    print("Done.")


if __name__ == "__main__":
    main()
