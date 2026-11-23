"""Charts for EDA, trajectories, model comparison, ablation, and SHAP values."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

sns.set_theme(style="whitegrid", context="notebook")

FEATURE_LABELS = {
    "club_teammate_avg_ovr": "Club Teammates Avg OVR",
    "league_peer_avg_ovr": "League Avg OVR",
    "age": "Age",
    "height_cm": "Height (cm)",
    "weight_kg": "Weight (kg)",
    "weak_foot": "Weak Foot",
    "skill_moves": "Skill Moves",
    "years_observed": "Years Observed",
    "has_history": "Has Prior History",
    "mean_all_skills": "Mean of 34 Skills",
    "top5_skill_mean": "Mean of Top-5 Skills",
    "skill_std": "Skill Profile Std",
    "movement_reactions": "Reactions",
    "skill_ball_control": "Ball Control",
    "skill_dribbling": "Dribbling (Sub-skill)",
    "attacking_short_passing": "Short Passing",
    "mentality_vision": "Vision",
    "mentality_positioning": "Positioning",
    "movement_sprint_speed": "Sprint Speed",
    "movement_acceleration": "Acceleration",
    "movement_agility": "Agility",
    "power_stamina": "Stamina",
    "power_strength": "Strength",
    "defending_marking_awareness": "Defensive Awareness",
    "defending_standing_tackle": "Standing Tackle",
    "mean_all_skills_peer_z": "Mean Skills (Age/Pos Z-Score)",
    "top5_skill_mean_peer_z": "Top-5 Skills (Age/Pos Z-Score)",
    "movement_reactions_peer_z": "Reactions (Age/Pos Z-Score)",
    "skill_ball_control_peer_z": "Ball Control (Age/Pos Z-Score)",
    "pace_peer_z": "Pace (Age/Pos Z-Score)",
    "dribbling_peer_z": "Dribbling (Age/Pos Z-Score)",
    "passing_peer_z": "Passing (Age/Pos Z-Score)",
    "defending_peer_z": "Defending (Age/Pos Z-Score)",
    "physic_peer_z": "Physic (Age/Pos Z-Score)",
    "mean_all_skills_growth_rate": "Mean Skills Growth (/yr)",
    "mean_all_skills_yoy_delta": "Mean Skills YoY Delta",
    "movement_reactions_growth_rate": "Reactions Growth (/yr)",
    "movement_reactions_yoy_delta": "Reactions YoY Delta",
    "skill_ball_control_growth_rate": "Ball Control Growth (/yr)",
    "skill_ball_control_yoy_delta": "Ball Control YoY Delta",
}


def format_feature_name(name):
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


def plot_eda(panel, out_dir, elite_thr=82):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    peak_ovr = panel.groupby("player_id")["overall"].max()
    elite_pct = (peak_ovr >= elite_thr).mean() * 100
    sns.histplot(peak_ovr, bins=35, kde=True, color="#2E86C1", alpha=0.75, ax=axes[0])
    axes[0].axvline(
        x=elite_thr,
        color="#C0392B",
        linestyle="--",
        linewidth=2.2,
        label=f"{elite_thr}+ OVR Cutoff (Top {elite_pct:.2f}%)",
    )
    axes[0].set_title("Career Peak OVR Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Career Peak Overall Rating", fontsize=11)
    axes[0].set_ylabel("Unique Outfield Players", fontsize=11)
    axes[0].legend(frameon=True)

    # only consecutive editions, otherwise a 2-year gap gets counted as 1-year growth
    df_s = panel.sort_values(["player_id", "fifa_version"]).copy()
    next_ver = df_s.groupby("player_id")["fifa_version"].shift(-1)
    next_ovr = df_s.groupby("player_id")["overall"].shift(-1)
    consec = df_s[(next_ver - df_s["fifa_version"]) == 1].copy()
    consec["yoy_growth"] = next_ovr.loc[consec.index] - consec["overall"]

    age_growth = (
        consec[(consec["age"] >= 17) & (consec["age"] <= 32)]
        .groupby("age", as_index=False)["yoy_growth"]
        .mean()
    )
    colors = ["#1E8449" if a <= 21 else "#7F8C8D" for a in age_growth["age"]]
    axes[1].bar(age_growth["age"], age_growth["yoy_growth"], color=colors, edgecolor="white")
    axes[1].axhline(y=0.0, color="#2C3E50", linewidth=1.0)
    axes[1].axvline(x=21.5, color="#C0392B", linestyle="--", linewidth=2.0, label="Scout Age Limit (<= 21)")
    axes[1].set_title("Mean 1-Year OVR Change by Age", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Player Age", fontsize=11)
    axes[1].set_ylabel("Mean Annual OVR Change", fontsize=11)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    out_path = out_dir / "eda_overview.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_trajectories(panel, out_dir):
    targets = [
        ("Rafael Leão", 20, "Scouted in FIFA 20 (Age 20, OVR 75 -> Peak 86)"),
        ("William Saliba", 21, "Scouted in FIFA 21 (Age 19, OVR 74 -> Peak 87)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, (name_query, scout_ver, subtitle) in zip(axes, targets):
        cand = panel[panel["canonical_name"].str.contains(name_query, case=False, na=False)]
        if cand.empty:
            continue
        best_pid = cand.groupby("player_id")["overall"].max().idxmax()
        p_df = panel[(panel["player_id"] == best_pid) & (panel["fifa_version"] >= 18)].sort_values("fifa_version")

        labels = [f"FC {v}" if v >= 24 else f"FIFA {v}" for v in p_df["fifa_version"]]
        x_idx = np.arange(len(labels))

        ax.plot(x_idx, p_df["overall"], marker="o", linewidth=2.8, color="#1F618D", label="Overall (Target Only)")
        ax.plot(x_idx, p_df["skill_ball_control"], marker="s", linestyle="--", linewidth=2.0, color="#C0392B", label="Ball Control")
        ax.plot(x_idx, p_df["movement_reactions"], marker="^", linestyle="-.", linewidth=2.0, color="#1E8449", label="Reactions")
        ax.plot(x_idx, p_df["movement_sprint_speed"], marker="d", linestyle=":", linewidth=2.0, color="#D68910", label="Sprint Speed")

        scout_pos = np.where(p_df["fifa_version"].to_numpy() == scout_ver)[0]
        if len(scout_pos) > 0:
            s_idx = int(scout_pos[0])
            ax.axvspan(0, s_idx, color="#F9E79F", alpha=0.35, label="Observation Window")
            ax.axvline(x=s_idx, color="#7D3C98", linestyle="--", linewidth=1.8)

        ax.axhline(y=82, color="#922B21", linestyle=":", linewidth=1.5, alpha=0.8, label="82+ Elite Threshold")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"{name_query}\n{subtitle}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Attribute Rating (1-99)", fontsize=11)
        ax.set_ylim(55, 95)
        ax.legend(loc="lower right", fontsize=8.5, frameon=True)

    fig.suptitle("Multi-Year Attribute Trajectories: Backtest Breakout Examples", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = out_dir / "player_trajectories.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_benchmark(summary_df, out_dir):
    df_plot = summary_df.sort_values("hard_roc_auc_mean", ascending=True).copy()
    models = df_plot["model"].tolist()
    y_pos = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    specs = [
        ("roc_auc_mean", "roc_auc_std", "Full Cohort ROC-AUC (<=75 OVR)", "#2980B9"),
        ("hard_roc_auc_mean", "hard_roc_auc_std", "Hard Cohort ROC-AUC (70-75 OVR)", "#8E44AD"),
        ("pr_auc_mean", "pr_auc_std", "Mean PR-AUC", "#16A085"),
    ]

    for ax, (mean_col, std_col, title, color) in zip(axes, specs):
        means = df_plot[mean_col].to_numpy()
        stds = df_plot[std_col].fillna(0.0).to_numpy()
        bars = ax.barh(y_pos, means, xerr=stds, color=color, alpha=0.85, capsize=4, edgecolor="white")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=10)
        max_extent = float(np.max(means + stds))
        for bar, val, err in zip(bars, means, stds):
            fmt = f"{val:.3f}"
            ax.text(
                val + err + (max_extent * 0.03),
                bar.get_y() + bar.get_height() / 2,
                fmt,
                va="center",
                fontsize=9.5,
                fontweight="bold",
            )
        ax.set_xlim(0, max_extent * 1.24)

    fig.suptitle("Rolling Backtest Results (2020-2022 Cutoffs, 4-Year Horizon)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = out_dir / "model_benchmark.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ablation(ablation_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    stages = ablation_df["stage"].tolist()
    x = np.arange(len(stages))

    axes[0].plot(x, ablation_df["mean_hard_roc_auc"], marker="o", linewidth=2.6, markersize=8, color="#8E44AD")
    for xi, val in zip(x, ablation_df["mean_hard_roc_auc"]):
        axes[0].annotate(f"{val:.3f}", (xi, val), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages, rotation=20, ha="right", fontsize=9.5)
    axes[0].set_title("XGBoost Hard Cohort ROC-AUC (70-75 OVR)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("ROC-AUC (70-75 OVR Prospects)", fontsize=11)
    axes[0].set_ylim(0.58, ablation_df["mean_hard_roc_auc"].max() + 0.03)

    bars = axes[1].bar(x, ablation_df["mean_pr_auc"], color="#1ABC9C", alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, ablation_df["mean_pr_auc"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.3f}", ha="center", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stages, rotation=20, ha="right", fontsize=9.5)
    axes[1].set_title("XGBoost Mean PR-AUC by Feature Set", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("PR-AUC", fontsize=11)
    axes[1].set_ylim(0, ablation_df["mean_pr_auc"].max() * 1.22)

    fig.suptitle("Feature Ablation Study (XGBoost, 2020-2022 Rolling Cutoffs)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = out_dir / "ablation_study.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_shap(shap_exp, fc26_df, pred_df, out_dir):
    labels = [format_feature_name(f) for f in shap_exp.feature_names]
    exp = shap.Explanation(
        values=shap_exp.values,
        base_values=shap_exp.base_values,
        data=shap_exp.data,
        feature_names=labels,
    )

    plt.figure(figsize=(10.5, 7.0))
    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.title("XGBoost SHAP Summary: Top 15 Features (FC 26 Youth)", fontsize=12.5, fontweight="bold", pad=12)
    plt.tight_layout()
    global_path = out_dir / "shap_summary.png"
    plt.savefig(global_path, dpi=300, bbox_inches="tight")
    plt.close()

    top_row = pred_df.iloc[0]
    row_idx = int(fc26_df.index[fc26_df["player_id"] == int(top_row["player_id"])][0])

    plt.figure(figsize=(10.5, 6.5))
    shap.plots.waterfall(exp[row_idx], max_display=12, show=False)
    plt.title(
        f"SHAP Breakdown: {top_row['player_name']} "
        f"({top_row['primary_position']}, Age {top_row['age']}, OVR {top_row['current_ovr']}, "
        f"{top_row['club']} | Ensemble: {top_row['ensemble_prob']*100:.1f}%, XGB: {top_row['xgb_prob']*100:.1f}%)",
        fontsize=11.5,
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()
    local_path = out_dir / "shap_waterfall.png"
    plt.savefig(local_path, dpi=300, bbox_inches="tight")
    plt.close()

    return global_path, local_path


def save_all_figures(panel, summary_df, ablation_df, shap_exp, fc26_df, pred_df, cfg):
    out_dir = cfg["resolved_paths"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_eda(panel, out_dir, elite_thr=cfg["cohort"]["elite_threshold"]),
        plot_trajectories(panel, out_dir),
        plot_benchmark(summary_df, out_dir),
        plot_ablation(ablation_df, out_dir),
    ]
    paths.extend(plot_shap(shap_exp, fc26_df, pred_df, out_dir))
    return paths
