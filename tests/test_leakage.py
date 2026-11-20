import numpy as np
import pandas as pd

from src.config import load_config
from src.features import add_horizon_labels, build_features, compute_player_weights, make_temporal_split


def _sample_panel():
    cfg = load_config()
    skill_cols = cfg["features"]["skill_cols"]
    rows = []

    # Player 101 skips v18 and reaches 84 in v20 (4y after v16)
    for v, age, ovr, val in [(16, 18, 68, 72), (17, 19, 72, 76), (19, 21, 75, 80), (20, 22, 84, 86)]:
        row = {
            "player_id": 101,
            "fifa_version": v,
            "display_name": "Player A",
            "canonical_name": "Player A",
            "player_positions": "LW",
            "primary_position": "LW",
            "pos_group": "ATT",
            "overall": ovr,
            "age": age,
            "height_cm": 180.0,
            "weight_kg": 74.0,
            "club_name": "Club A",
            "league_name": "League 1",
            "nationality_name": "France",
            "preferred_foot": "Right",
            "weak_foot": 4,
            "skill_moves": 4,
        }
        for c in skill_cols:
            row[c] = float(val)
        rows.append(row)

    # Player 102 appears in both v16 and v20 -> should be dropped from train at cutoff=20
    for v, age, ovr, val in [(16, 17, 65, 70), (20, 21, 74, 78), (23, 24, 83, 84)]:
        row = {
            "player_id": 102,
            "fifa_version": v,
            "display_name": "Player B",
            "canonical_name": "Player B",
            "player_positions": "CM",
            "primary_position": "CM",
            "pos_group": "MID",
            "overall": ovr,
            "age": age,
            "height_cm": 178.0,
            "weight_kg": 72.0,
            "club_name": "Club A",
            "league_name": "League 1",
            "nationality_name": "Spain",
            "preferred_foot": "Left",
            "weak_foot": 3,
            "skill_moves": 3,
        }
        for c in skill_cols:
            row[c] = float(val)
        rows.append(row)

    # Player 103 stays below 82
    for v, age, ovr, val in [(16, 19, 64, 64), (17, 20, 65, 65), (20, 20, 67, 66), (24, 24, 70, 67)]:
        row = {
            "player_id": 103,
            "fifa_version": v,
            "display_name": "Player C",
            "canonical_name": "Player C",
            "player_positions": "CB",
            "primary_position": "CB",
            "pos_group": "DEF",
            "overall": ovr,
            "age": age,
            "height_cm": 188.0,
            "weight_kg": 82.0,
            "club_name": "Club A",
            "league_name": "League 1",
            "nationality_name": "Italy",
            "preferred_foot": "Right",
            "weak_foot": 2,
            "skill_moves": 2,
        }
        for c in skill_cols:
            row[c] = float(val)
        rows.append(row)

    return pd.DataFrame(rows)


def test_overall_not_in_features():
    cfg = load_config()
    _, groups = build_features(_sample_panel(), cfg)
    assert "overall" not in groups["all"]
    assert all("overall" not in col for col in groups["all"])


def test_leave_one_out_club_average():
    cfg = load_config()
    feat, _ = build_features(_sample_panel(), cfg)
    v16 = feat[feat["fifa_version"] == 16].set_index("player_id")
    assert np.isclose(v16.loc[101, "club_teammate_avg_ovr"], (65 + 64) / 2)
    assert np.isclose(v16.loc[102, "club_teammate_avg_ovr"], (68 + 64) / 2)
    assert np.isclose(v16.loc[103, "club_teammate_avg_ovr"], (68 + 65) / 2)


def test_yoy_delta_ignores_skipped_years():
    cfg = load_config()
    feat, _ = build_features(_sample_panel(), cfg)
    p101 = feat[feat["player_id"] == 101].set_index("fifa_version")
    assert p101.loc[16, "pace_yoy_delta"] == 0.0
    assert p101.loc[17, "pace_yoy_delta"] == 4.0
    assert p101.loc[19, "pace_yoy_delta"] == 0.0
    assert np.isclose(p101.loc[19, "pace_growth_rate"], 8.0 / 3.0)


def test_temporal_split_no_overlap_or_future_labels():
    cfg = load_config()
    feat, _ = build_features(_sample_panel(), cfg)
    labeled = add_horizon_labels(feat, horizon_years=4, elite_threshold=82)
    train_df, test_df = make_temporal_split(labeled, cutoff_version=20, cfg=cfg)

    train_pids = set(train_df["player_id"])
    test_pids = set(test_df["player_id"])
    assert train_pids.isdisjoint(test_pids)
    assert 102 in test_pids and 102 not in train_pids
    assert (train_df["fifa_version"] + cfg["cohort"]["horizon_years"] <= 20).all()
    assert int(train_df.loc[train_df["player_id"] == 101, "target_elite"].iloc[0]) == 1


def test_player_weights_equalize_repeated_observations():
    panel = _sample_panel()
    w = compute_player_weights(panel)
    weighted = panel.assign(w=w)
    per_player_totals = weighted.groupby("player_id")["w"].sum()
    assert np.allclose(per_player_totals.values, per_player_totals.iloc[0])
    assert np.isclose(w.mean(), 1.0)
