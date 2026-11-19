"""Load and align FIFA 15-24, FC 25, and FC 26 datasets into a single outfield panel."""

import json
import numpy as np
import pandas as pd

POSITION_TO_GROUP = {
    "ST": "ATT", "CF": "ATT", "RW": "ATT", "LW": "ATT", "RF": "ATT", "LF": "ATT",
    "CAM": "MID", "CM": "MID", "CDM": "MID", "RM": "MID", "LM": "MID",
    "CB": "DEF", "RB": "DEF", "LB": "DEF", "RWB": "DEF", "LWB": "DEF",
}

MAP_FC25 = {
    "Name": "display_name",
    "OVR": "overall",
    "PAC": "pace",
    "SHO": "shooting",
    "PAS": "passing",
    "DRI": "dribbling",
    "DEF": "defending",
    "PHY": "physic",
    "Acceleration": "movement_acceleration",
    "Sprint Speed": "movement_sprint_speed",
    "Positioning": "mentality_positioning",
    "Finishing": "attacking_finishing",
    "Shot Power": "power_shot_power",
    "Long Shots": "power_long_shots",
    "Volleys": "attacking_volleys",
    "Penalties": "mentality_penalties",
    "Vision": "mentality_vision",
    "Crossing": "attacking_crossing",
    "Free Kick Accuracy": "skill_fk_accuracy",
    "Short Passing": "attacking_short_passing",
    "Long Passing": "skill_long_passing",
    "Curve": "skill_curve",
    "Dribbling": "skill_dribbling",
    "Agility": "movement_agility",
    "Balance": "movement_balance",
    "Reactions": "movement_reactions",
    "Ball Control": "skill_ball_control",
    "Interceptions": "mentality_interceptions",
    "Heading Accuracy": "attacking_heading_accuracy",
    "Def Awareness": "defending_marking_awareness",
    "Standing Tackle": "defending_standing_tackle",
    "Sliding Tackle": "defending_sliding_tackle",
    "Jumping": "power_jumping",
    "Stamina": "power_stamina",
    "Strength": "power_strength",
    "Aggression": "mentality_aggression",
    "Position": "player_positions",
    "Weak foot": "weak_foot",
    "Skill moves": "skill_moves",
    "Preferred foot": "preferred_foot",
    "Height": "height_cm",
    "Weight": "weight_kg",
    "Age": "age",
    "Nation": "nationality_name",
    "League": "league_name",
    "Team": "club_name",
}

MAP_FC26 = {
    "id": "player_id",
    "overallRating": "overall",
    "pac": "pace",
    "sho": "shooting",
    "pas": "passing",
    "dri": "dribbling",
    "def": "defending",
    "phy": "physic",
    "acceleration": "movement_acceleration",
    "sprintSpeed": "movement_sprint_speed",
    "positioning": "mentality_positioning",
    "finishing": "attacking_finishing",
    "shotPower": "power_shot_power",
    "longShots": "power_long_shots",
    "volleys": "attacking_volleys",
    "penalties": "mentality_penalties",
    "vision": "mentality_vision",
    "crossing": "attacking_crossing",
    "freeKickAccuracy": "skill_fk_accuracy",
    "shortPassing": "attacking_short_passing",
    "longPassing": "skill_long_passing",
    "curve": "skill_curve",
    "dribbling": "skill_dribbling",
    "ballControl": "skill_ball_control",
    "agility": "movement_agility",
    "balance": "movement_balance",
    "reactions": "movement_reactions",
    "interceptions": "mentality_interceptions",
    "headingAccuracy": "attacking_heading_accuracy",
    "defensiveAwareness": "defending_marking_awareness",
    "standingTackle": "defending_standing_tackle",
    "slidingTackle": "defending_sliding_tackle",
    "jumping": "power_jumping",
    "stamina": "power_stamina",
    "strength": "power_strength",
    "aggression": "mentality_aggression",
    "position": "player_positions",
    "weakFootAbility": "weak_foot",
    "skillMoves": "skill_moves",
    "preferredFoot": "preferred_foot",
    "height": "height_cm",
    "weight": "weight_kg",
    "birthdate": "dob",
    "nationality": "nationality_name",
    "leagueName": "league_name",
    "team": "club_name",
}


def extract_leading_number(series):
    # FC 25 stores height/weight like "182cm / 6'0\""
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")


def extract_player_id_from_url(url_series):
    # EA player_id is the last path segment in FC 25 rating URLs
    return pd.to_numeric(
        url_series.astype(str).str.rstrip("/").str.split("/").str[-1],
        errors="coerce",
    )


def compute_age_on_july_1(dob_series, ref_year):
    # FIFA 15-24 ages match exact age as of July 1 of (season_year - 1)
    dob = pd.to_datetime(dob_series, format="mixed", errors="coerce")
    after_july_1 = (dob.dt.month > 7) | ((dob.dt.month == 7) & (dob.dt.day > 1))
    return (ref_year - dob.dt.year - after_july_1.astype("Int64")).astype("float")


def map_broad_position(pos_str):
    if pd.isna(pos_str):
        return "MID"
    primary = str(pos_str).split(",")[0].strip().upper()
    if primary == "GK":
        return "GK"
    return POSITION_TO_GROUP.get(primary, "MID")


def load_fc_15_24(path):
    df = pd.read_csv(path, low_memory=False)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["display_name"] = df["short_name"].astype(str)
    df["dob"] = pd.to_datetime(df["dob"], format="mixed", errors="coerce")
    return df


def load_fc_25(path):
    raw = pd.read_csv(path, low_memory=False)
    df = raw.rename(columns=MAP_FC25).copy()
    df["player_id"] = extract_player_id_from_url(raw["url"])
    df["height_cm"] = extract_leading_number(df["height_cm"])
    df["weight_kg"] = extract_leading_number(df["weight_kg"])
    df["dob"] = pd.NaT
    df["fifa_version"] = 25
    return df


def load_fc_26(path):
    raw = pd.read_csv(path, low_memory=False)
    df = raw.rename(columns=MAP_FC26).copy()
    has_common = raw["commonName"].notna() & (raw["commonName"].astype(str).str.strip() != "")
    full_name = (
        raw["firstName"].fillna("").astype(str).str.strip()
        + " "
        + raw["lastName"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["display_name"] = np.where(has_common, raw["commonName"].astype(str).str.strip(), full_name)
    df["dob"] = pd.to_datetime(df["dob"], format="mixed", errors="coerce")
    df["age"] = compute_age_on_july_1(df["dob"], ref_year=2025)
    df["preferred_foot"] = df["preferred_foot"].map({1: "Right", 2: "Left"}).fillna("Right")
    df["height_cm"] = pd.to_numeric(df["height_cm"], errors="coerce")
    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")
    df["fifa_version"] = 26
    return df


def build_panel_dataset(cfg):
    paths = cfg["resolved_paths"]
    raw_dir = paths["raw_dir"]
    skill_cols = cfg["features"]["skill_cols"]

    keep_cols = [
        "player_id",
        "fifa_version",
        "display_name",
        "player_positions",
        "overall",
        "age",
        "dob",
        "height_cm",
        "weight_kg",
        "club_name",
        "league_name",
        "nationality_name",
        "preferred_foot",
        "weak_foot",
        "skill_moves",
    ] + skill_cols

    d15_24 = load_fc_15_24(raw_dir / cfg["raw_files"]["fc_15_24"])
    d25 = load_fc_25(raw_dir / cfg["raw_files"]["fc_25"])
    d26 = load_fc_26(raw_dir / cfg["raw_files"]["fc_26"])

    raw_total = len(d15_24) + len(d25) + len(d26)
    panel = pd.concat([d15_24[keep_cols], d25[keep_cols], d26[keep_cols]], ignore_index=True)

    panel = panel.dropna(subset=["player_id", "fifa_version", "overall"]).copy()
    panel["player_id"] = panel["player_id"].astype(int)
    panel["fifa_version"] = panel["fifa_version"].astype(int)

    # Propagate known date of birth across editions by player_id so FC 25 uses the same July-1 age cutoff
    known_dob = panel.dropna(subset=["dob"]).groupby("player_id")["dob"].first()
    panel["dob"] = panel["player_id"].map(known_dob)
    has_dob = panel["dob"].notna()
    after_july_1 = (panel["dob"].dt.month > 7) | ((panel["dob"].dt.month == 7) & (panel["dob"].dt.day > 1))
    dob_age = (1999 + panel["fifa_version"]) - panel["dob"].dt.year - after_july_1.astype("Int64")
    panel["age"] = np.where(has_dob, dob_age, panel["age"])
    panel = panel.dropna(subset=["age"]).copy()
    panel["age"] = panel["age"].round().astype(int)
    panel = panel.drop(columns=["dob"])

    # Drop goalkeepers since their outfield attributes are mostly empty/different
    gk_mask = panel["player_positions"].astype(str).str.contains("GK", case=False, na=False)
    gk_rows_removed = int(gk_mask.sum())
    panel = panel[~gk_mask].copy()

    panel["club_name"] = panel["club_name"].fillna("Free Agent").astype(str).str.strip()
    panel["league_name"] = panel["league_name"].fillna("Unknown League").astype(str).str.strip()
    panel["nationality_name"] = panel["nationality_name"].fillna("Unknown").astype(str).str.strip()
    panel["preferred_foot"] = panel["preferred_foot"].fillna("Right").astype(str).str.strip()
    panel["primary_position"] = panel["player_positions"].astype(str).str.split(",").str[0].str.strip().str.upper()
    panel["pos_group"] = panel["primary_position"].apply(map_broad_position)

    numeric_cols = ["overall", "age", "height_cm", "weight_kg", "weak_foot", "skill_moves"] + skill_cols
    for col in numeric_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel = (
        panel.sort_values(["player_id", "fifa_version"])
        .drop_duplicates(subset=["player_id", "fifa_version"], keep="first")
        .reset_index(drop=True)
    )

    panel["canonical_name"] = panel.groupby("player_id")["display_name"].transform("last")

    linkage_rates = {}
    age_step_exact_pct = {}
    versions = sorted(panel["fifa_version"].unique())
    prev_ver = panel.groupby("player_id")["fifa_version"].shift(1)
    prev_age = panel.groupby("player_id")["age"].shift(1)
    consec = panel[(panel["fifa_version"] - prev_ver) == 1]

    for v1, v2 in zip(versions[:-1], versions[1:]):
        s1 = set(panel.loc[panel["fifa_version"] == v1, "player_id"])
        s2 = set(panel.loc[panel["fifa_version"] == v2, "player_id"])
        linkage_rates[f"v{v1}_to_v{v2}"] = round(len(s1 & s2) / max(1, len(s2)) * 100, 2)
        c_v2 = consec[consec["fifa_version"] == v2]
        if len(c_v2) > 0:
            exact_plus_one = ((c_v2["age"] - prev_age.loc[c_v2.index]) == 1).mean() * 100
            age_step_exact_pct[f"v{v1}_to_v{v2}"] = round(float(exact_plus_one), 2)

    audit = {
        "raw_total_rows": raw_total,
        "gk_rows_removed": gk_rows_removed,
        "clean_panel_rows": int(len(panel)),
        "unique_outfield_players": int(panel["player_id"].nunique()),
        "versions": [int(v) for v in versions],
        "missing_skill_values": int(panel[skill_cols].isna().sum().sum()),
        "linkage_rates_pct": linkage_rates,
        "consecutive_age_plus_one_pct": age_step_exact_pct,
    }

    paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    paths["metrics_dir"].mkdir(parents=True, exist_ok=True)
    panel.to_csv(paths["panel_path"], index=False)
    with (paths["metrics_dir"] / "data_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    return panel, audit
