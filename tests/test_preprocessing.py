import pandas as pd
import pytest

from src.preprocess import (
    compute_age_on_july_1,
    extract_leading_number,
    extract_player_id_from_url,
    map_broad_position,
)


def test_extract_leading_number():
    heights = pd.Series(['182cm / 6\'0"', '195cm / 6\'5"', "178", None])
    parsed = extract_leading_number(heights)
    assert parsed.iloc[0] == 182.0
    assert parsed.iloc[1] == 195.0
    assert parsed.iloc[2] == 178.0
    assert pd.isna(parsed.iloc[3])


def test_extract_player_id_from_url():
    urls = pd.Series([
        "https://www.ea.com/games/ea-sports-fc/ratings/player-ratings/kylian-mbappe/231747",
        "https://www.ea.com/games/ea-sports-fc/ratings/player-ratings/erling-haaland/239085/",
        "invalid-url",
    ])
    ids = extract_player_id_from_url(urls)
    assert ids.iloc[0] == 231747
    assert ids.iloc[1] == 239085
    assert pd.isna(ids.iloc[2])


def test_compute_age_on_july_1():
    # Dec 20 1998 -> 26 on 2025-07-01; Jun 15 1992 -> 33; July 1 boundary check
    dobs = pd.Series([
        "12/20/1998 12:00:00 AM",
        "6/15/1992 12:00:00 AM",
        "2005-07-01",
        "2005-07-02",
    ])
    ages = compute_age_on_july_1(dobs, ref_year=2025)
    assert list(ages) == [26.0, 33.0, 20.0, 19.0]


@pytest.mark.parametrize(
    "raw_pos,expected",
    [
        ("ST, LW", "ATT"),
        ("RW", "ATT"),
        ("CAM, CM", "MID"),
        ("CDM", "MID"),
        ("CB", "DEF"),
        ("LWB, LB", "DEF"),
        ("GK", "GK"),
    ],
)
def test_map_broad_position(raw_pos, expected):
    assert map_broad_position(raw_pos) == expected
