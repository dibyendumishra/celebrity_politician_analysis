"""
tests/test_extractor.py
------------------------
Unit tests for the interaction extractor.
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import pytest

from src.interactions.extractor import _is_not_null, _classify_tweet, INTERACTION_COLUMNS


# ── _is_not_null ──────────────────────────────────────────────────────────────

def test_is_not_null_none():
    assert _is_not_null(None) is False

def test_is_not_null_nan():
    import math
    assert _is_not_null(float("nan")) is False

def test_is_not_null_value():
    assert _is_not_null("hello") is True
    assert _is_not_null({"key": "val"}) is True
    assert _is_not_null(0) is True


# ── _classify_tweet ───────────────────────────────────────────────────────────

TARGET_SCREENS = {"pol1", "pol2"}
PARTY_MAP = {"pol1": "BJP", "pol2": "INC"}

def _make_row(**kwargs):
    defaults = dict(
        screen_name="celeb1",
        full_text="some tweet",
        date="2020-01-01",
        id=1,
        lang="en",
        retweet_count=0,
        quoted_status=None,
        retweeted_status=None,
        in_reply_to_screen_name=None,
        entities={"user_mentions": []},
    )
    defaults.update(kwargs)
    return defaults


def test_classify_quote():
    row = _make_row(quoted_status={"user": {"screen_name": "pol1"}})
    result = _classify_tweet(row, TARGET_SCREENS, PARTY_MAP)
    assert result is not None
    assert result[5] == "quote"
    assert result[4] == "BJP"


def test_classify_retweet():
    row = _make_row(retweeted_status={"user": {"screen_name": "pol2"}})
    result = _classify_tweet(row, TARGET_SCREENS, PARTY_MAP)
    assert result is not None
    assert result[5] == "rt"
    assert result[4] == "INC"


def test_classify_reply():
    row = _make_row(in_reply_to_screen_name="pol1")
    result = _classify_tweet(row, TARGET_SCREENS, PARTY_MAP)
    assert result is not None
    assert result[5] == "reply"


def test_classify_mention():
    row = _make_row(entities={"user_mentions": [{"screen_name": "pol2"}]})
    result = _classify_tweet(row, TARGET_SCREENS, PARTY_MAP)
    assert result is not None
    assert result[5] == "mention"


def test_classify_no_match():
    row = _make_row(entities={"user_mentions": [{"screen_name": "nobody"}]})
    result = _classify_tweet(row, TARGET_SCREENS, PARTY_MAP)
    assert result is None


def test_classify_no_text():
    row = _make_row(full_text=None)
    result = _classify_tweet(row, TARGET_SCREENS, PARTY_MAP)
    assert result is None


def test_classify_quote_priority_over_rt():
    """Quote takes priority when both quoted_status and retweeted_status are present."""
    row = _make_row(
        quoted_status={"user": {"screen_name": "pol1"}},
        retweeted_status={"user": {"screen_name": "pol2"}},
    )
    result = _classify_tweet(row, TARGET_SCREENS, PARTY_MAP)
    assert result[5] == "quote"
