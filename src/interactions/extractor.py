"""
src/interactions/extractor.py
------------------------------
Extract celebrity ↔ politician interactions from raw tweet JSON files.

Supported interaction types:
  - rt      : retweet
  - quote   : quote-tweet
  - reply   : reply
  - mention : @mention (only when no other type applies)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Columns in the output DataFrame
INTERACTION_COLUMNS = [
    "date",
    "screen_name",
    "full_text",
    "target",
    "party",
    "type",
    "id",
    "lang",
    "retweet_count",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_not_null(value) -> bool:
    """Return True when *value* is a non-None, non-NaN object."""
    return value is not None and value == value


def _safe_read_json(path: str) -> Optional[pd.DataFrame]:
    """Read a JSON tweet file; return None on any error."""
    try:
        df = pd.read_json(path)
        return df if df.shape[0] > 0 else None
    except Exception as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return None


def _prepare_tweet_df(df: pd.DataFrame, date_start: str, date_end: str) -> Optional[pd.DataFrame]:
    """
    Add required columns and apply the date range filter.
    Returns None when the resulting DataFrame is empty.
    """
    for col in ("quoted_status", "retweeted_status"):
        if col not in df.columns:
            df[col] = None

    if "full_text" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["created_at"]).dt.date
    df = df[(df["created_at"] >= date_start) & (df["created_at"] <= date_end)]
    return df if df.shape[0] > 0 else None


# ── Core extraction ───────────────────────────────────────────────────────────

def _classify_tweet(row, target_screens: set, party_map: Dict[str, str]) -> Optional[list]:
    """
    Classify a single tweet row and return an interaction record list,
    or None if no relevant interaction is found.

    Priority: quote > retweet > reply > mention
    """
    txt = row.get("full_text")
    if not txt:
        return None

    # Quote tweet
    qs = row.get("quoted_status")
    if _is_not_null(qs):
        target = qs["user"]["screen_name"]
        if target in target_screens:
            return [row["date"], row["screen_name"], txt, target,
                    party_map[target], "quote", row["id"], row["lang"], row["retweet_count"]]

    # Retweet
    rs = row.get("retweeted_status")
    if _is_not_null(rs):
        target = rs["user"]["screen_name"]
        if target in target_screens:
            return [row["date"], row["screen_name"], txt, target,
                    party_map[target], "rt", row["id"], row["lang"], row["retweet_count"]]

    # Reply
    rp = row.get("in_reply_to_screen_name")
    if _is_not_null(rp) and rp in target_screens:
        return [row["date"], row["screen_name"], txt, rp,
                party_map[rp], "reply", row["id"], row["lang"], row["retweet_count"]]

    # Mention (any user mention that matches a target)
    entities = row.get("entities") or {}
    for mention in entities.get("user_mentions", []):
        mention_screen = mention.get("screen_name", "")
        if mention_screen in target_screens:
            return [row["date"], row["screen_name"], txt, mention_screen,
                    party_map[mention_screen], "mention", row["id"], row["lang"], row["retweet_count"]]

    return None


def extract_interactions(
    source_files: Sequence[str],
    source_screens: set,
    target_screens: set,
    party_map: Dict[str, str],
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    Scan tweet JSON files for all interactions from *source_screens* directed
    at *target_screens*.

    Parameters
    ----------
    source_files  : Paths to JSON tweet files to scan.
    source_screens: Screen names whose files to scan (files not in this set are skipped).
    target_screens: Screen names to look for as interaction targets.
    party_map     : Mapping from target screen name → party label.
    date_start    : ISO date string (inclusive).
    date_end      : ISO date string (inclusive).

    Returns
    -------
    pd.DataFrame with columns defined in INTERACTION_COLUMNS.
    """
    records: List[list] = []
    n_files = len(source_files)

    for i, fpath in enumerate(source_files):
        screen_name = Path(fpath).stem
        if screen_name not in source_screens:
            continue

        if i % 250 == 0:
            logger.info("Processing file %d / %d", i, n_files)

        df = _safe_read_json(fpath)
        if df is None:
            continue

        df = _prepare_tweet_df(df, date_start, date_end)
        if df is None:
            continue

        df["screen_name"] = screen_name
        for _, row in df.iterrows():
            record = _classify_tweet(row, target_screens, party_map)
            if record is not None:
                records.append(record)

    result = pd.DataFrame(records, columns=INTERACTION_COLUMNS)
    logger.info(
        "Extracted %d interactions (BJP=%d  INC=%d)",
        len(result),
        (result["party"] == "BJP").sum(),
        (result["party"] == "INC").sum(),
    )
    return result


def log_interaction_summary(df: pd.DataFrame, label: str = "") -> None:
    """Print a short summary of an interactions DataFrame."""
    prefix = f"[{label}] " if label else ""
    total = df.shape[0]
    bjp = (df["party"] == "BJP").sum()
    inc = (df["party"] == "INC").sum()
    non_rt = df[df["type"] != "rt"]
    logger.info("%sTotal: %d  BJP: %d  INC: %d", prefix, total, bjp, inc)
    logger.info(
        "%sNon-RT — BJP: %d  INC: %d",
        prefix,
        (non_rt["party"] == "BJP").sum(),
        (non_rt["party"] == "INC").sum(),
    )
