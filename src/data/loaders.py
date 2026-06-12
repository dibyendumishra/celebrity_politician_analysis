"""
src/data/loaders.py
-------------------
Data loading helpers.  All paths come from configs/config.yaml so callers
never hard-code file locations.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def load_politicians(config: dict) -> pd.DataFrame:
    """
    Load and clean the politician metadata CSV.

    Applies the purge list and individual-account filter defined in config,
    then balances each party to ``politicians_per_party`` rows.
    """
    df = pd.read_csv(config["data"]["politicians_csv"])

    purge = set(config.get("politician_purge_list", []))
    df = df[~df["screen_name"].isin(purge)]
    df = df[df["individual"] == 1]
    df = df.sort_values("followers_count", ascending=False)

    n = config.get("politicians_per_party", 1000)
    df_inc = df[df["party_x"] == "INC"].iloc[:n]
    df_bjp = df[df["party_x"] == "BJP"].iloc[:n]
    df = pd.concat([df_inc, df_bjp], ignore_index=True)

    logger.info(
        "Loaded politicians: INC=%d  BJP=%d",
        df[df["party_x"] == "INC"].shape[0],
        df[df["party_x"] == "BJP"].shape[0],
    )
    return df


def load_celebrity_engagement(config: dict) -> pd.DataFrame:
    """Load the celebrity engagement summary spreadsheet."""
    path = config["data"]["engagement_excel"]
    df = pd.read_excel(path)
    logger.info("Loaded celebrity engagement: %d rows", df.shape[0])
    return df


def load_document_topics(party: str, group: str, config: dict) -> pd.DataFrame:
    """
    Load the document-topics spreadsheet for a given party and user group.

    Parameters
    ----------
    party : "INC" or "BJP"
    group : "politicians" or "celebs"
    config : loaded config dict
    """
    base = Path(config["outputs"]["topic_models_dir"])
    path = base / group / party / "document_topics.xlsx"
    df = pd.read_excel(path)
    logger.info("Loaded document topics (%s / %s): %d rows", group, party, df.shape[0])
    return df


def build_politician_party_map(df_pol: pd.DataFrame) -> Dict[str, str]:
    """Return {screen_name: party} mapping from the politician DataFrame."""
    return df_pol.set_index("screen_name")["party_x"].to_dict()


def list_tweet_files(tweets_dir: str) -> List[str]:
    """Return a sorted list of JSON tweet file paths in *tweets_dir*."""
    return sorted(
        os.path.join(tweets_dir, f)
        for f in os.listdir(tweets_dir)
        if f.endswith(".json")
    )
