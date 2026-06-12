"""
src/visualization/plots.py
---------------------------
Reusable plotting functions for the celebrity-politician analysis.

All functions accept a ``config`` dict and an optional ``ax`` / figure
argument so they can be embedded in larger composite figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_output_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, path: str, dpi: int = 300) -> None:
    _ensure_output_dir(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure → %s", path)
    plt.close(fig)


# ── Topic frequency bar charts ────────────────────────────────────────────────

def plot_topic_freq_by_party(
    df: pd.DataFrame,
    category: str,
    palette: Dict[str, str],
    title: str,
    out_path: str,
    dpi: int = 300,
) -> None:
    """
    Bar chart of topic label frequency by party for a given celebrity category.

    Parameters
    ----------
    df       : DataFrame with columns [party, category, topic_label, freq].
    category : "sports" or "entertainment".
    palette  : Colour map for topic labels.
    title    : Plot title.
    out_path : File path for saved figure.
    dpi      : Output resolution.
    """
    subset = df[df["category"] == category]
    g = sns.catplot(
        x="party", y="freq", hue="topic_label",
        kind="bar", data=subset, palette=palette,
    )
    g.fig.suptitle(title)
    _ensure_output_dir(out_path)
    g.savefig(out_path, dpi=dpi)
    plt.close(g.fig)
    logger.info("Saved → %s", out_path)


# ── Monthly trend line charts ─────────────────────────────────────────────────

def plot_monthly_trends(
    df: pd.DataFrame,
    category: str,
    palette: Dict[str, str],
    title: str,
    out_path: str,
    dpi: int = 300,
) -> None:
    """
    Line plot of monthly topic-label frequency, split by party.

    Parameters
    ----------
    df       : DataFrame with columns [month, category, topic_label, party, freq].
    category : "sports" or "entertainment".
    palette  : Colour map for topic labels.
    title    : Plot title.
    out_path : File path for saved figure.
    dpi      : Output resolution.
    """
    subset = df[df["category"] == category]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        x="month", y="freq", style="party", hue="topic_label",
        data=subset, palette=palette, ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save(fig, out_path, dpi=dpi)


# ── Partisanship histogram ────────────────────────────────────────────────────

def plot_partisanship_histograms(
    plotdf: pd.DataFrame,
    interaction_types: List[str],
    vocations: List[str],
    out_path: str,
    ylims: Optional[Dict[str, int]] = None,
    dpi: int = 300,
) -> None:
    """
    2 × N grid of partisanship score histograms (one row per vocation,
    one column per interaction type), stacked by gender.

    Partisanship = eng_with_bjp_<type> − eng_with_inc_<type>
    (positive → BJP leaning, negative → INC leaning).

    Parameters
    ----------
    plotdf           : Engagement DataFrame with partisanship_<type> columns.
    interaction_types: e.g. ["rt", "quote", "mention", "reply"].
    vocations        : e.g. ["entertainment", "sports"].
    out_path         : File path for saved figure.
    ylims            : Optional {vocation: y_axis_max} dict.
    dpi              : Output resolution.
    """
    n_cols = len(interaction_types)
    n_rows = len(vocations)
    xlabels = {"rt": "Retweet", "quote": "Quote", "mention": "Mention", "reply": "Reply"}
    ylims = ylims or {}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             sharex=False, sharey=False)
    axes = axes.reshape(n_rows, n_cols)

    for row_ix, vocation in enumerate(vocations):
        for col_ix, ttype in enumerate(interaction_types):
            ax = axes[row_ix, col_ix]
            col = f"partisanship_{ttype}"
            eng_bjp = f"eng_with_bjp_{ttype}"
            eng_inc = f"eng_with_inc_{ttype}"

            subset = plotdf[
                ~((plotdf[eng_bjp] == 0) & (plotdf[eng_inc] == 0))
                & (plotdf["vocation"] == vocation)
            ]
            sns.histplot(
                data=subset, x=col, hue="gender", fill=True,
                hue_order=["M", "F"], multiple="stack", ax=ax,
            )
            for gender, colour in [("M", "b"), ("F", "r")]:
                mean_val = subset[subset["gender"] == gender][col].mean()
                ax.axvline(x=mean_val, color=colour, linestyle="--")

            ax.axvline(x=0, color="k", linestyle="-")
            ax.grid(axis="y")
            ax.set_xlabel(xlabels.get(ttype, ttype), fontsize=13)
            if vocation in ylims:
                ax.set_ylim([0, ylims[vocation]])
            if col_ix == 0:
                ax.set_ylabel(vocation.capitalize(), fontsize=13)

    n_celebs = plotdf["screen_name"].nunique() if "screen_name" in plotdf.columns else len(plotdf)
    fig.suptitle(f"Celebrity engagement with politicians, N={n_celebs}", fontsize=14)
    fig.tight_layout()
    _save(fig, out_path, dpi=dpi)


# ── Language / tweet-type distribution ───────────────────────────────────────

def plot_language_distribution(
    data_by_group: Dict[str, pd.DataFrame],
    titles: List[str],
    pal: Dict[str, str],
    out_path: str,
    dpi: int = 300,
) -> None:
    """
    Stacked bar chart of language × tweet-type frequency for each group.

    Parameters
    ----------
    data_by_group : {label: DataFrame with columns [language, tweet_type, frequency]}.
    titles        : Title for each subplot (same order as data_by_group).
    pal           : Colour map for tweet_type.
    out_path      : File path for saved figure.
    dpi           : Output resolution.
    """
    n = len(data_by_group)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    sns.set_style("whitegrid")

    for ax, (label, df), title in zip(axes, data_by_group.items(), titles):
        sns.histplot(
            df, x="language", weights="frequency", hue="tweet_type",
            multiple="stack", edgecolor="white", shrink=0.8, palette=pal, ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="x")
        ax.set_title(title)

    fig.tight_layout()
    _save(fig, out_path, dpi=dpi)


# ── Topic-bucket distribution ─────────────────────────────────────────────────

def plot_topic_bucket_distribution(
    df: pd.DataFrame,
    pal: Dict[str, str],
    title: str,
    out_path: str,
    ax: Optional[plt.Axes] = None,
    dpi: int = 300,
) -> None:
    """
    Stacked histogram of topic-bucket frequency by tweet type.

    Parameters
    ----------
    df       : DataFrame with columns [bucket, tweet_type, frequency].
    pal      : Colour map for tweet_type.
    title    : Subplot title.
    out_path : File path (only used when *ax* is None).
    ax       : If provided, draw into this Axes; otherwise create a new figure.
    dpi      : Output resolution (ignored when *ax* is provided).
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(
        df, x="bucket", weights="frequency", hue="tweet_type",
        multiple="stack", edgecolor="white", shrink=0.8, palette=pal, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(axis="x")
    ax.set_title(title)

    if own_fig:
        fig.tight_layout()
        _save(fig, out_path, dpi=dpi)
