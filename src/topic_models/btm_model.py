"""
src/topical_models/btm_model.py
--------------------------------
Biterm Topic Model (BTM) pipeline:
  1. Text preprocessing (URL removal, stopword filtering, lemmatisation)
  2. Bigram / trigram phrase modelling
  3. BTM fitting with coherence-based topic count selection
  4. Saving results (topics per document, overall topic summaries)
"""

from __future__ import annotations

import logging
import os
import re
import string
import time
from pathlib import Path
from typing import List, Optional, Tuple

import en_core_web_sm
import gensim
import gensim.corpora as corpora
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

# ── Text preprocessing ────────────────────────────────────────────────────────

_EXTRA_STOPWORDS = {"amp", "rt", "nt"}
_PUNCT_DIGITS = string.punctuation.replace("@", "").replace("#", "") + "1234567890"
_PUNCT_RE = re.compile("|".join(map(re.escape, _PUNCT_DIGITS)))


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+", "", text)


def remove_newlines(text: str) -> str:
    return re.sub(r"\n", " ", text)


def remove_extra_spaces(text: str) -> str:
    return re.sub(r" {2,}", " ", text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    sw = set(stopwords.words("english")) | _EXTRA_STOPWORDS
    return [w for w in tokens if w not in sw]


def remove_meta(tokens: List[str]) -> List[str]:
    """Drop @mentions and empty strings."""
    return [w for w in tokens if "@" not in w and w != ""]


def preprocess(text: str) -> List[str]:
    """
    Full preprocessing pipeline for a single tweet string.

    Returns a list of cleaned, ASCII-only tokens.
    """
    text = remove_urls(text).lower()
    text = remove_newlines(text)
    text = remove_extra_spaces(text)
    text = _PUNCT_RE.sub(" ", text)
    tokens = text.split()
    tokens = remove_stopwords(tokens)
    tokens = remove_meta(tokens)
    tokens = [t for t in tokens if t.isascii()]
    return tokens


# ── Phrase models ─────────────────────────────────────────────────────────────

def build_phrase_models(
    data_words: List[List[str]],
    min_count: int = 5,
    bigram_threshold: float = 60.0,
    trigram_threshold: float = 60.0,
) -> Tuple[gensim.models.phrases.Phraser, gensim.models.phrases.Phraser]:
    """Return (bigram_mod, trigram_mod) Phraser objects."""
    bigram = gensim.models.Phrases(data_words, min_count=min_count, threshold=bigram_threshold)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=trigram_threshold)
    return gensim.models.phrases.Phraser(bigram), gensim.models.phrases.Phraser(trigram)


def apply_bigrams(data_words: List[List[str]], bigram_mod) -> List[List[str]]:
    return [bigram_mod[doc] for doc in data_words]


# ── Lemmatisation ─────────────────────────────────────────────────────────────

def lemmatize(texts: List[List[str]], nlp=None) -> List[List[str]]:
    """Lemmatise a list of tokenised texts using spaCy."""
    if nlp is None:
        nlp = en_core_web_sm.load()
    return [[token.lemma_ for token in nlp(" ".join(sent))] for sent in texts]


# ── BTM fitting ───────────────────────────────────────────────────────────────

def fit_btm(
    df: pd.DataFrame,
    topic_range: range,
    iterations: int = 40,
    max_df: float = 0.95,
    min_df: int = 5,
) -> Tuple[np.ndarray, object, object]:
    """
    Fit a BTM model, selecting the number of topics by coherence.

    Parameters
    ----------
    df          : DataFrame with a ``words`` column (space-joined lemmas).
    topic_range : range of topic counts to try.
    iterations  : BTM fitting iterations.
    max_df / min_df : CountVectorizer document-frequency thresholds.

    Returns
    -------
    topics      : Document-topic matrix for the best model.
    best_summary: topic_summuary output for the best model.
    btm         : The fitted oBTM object.
    """
    try:
        from biterm.btm import oBTM
        from biterm.utility import vec_to_biterms, topic_summuary
    except ImportError as exc:
        raise ImportError("Install biterm: pip install biterm") from exc

    vec = CountVectorizer(
        max_df=max_df, min_df=min_df, token_pattern=r"\w+|\$[\d\.]+|\S+"
    )
    X = vec.fit_transform(df["words"]).toarray()
    vocab = np.array(vec.get_feature_names_out())
    biterms = vec_to_biterms(X)

    coherence_scores: List[float] = []
    all_topics: List[np.ndarray] = []
    all_summaries: List[object] = []

    for n in topic_range:
        btm = oBTM(num_topics=n, V=vocab)
        t_start = time.time()
        topics = btm.fit_transform(biterms, iterations=iterations)
        summary = topic_summuary(btm.phi_wz.T, X, vocab, 20, verbose=False)
        coherence_scores.append(float(np.mean(list(summary["coherence"]))))
        all_topics.append(topics)
        all_summaries.append(summary)
        logger.info("n_topics=%d  coherence=%.4f  time=%.1fs", n, coherence_scores[-1], time.time() - t_start)

    best_ix = int(np.argmax(coherence_scores))
    logger.info("Best n_topics=%d  coherence=%.4f", list(topic_range)[best_ix], coherence_scores[best_ix])
    return all_topics[best_ix], all_summaries[best_ix], btm


# ── Save results ─────────────────────────────────────────────────────────────

def save_btm_results(
    df: pd.DataFrame,
    topics: np.ndarray,
    summary,
    coherence_scores: List[float],
    out_dir: str,
) -> None:
    """Write topic model outputs to *out_dir*."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["topics"] = list(topics)
    df["dominant_topic"] = df["topics"].apply(lambda x: int(np.argmax(x)))

    coherence_df = pd.DataFrame({"coherence_scores": coherence_scores})
    topic_rows = [
        [i, list(words), summary["coherence"][i]]
        for i, words in enumerate(summary["top_words"])
    ]
    topic_df = pd.DataFrame(topic_rows, columns=["topic_no", "top_words", "topic_coherence"])

    coherence_df.to_csv(os.path.join(out_dir, "rt_coherence.csv"), index=False)
    topic_df.to_csv(os.path.join(out_dir, "rt_overall_topics.csv"), index=False)
    df.to_json(os.path.join(out_dir, "rt_document_topics.json"))
    logger.info("BTM results saved to %s", out_dir)
