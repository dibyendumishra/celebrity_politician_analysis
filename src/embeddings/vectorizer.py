"""
src/embeddings/vectorizer.py
-----------------------------
TF-IDF weighted word2vec (GloVe) document vectorizer, plus helpers
for computing Word Mover's Distance and agglomerative clustering.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances

logger = logging.getLogger(__name__)


class TfidfEmbeddingVectorizer:
    """
    Compute TF-IDF–weighted average word-vector representations.

    Weights each token in a document by its IDF score before averaging the
    corresponding GloVe / word2vec vectors.  Unknown tokens receive the
    maximum IDF weight (i.e. are treated as rare / informative).

    Parameters
    ----------
    word2vec : gensim KeyedVectors (or any dict-like mapping word → vector).
    dim      : Embedding dimensionality (default 50 for glove-twitter-50).
    """

    def __init__(self, word2vec, dim: int = 50):
        self.word2vec = word2vec
        self.dim = dim
        self.word2weight: Optional[defaultdict] = None

    def fit(self, X: List[List[str]]) -> "TfidfEmbeddingVectorizer":
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            {w: tfidf.idf_[i] for w, i in tfidf.vocabulary_.items()},
        )
        return self

    def transform(self, X: List[List[str]]) -> np.ndarray:
        if self.word2weight is None:
            raise RuntimeError("Call .fit() before .transform()")
        return np.array(
            [
                np.mean(
                    [
                        self.word2vec[w] * self.word2weight[w]
                        for w in words
                        if w in self.word2vec
                    ]
                    or [np.zeros(self.dim)],
                    axis=0,
                )
                for words in X
            ]
        )

    def fit_transform(self, X: List[List[str]]) -> np.ndarray:
        return self.fit(X).transform(X)


# ── Word Mover's Distance metric ──────────────────────────────────────────────

def wmd_metric(data: List[List[str]], model):
    """
    Return a distance function suitable for ``sklearn.metrics.pairwise_distances``.

    The returned callable accepts index arrays (as produced by the pairwise
    API) and looks up the corresponding documents in *data*.
    """

    def _metric(x: np.ndarray, y: np.ndarray) -> float:
        score = model.wmdistance(data[int(x[0])], data[int(y[0])])
        return 0.0 if score == math.inf else float(score)

    return _metric


def compute_wmd_distance_matrix(
    data: List[List[str]], model, verbose: bool = True
) -> np.ndarray:
    """
    Compute the full pairwise WMD distance matrix for *data*.

    Parameters
    ----------
    data    : List of tokenised documents.
    model   : gensim KeyedVectors with ``wmdistance`` support.
    verbose : Log progress.
    """
    n = len(data)
    if verbose:
        logger.info("Computing WMD distance matrix for %d documents …", n)
    X = np.arange(n).reshape(-1, 1)
    metric = wmd_metric(data, model)
    return pairwise_distances(X, X, metric=metric)


# ── Agglomerative clustering ──────────────────────────────────────────────────

def cluster_documents(
    embeddings: np.ndarray,
    n_clusters: int = 4,
    linkage: str = "average",
    affinity: str = "euclidean",
) -> np.ndarray:
    """
    Run agglomerative clustering on pre-computed document embeddings.

    For pre-computed distance matrices pass ``affinity='precomputed'``.

    Returns
    -------
    labels : np.ndarray of shape (n_documents,) with integer cluster labels.
    """
    agg = AgglomerativeClustering(
        n_clusters=n_clusters, metric=affinity, linkage=linkage
    )
    labels = agg.fit_predict(embeddings)
    logger.info(
        "Clustering complete: %d clusters, sizes=%s",
        n_clusters,
        [int((labels == c).sum()) for c in range(n_clusters)],
    )
    return labels
