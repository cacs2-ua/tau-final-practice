from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TfidfConfig:
    """
    Configuration for a TF-IDF (TfidfVectorizer) run.

    Required toggles for the assignment:
      - lowercase: ON vs OFF
      - stop_words: None vs 'english'
    """
    name: str
    lowercase: bool = True
    stop_words: Optional[str] = None
    max_features: Optional[int] = None
    ngram_range: Tuple[int, int] = (1, 1)

    # Keep sklearn defaults explicit for clarity in the report
    norm: Optional[str] = "l2"
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = False


def _validate_texts(texts: Sequence[str]) -> None:
    if not isinstance(texts, (list, tuple)):
        raise TypeError(f"texts must be a list/tuple of strings, got {type(texts)}")
    if len(texts) == 0:
        raise ValueError("texts is empty (N=0).")
    if any(not isinstance(t, str) for t in texts):
        bad = next(type(t) for t in texts if not isinstance(t, str))
        raise TypeError(f"All texts must be strings. Found non-string: {bad}")
    if any(len(t) == 0 for t in texts):
        raise ValueError("texts contains empty strings (unexpected for this dataset).")


def build_tfidf_vectors(
    texts: Sequence[str],
    config: TfidfConfig,
):
    """
    Fit a TfidfVectorizer and return:
      - vectorizer (fitted)
      - X (document-term TF-IDF matrix, sparse)
    """
    _validate_texts(texts)

    vectorizer = TfidfVectorizer(
        lowercase=config.lowercase,
        stop_words=config.stop_words,
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        norm=config.norm,
        use_idf=config.use_idf,
        smooth_idf=config.smooth_idf,
        sublinear_tf=config.sublinear_tf,
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def sparse_matrix_stats(X) -> Dict[str, Any]:
    """Compute basic inspection stats for a sparse doc-term matrix."""
    n_docs, n_vocab = X.shape
    nnz = int(X.nnz)
    total = int(n_docs) * int(n_vocab)
    density = float(nnz / total) if total > 0 else 0.0
    avg_nnz_per_doc = float(nnz / n_docs) if n_docs > 0 else 0.0

    return {
        "n_docs": int(n_docs),
        "n_vocab": int(n_vocab),
        "shape": (int(n_docs), int(n_vocab)),
        "nnz": nnz,
        "density": density,
        "avg_nnz_per_doc": avg_nnz_per_doc,
        "matrix_type": type(X).__name__,
        "dtype": str(getattr(X, "dtype", "unknown")),
    }


def feature_examples(vectorizer: TfidfVectorizer, n: int = 20) -> List[str]:
    feats = vectorizer.get_feature_names_out()
    return feats[:n].tolist()


def case_sensitive_stopword_survivors(vectorizer: TfidfVectorizer) -> int:
    """
    Detect the nuance: with lowercase=False and stop_words='english',
    capitalized variants of stopwords (e.g., 'The') can survive.

    Returns how many vocabulary terms are "stopwords after lowercasing"
    but are not exactly in the stopword list (case mismatch).
    """
    stop = vectorizer.get_stop_words()
    if not stop:
        return 0

    feats = vectorizer.get_feature_names_out()
    stop_set = set(stop)

    survivors = 0
    for f in feats:
        fl = f.lower()
        if fl in stop_set and f not in stop_set:
            survivors += 1
    return survivors


def l2_norm_sample_stats(X, sample_n: int = 50, seed: int = 0) -> Dict[str, float]:
    """
    TF-IDF vectors in sklearn are L2-normalized by default (norm='l2').
    This helper checks that empirically on a sample of rows.
    """
    n_docs = X.shape[0]
    if n_docs == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}

    sample_n = int(min(sample_n, n_docs))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_docs, size=sample_n, replace=False)

    Xs = X[idx]
    norms = np.sqrt(np.asarray(Xs.multiply(Xs).sum(axis=1)).ravel())

    return {
        "min": float(np.min(norms)),
        "mean": float(np.mean(norms)),
        "max": float(np.max(norms)),
    }


def top_terms_for_doc(X, vectorizer: TfidfVectorizer, doc_index: int, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Return top-N (term, tf-idf weight) pairs for one document row in a sparse matrix.
    """
    feats = vectorizer.get_feature_names_out()
    row = X.getrow(doc_index)
    if row.nnz == 0:
        return []

    order = np.argsort(row.data)[::-1][:top_n]
    terms = [(feats[row.indices[i]], float(row.data[i])) for i in order]
    return terms


def run_tfidf_grid(texts: Sequence[str], sample_n_for_norms: int = 50) -> Dict[str, Dict[str, Any]]:
    """
    Run the 2x2 grid required by the assignment:
      - lowercase ON/OFF
      - stopwords OFF/ON ('english')

    Returns a dict keyed by config name with vectorizer, X, stats, norms.
    """
    configs = [
        TfidfConfig(name="tfidf_lc_on_sw_off",  lowercase=True,  stop_words=None),
        TfidfConfig(name="tfidf_lc_on_sw_on",   lowercase=True,  stop_words="english"),
        TfidfConfig(name="tfidf_lc_off_sw_off", lowercase=False, stop_words=None),
        TfidfConfig(name="tfidf_lc_off_sw_on",  lowercase=False, stop_words="english"),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    for cfg in configs:
        vect, X = build_tfidf_vectors(texts, cfg)
        stats = sparse_matrix_stats(X)
        survivors = case_sensitive_stopword_survivors(vect) if cfg.stop_words else 0
        norms = l2_norm_sample_stats(X, sample_n=sample_n_for_norms, seed=0)

        results[cfg.name] = {
            "config": cfg,
            "vectorizer": vect,
            "X": X,
            "stats": stats,
            "l2_norm_sample": norms,
            "case_sensitive_stopword_survivors": survivors,
        }

    return results
