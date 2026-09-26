#!/usr/bin/env python3
"""
entity_resolution.py
====================

End-to-end solution for the Amazon ML **Business Entity Resolution Challenge**.

For every Source-1 entity, find all matching Source-2 / Source-3 records. The
metric is a precision-heavy macro-F0.5, so the pipeline is tuned to avoid false
merges (a wrong match hurts ~2x more than a miss).

Pipeline
--------
  1. BLOCKING (candidate generation), done per country:
       * char n-gram TF-IDF cosine  (top-K)         -> lexical recall
       * Qwen3-Embedding cosine     (top-K)         -> semantic recall
       * exact (country, pincode) join              -> structured recall
     Union of the three = the candidate set (written to candidate_pairs.tsv).

  2. FEATURES per (S1, candidate) pair:
       TF-IDF cosines, embedding cosine, RapidFuzz name/address ratios,
       pincode / country agreement, length ratios, and (optionally) a
       Qwen3-Reranker relevance score -- the strongest precision signal.

  3. MATCHER: a LightGBM classifier trained on the ground truth (candidate
     pairs labelled match / no-match), then a global decision threshold tuned
     on a held-out split of Source-1 entities to maximise macro-F0.5.

  4. OUTPUT: output/matching_results.tsv (scored) and output/candidate_pairs.tsv
     in the exact challenge format, followed by a self-validation pass.

Models (challenge rule: Apache-2.0 / MIT, <= 8B params) are loaded lazily and
degrade gracefully -- see er_models.py. Pass --emb-model hashing / --no-reranker
to run a fast, dependency-light version anywhere.

Usage
-----
    python scripts/entity_resolution.py \
        --processed-dir data/processed \
        --ground-truth data/train_ground_truth.tsv \
        --output-dir output
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from er_models import EmbeddingModel, Reranker

# RapidFuzz is optional; fall back to difflib if it is missing.
try:
    from rapidfuzz import fuzz as _fuzz

    def _ratio(kind, a, b):
        if kind == "token_set":
            return _fuzz.token_set_ratio(a, b) / 100.0
        if kind == "token_sort":
            return _fuzz.token_sort_ratio(a, b) / 100.0
        if kind == "partial":
            return _fuzz.partial_ratio(a, b) / 100.0
        return _fuzz.WRatio(a, b) / 100.0

    _HAVE_RAPIDFUZZ = True
except Exception:  # noqa: BLE001
    import difflib

    def _ratio(kind, a, b):
        return difflib.SequenceMatcher(None, a, b).ratio()

    _HAVE_RAPIDFUZZ = False


REQUIRED_COLS = [
    "entity_id", "clean_name", "clean_address", "extracted_pincode", "normalized_country",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def f_beta(pred: set[str], true: set[str], beta: float = 0.5) -> float:
    """F-beta for one Source-1 entity (empty/empty == 1.0, per the PS)."""
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def macro_f_beta(pred_map: dict[str, set[str]], true_map: dict[str, set[str]],
                 entities: list[str], beta: float = 0.5) -> float:
    """Macro-average of F-beta over the given Source-1 entities."""
    if not entities:
        return 0.0
    total = sum(
        f_beta(pred_map.get(e, set()), true_map.get(e, set()), beta) for e in entities
    )
    return total / len(entities)


def macro_scores(pred_map: dict[str, set[str]], true_map: dict[str, set[str]],
                 entities: list[str], beta: float = 0.5) -> dict[str, float]:
    """Macro-averaged precision / recall / F-beta over Source-1 entities.

    Uses the challenge's empty-set convention (empty/empty == perfect for that
    entity, any mismatch involving an empty side == 0)."""
    if not entities:
        return {"precision": 0.0, "recall": 0.0, "f_beta": 0.0}
    ps = rs = fs = 0.0
    for e in entities:
        pred, true = pred_map.get(e, set()), true_map.get(e, set())
        if not pred and not true:
            pr = rc = 1.0
        elif not pred or not true:
            pr = rc = 0.0
        else:
            tp = len(pred & true)
            pr = tp / len(pred)
            rc = tp / len(true)
        ps += pr; rs += rc; fs += f_beta(pred, true, beta)
    n = len(entities)
    return {"precision": ps / n, "recall": rs / n, "f_beta": fs / n}


def pairwise_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Micro (pair-level) precision / recall / accuracy of the classifier."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    total = max(tp + fp + fn + tn, 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": prec, "recall": rec, "accuracy": (tp + tn) / total,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_source(processed_dir: Path, split: str, idx: int) -> pd.DataFrame:
    """Load one cleaned source parquet and normalize the columns we need."""
    path = processed_dir / f"clean_{split}_source{idx}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run preprocess.py first.")
    df = pd.read_parquet(path)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[REQUIRED_COLS].copy()
    for col in REQUIRED_COLS:
        df[col] = df[col].fillna("").astype(str)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the matching fields (name/addr/combined/pincode/country/rr_text)."""
    out = pd.DataFrame({"entity_id": df["entity_id"].to_numpy()})
    out["name"] = df["clean_name"].str.strip()
    out["addr"] = df["clean_address"].str.strip()
    out["combined"] = (out["name"] + " " + out["addr"]).str.strip()
    out["pincode"] = df["extracted_pincode"].str.strip()
    out["country"] = df["normalized_country"].str.strip()
    out["rr_text"] = out["name"] + " | " + out["addr"] + " | " + out["country"]
    return out


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    """Parse train_ground_truth.tsv -> {source1_id: {matched ids}}."""
    gt: dict[str, set[str]] = {}
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    id_col = "source1_entity_id" if "source1_entity_id" in df.columns else df.columns[0]
    match_col = "matched_entity_ids" if "matched_entity_ids" in df.columns else df.columns[1]
    for s1, matches in zip(df[id_col], df[match_col]):
        ids = {m.strip() for m in str(matches).split(",") if m.strip()}
        gt[str(s1).strip()] = ids
    return gt


# ---------------------------------------------------------------------------
# Blocking helpers
# ---------------------------------------------------------------------------

def _country_groups(countries: np.ndarray) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(countries):
        groups[c].append(i)
    return {c: np.asarray(ix, dtype=np.int64) for c, ix in groups.items()}


def blocked_topk(q_ids, Q, q_ctry, t_ids, T, t_ctry, k, q_chunk=256, t_chunk=50_000):
    """
    Top-k most similar targets for each query, restricted to the same country.

    Q, T may be scipy-sparse (TF-IDF) or dense numpy (embeddings); both assumed
    L2-normalized so a dot product is cosine similarity. Returns
    {q_id: set(t_id)} keeping only strictly-positive similarities.

    Memory-bounded: similarities are computed in ``q_chunk x t_chunk`` blocks
    with a running top-k, so the full (queries x targets) matrix is NEVER
    materialized. Peak temporary memory ~= q_chunk * t_chunk floats regardless
    of how many records a country has (this is what avoids the Kaggle OOM).
    """
    is_sparse = sparse.issparse(Q)
    result: dict[str, set[str]] = defaultdict(set)
    t_groups = _country_groups(t_ctry)
    q_groups = _country_groups(q_ctry)

    for country, q_idx in q_groups.items():
        t_idx = t_groups.get(country)
        if t_idx is None or len(t_idx) == 0:
            continue
        T_c = T[t_idx]
        n_t = len(t_idx)
        kk = int(min(k, n_t))
        for qs in range(0, len(q_idx), q_chunk):
            qb = q_idx[qs:qs + q_chunk]
            Qb = Q[qb]
            B = len(qb)
            best_val = np.full((B, kk), -1.0, dtype=np.float32)
            best_col = np.full((B, kk), -1, dtype=np.int64)
            for ts in range(0, n_t, t_chunk):
                te = min(ts + t_chunk, n_t)
                block = Qb @ T_c[ts:te].T                       # small B x (te-ts)
                block = block.toarray() if is_sparse else np.asarray(block)
                block = block.astype(np.float32, copy=False)
                # merge this block with the running top-k and re-select top-k
                cand_val = np.concatenate([best_val, block], axis=1)
                idx_block = np.broadcast_to(np.arange(ts, te), (B, te - ts))
                cand_col = np.concatenate([best_col, idx_block], axis=1)
                kk2 = min(kk, cand_val.shape[1])
                part = np.argpartition(-cand_val, kk2 - 1, axis=1)[:, :kk2]
                rows = np.arange(B)[:, None]
                best_val = cand_val[rows, part]
                best_col = cand_col[rows, part]
            for r in range(B):
                qid = q_ids[qb[r]]
                for j in range(kk):
                    col = best_col[r, j]
                    if col >= 0 and best_val[r, j] > 0.0:
                        result[qid].add(t_ids[t_idx[int(col)]])
    return result


def sparse_topk_blocking(q_ids, Qs, q_ctry, t_ids, Ts, t_ctry, k,
                         q_chunk=1000, tag=""):
    """
    Scalable per-country top-k blocking via SPARSE cosine (inverted-index join).

    Qs, Ts are L2-normalized *sparse* word-level (df-capped) TF-IDF matrices, so
    ``Qs[block] @ Ts.T`` is sparse: only records sharing a distinctive token are
    ever compared. No all-pairs loop and no dense (queries x targets) matrix, so
    this stays tractable at 10M+ records where brute-force blocking cannot.
    Emits progress so the run never looks "stuck".
    """
    result: dict[str, set[str]] = defaultdict(set)
    t_ids = np.asarray(t_ids)
    t_groups = _country_groups(t_ctry)
    q_groups = _country_groups(q_ctry)
    total_q = sum(len(v) for v in q_groups.values())
    done, next_report, t0 = 0, 0, time.time()

    for country, q_idx in q_groups.items():
        t_idx = t_groups.get(country)
        if t_idx is None or len(t_idx) == 0:
            done += len(q_idx)
            continue
        Tt = Ts[t_idx].T.tocsr()                 # V x n_t (sparse)
        tid_c = t_ids[t_idx]
        for s in range(0, len(q_idx), q_chunk):
            qb = q_idx[s:s + q_chunk]
            sims = (Qs[qb] @ Tt).tocsr()         # (B x n_t) sparse
            indptr, data, indices = sims.indptr, sims.data, sims.indices
            for r in range(len(qb)):
                a, b = indptr[r], indptr[r + 1]
                if b == a:
                    continue
                d = data[a:b]
                cols = indices[a:b]
                top = np.argpartition(-d, k - 1)[:k] if d.size > k else np.arange(d.size)
                add = result[q_ids[qb[r]]].add
                for j in top:
                    add(tid_c[cols[j]])
            done += len(qb)
            if tag and done >= next_report:
                rate = done / max(time.time() - t0, 1e-6)
                log(f"    [{tag}] blocked {done:,}/{total_q:,} "
                    f"({100.0 * done / max(total_q, 1):.0f}%)  ~{rate:,.0f} q/s")
                next_report += 100_000
    return result


def pincode_candidates(q, t) -> dict[str, set[str]]:
    """Exact (country, pincode) matches -- a strong structured blocking key."""
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for tid, ctry, pin in zip(t["entity_id"], t["country"], t["pincode"]):
        if pin and len(pin) >= 5:
            buckets[(ctry, pin)].append(tid)
    out: dict[str, set[str]] = defaultdict(set)
    for qid, ctry, pin in zip(q["entity_id"], q["country"], q["pincode"]):
        if pin and len(pin) >= 5:
            hits = buckets.get((ctry, pin))
            if hits:
                out[qid].update(hits)
    return out


def merge_candidates(*dicts) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for d in dicts:
        for qid, tids in d.items():
            merged[qid].update(tids)
    return merged


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _row_cos_sparse(A, B) -> np.ndarray:
    """Per-row cosine of two aligned, L2-normalized sparse matrices."""
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


def build_pairs_and_features(
    s1: pd.DataFrame,
    tgt: pd.DataFrame,
    candidates: dict[str, set[str]],
    name_vec: TfidfVectorizer,
    comb_vec: TfidfVectorizer,
    s1_name_tfidf, tgt_name_tfidf,
    s1_comb_tfidf, tgt_comb_tfidf,
    s1_emb, tgt_emb,
    reranker: Reranker | None,
    use_emb: bool,
):
    """Flatten candidates to pairs and compute the feature matrix."""
    s1_pos = {eid: i for i, eid in enumerate(s1["entity_id"].to_numpy())}
    tgt_pos = {eid: i for i, eid in enumerate(tgt["entity_id"].to_numpy())}

    q_id, t_id, qi, ti = [], [], [], []
    for s1_id, cand in candidates.items():
        if s1_id not in s1_pos:
            continue
        for c in cand:
            if c not in tgt_pos:
                continue
            q_id.append(s1_id); t_id.append(c)
            qi.append(s1_pos[s1_id]); ti.append(tgt_pos[c])
    qi = np.asarray(qi, dtype=np.int64)
    ti = np.asarray(ti, dtype=np.int64)

    feats: dict[str, np.ndarray] = {}
    if len(qi) == 0:
        empty = pd.DataFrame()
        return [], [], empty

    feats["name_tfidf_cos"] = _row_cos_sparse(s1_name_tfidf[qi], tgt_name_tfidf[ti])
    feats["comb_tfidf_cos"] = _row_cos_sparse(s1_comb_tfidf[qi], tgt_comb_tfidf[ti])
    if use_emb and s1_emb is not None and tgt_emb is not None:
        feats["emb_cos"] = np.sum(s1_emb[qi] * tgt_emb[ti], axis=1)

    qn = s1["name"].to_numpy(); tn = tgt["name"].to_numpy()
    qa = s1["addr"].to_numpy(); ta = tgt["addr"].to_numpy()
    qp = s1["pincode"].to_numpy(); tp = tgt["pincode"].to_numpy()

    feats["name_token_set"] = np.array([_ratio("token_set", qn[a], tn[b]) for a, b in zip(qi, ti)])
    feats["name_token_sort"] = np.array([_ratio("token_sort", qn[a], tn[b]) for a, b in zip(qi, ti)])
    feats["name_partial"] = np.array([_ratio("partial", qn[a], tn[b]) for a, b in zip(qi, ti)])
    feats["addr_token_set"] = np.array([_ratio("token_set", qa[a], ta[b]) for a, b in zip(qi, ti)])
    feats["addr_partial"] = np.array([_ratio("partial", qa[a], ta[b]) for a, b in zip(qi, ti)])

    # Structured agreement + shape features.
    q_pin = qp[qi]; t_pin = tp[ti]
    both_pin = (np.char.str_len(q_pin.astype(str)) >= 5) & (np.char.str_len(t_pin.astype(str)) >= 5)
    feats["pincode_exact"] = ((q_pin == t_pin) & both_pin).astype(np.float32)
    feats["pincode_both"] = both_pin.astype(np.float32)

    qn_len = np.char.str_len(qn[qi].astype(str)).astype(np.float32)
    tn_len = np.char.str_len(tn[ti].astype(str)).astype(np.float32)
    feats["name_len_ratio"] = np.minimum(qn_len, tn_len) / np.maximum(np.maximum(qn_len, tn_len), 1.0)

    # Name token Jaccard.
    def _jacc(a, b):
        sa, sb = set(a.split()), set(b.split())
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    feats["name_jaccard"] = np.array([_jacc(qn[a], tn[b]) for a, b in zip(qi, ti)])

    # Optional cross-encoder reranker score -- strongest precision feature.
    if reranker is not None and reranker.available:
        qrr = s1["rr_text"].to_numpy(); trr = tgt["rr_text"].to_numpy()
        pairs = [(qrr[a], trr[b]) for a, b in zip(qi, ti)]
        log(f"  reranking {len(pairs):,} pairs ...")
        scores = reranker.score(pairs)
        feats["reranker_score"] = np.array([s if s is not None else 0.0 for s in scores],
                                           dtype=np.float32)

    features = pd.DataFrame(feats).astype(np.float32)
    return q_id, t_id, features


# ---------------------------------------------------------------------------
# Candidate generation for one split (train or test)
# ---------------------------------------------------------------------------

def _cached_embed(emb_model, texts, ids, cache_dir: Path, tag: str,
                  is_query: bool, use_cache: bool) -> np.ndarray:
    """Encode texts, caching to disk and reusing only if the id-list matches."""
    npy = cache_dir / f"emb_{tag}.npy"
    idp = cache_dir / f"emb_{tag}.ids.txt"
    ids = [str(i) for i in ids]
    if use_cache and npy.exists() and idp.exists():
        if idp.read_text().splitlines() == ids:
            log(f"  [ckpt] reuse embeddings {npy.name}")
            return np.load(npy)
        log(f"  [ckpt] {npy.name} stale (ids changed) -> re-encoding")
    emb = emb_model.encode(list(texts), is_query=is_query)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy, emb)
        idp.write_text("\n".join(ids))
        log(f"  [ckpt] saved embeddings {npy.name} shape={emb.shape}")
    return emb


def _save_candidates(cache_dir: Path, split: str, candidates: dict[str, set[str]]) -> None:
    """Persist the candidate set for resume / blocking analysis."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = {"source1_entity_id": list(candidates.keys()),
            "candidate_entity_ids": [",".join(sorted(v)) for v in candidates.values()]}
    pd.DataFrame(rows).to_parquet(cache_dir / f"candidates_{split}.parquet")


def generate_split(processed_dir: Path, split: str, emb_model: EmbeddingModel,
                   args, s1_filter: set[str] | None = None) -> dict:
    """Load a split, fit TF-IDF, embed, and build the candidate set + vectors.

    If ``s1_filter`` is given, only those Source-1 entities are processed as
    queries (used for Kaggle-style batching); the S2/S3 target pool is unchanged.
    """
    log(f"[{split}] loading sources ...")
    s1 = prepare(load_source(processed_dir, split, 1))
    s2 = prepare(load_source(processed_dir, split, 2))
    s3 = prepare(load_source(processed_dir, split, 3))
    tgt = pd.concat([s2, s3], ignore_index=True)
    if s1_filter is not None:
        s1 = s1[s1["entity_id"].isin(s1_filter)].reset_index(drop=True)
    log(f"[{split}] S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}  targets={len(tgt):,}")

    # Word-level, df-capped TF-IDF -> sparse & distinctive, so blocking is an
    # inverted-index join (only shared-token records compared). max_df drops
    # ultra-common words (e.g. "road") so posting lists stay short; min_df drops
    # hapax/typo-unique tokens. char-level typos are handled later by RapidFuzz.
    name_vec = TfidfVectorizer(min_df=args.min_df, max_df=args.max_df, sublinear_tf=True)
    comb_vec = TfidfVectorizer(min_df=args.min_df, max_df=args.max_df, sublinear_tf=True)
    name_vec.fit(pd.concat([s1["name"], tgt["name"]], ignore_index=True))
    comb_vec.fit(pd.concat([s1["combined"], tgt["combined"]], ignore_index=True))

    s1_name_tfidf = name_vec.transform(s1["name"]).tocsr()
    tgt_name_tfidf = name_vec.transform(tgt["name"]).tocsr()
    s1_comb_tfidf = comb_vec.transform(s1["combined"]).tocsr()
    tgt_comb_tfidf = comb_vec.transform(tgt["combined"]).tocsr()

    s1_emb = tgt_emb = None
    if args.use_emb:
        log(f"[{split}] embedding {len(s1):,} queries + {len(tgt):,} targets ...")
        cache_dir = Path(args.artifacts_dir)
        use_cache = not args.no_cache
        # Targets (S2+S3) are embedded once and reused across every shard; the
        # per-shard S1 slice is small, so it is embedded fresh (not cached).
        s1_cache = use_cache and s1_filter is None
        s1_emb = _cached_embed(emb_model, s1["combined"].tolist(), s1["entity_id"].tolist(),
                               cache_dir, f"{split}_s1", True, s1_cache)
        tgt_emb = _cached_embed(emb_model, tgt["combined"].tolist(), tgt["entity_id"].tolist(),
                                cache_dir, f"{split}_tgt", False, use_cache)

    log(f"[{split}] blocking (word-tfidf top-k={args.k_tfidf}, df_cap={args.max_df}) ...")
    s1_ids_a = s1["entity_id"].to_numpy(); s1_ctry_a = s1["country"].to_numpy()
    tgt_ids_a = tgt["entity_id"].to_numpy(); tgt_ctry_a = tgt["country"].to_numpy()
    cand_name = sparse_topk_blocking(s1_ids_a, s1_name_tfidf, s1_ctry_a,
                                     tgt_ids_a, tgt_name_tfidf, tgt_ctry_a,
                                     args.k_tfidf, tag=f"{split}:name")
    cand_comb = sparse_topk_blocking(s1_ids_a, s1_comb_tfidf, s1_ctry_a,
                                     tgt_ids_a, tgt_comb_tfidf, tgt_ctry_a,
                                     args.k_tfidf, tag=f"{split}:comb")
    parts = [cand_name, cand_comb, pincode_candidates(s1, tgt)]
    if args.use_emb:
        # Dense embedding blocking (only viable on GPU / small data); off by
        # default. Uses the memory-bounded blocked_topk.
        cand_emb = blocked_topk(s1_ids_a, s1_emb, s1_ctry_a,
                                tgt_ids_a, tgt_emb, tgt_ctry_a, args.k_emb)
        parts.append(cand_emb)
    candidates = merge_candidates(*parts)

    n_pairs = sum(len(v) for v in candidates.values())
    log(f"[{split}] candidates: {n_pairs:,} pairs over {len(candidates):,} S1 entities "
        f"(avg {n_pairs / max(len(s1), 1):.1f}/entity)")
    if not args.no_cache and s1_filter is None:
        _save_candidates(Path(args.artifacts_dir), split, candidates)

    return dict(
        s1=s1, tgt=tgt, candidates=candidates,
        name_vec=name_vec, comb_vec=comb_vec,
        s1_name_tfidf=s1_name_tfidf, tgt_name_tfidf=tgt_name_tfidf,
        s1_comb_tfidf=s1_comb_tfidf, tgt_comb_tfidf=tgt_comb_tfidf,
        s1_emb=s1_emb, tgt_emb=tgt_emb,
    )


# ---------------------------------------------------------------------------
# Submission I/O + validation
# ---------------------------------------------------------------------------

def write_submission(path: Path, s1_ids: list[str], id_lists: dict[str, list[str]],
                     header_col: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{header_col}\n")
        for s1 in s1_ids:
            ids = id_lists.get(s1, [])
            fh.write(f"{s1}\t{','.join(ids)}\n")
    log(f"  wrote {path} ({len(s1_ids):,} rows)")


def validate_submission(matching: dict[str, list[str]], candidates: dict[str, list[str]],
                        test_s1: list[str], valid_targets: set[str]) -> bool:
    """Replicate the challenge's key format rules; return True if all pass."""
    issues: list[str] = []
    if set(matching) != set(test_s1):
        issues.append(f"matching_results S1 set != test S1 set "
                      f"(missing {len(set(test_s1) - set(matching))}, "
                      f"extra {len(set(matching) - set(test_s1))})")
    for s1, ids in matching.items():
        if len(ids) != len(set(ids)):
            issues.append(f"{s1}: duplicate IDs in matched list")
        for i in ids:
            if not (i.startswith("S2-") or i.startswith("S3-")):
                issues.append(f"{s1}: non-S2/S3 id {i}"); break
            if i not in valid_targets:
                issues.append(f"{s1}: id {i} not in test set"); break
        cand_set = set(candidates.get(s1, []))
        if not set(ids).issubset(cand_set):
            issues.append(f"{s1}: matched IDs not a subset of candidates")
    if issues:
        log(f"[VALIDATE] {len(issues)} issue(s):")
        for msg in issues[:20]:
            log(f"   - {msg}")
        return False
    log("[VALIDATE] PASS - submission format looks correct.")
    return True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def tune_threshold(prob_by_s1: dict[str, list[tuple[str, float]]],
                   true_map: dict[str, set[str]], val_ids: list[str]) -> tuple[float, float]:
    """Grid-search the decision threshold that maximises macro-F0.5 on val."""
    best_thr, best_f = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        pred_map = {
            s1: {tid for tid, p in prob_by_s1.get(s1, []) if p >= thr} for s1 in val_ids
        }
        f = macro_f_beta(pred_map, true_map, val_ids)
        if f > best_f:
            best_f, best_thr = f, float(thr)
    return best_thr, best_f


def train_matcher(train_data: dict, gt: dict[str, set[str]], args):
    """Label candidate pairs, train LightGBM, and tune the F0.5 threshold."""
    import lightgbm as lgb

    s1 = train_data["s1"]
    reranker = _maybe_reranker(args)
    q_id, t_id, X = build_pairs_and_features(
        train_data["s1"], train_data["tgt"], train_data["candidates"],
        train_data["name_vec"], train_data["comb_vec"],
        train_data["s1_name_tfidf"], train_data["tgt_name_tfidf"],
        train_data["s1_comb_tfidf"], train_data["tgt_comb_tfidf"],
        train_data["s1_emb"], train_data["tgt_emb"],
        reranker, args.use_emb,
    )
    if len(q_id) == 0:
        raise RuntimeError("No training candidate pairs were generated.")

    y = np.array([1 if t in gt.get(q, set()) else 0 for q, t in zip(q_id, t_id)], dtype=np.int8)
    log(f"[train] pairs={len(y):,}  positives={int(y.sum()):,}  features={list(X.columns)}")

    # Blocking recall ceiling (diagnostic).
    covered = defaultdict(set)
    for q, t in zip(q_id, t_id):
        if t in gt.get(q, set()):
            covered[q].add(t)
    total_true = sum(len(v) for v in gt.values())
    found_true = sum(len(v) for v in covered.values())
    log(f"[train] blocking recall ceiling: {found_true:,}/{total_true:,} "
        f"({100.0 * found_true / max(total_true, 1):.1f}%)")

    # Split by S1 entity so val entities are unseen during training.
    rng = random.Random(args.seed)
    train_s1 = sorted(set(q_id))
    rng.shuffle(train_s1)
    n_val = max(1, int(len(train_s1) * args.val_frac))
    val_set = set(train_s1[:n_val])
    is_val = np.array([q in val_set for q in q_id])

    clf = lgb.LGBMClassifier(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate,
        num_leaves=args.num_leaves, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=args.seed, n_jobs=-1, verbosity=-1,
    )
    clf.fit(X[~is_val], y[~is_val])

    # Tune threshold on the validation split (all val S1 entities, incl. singletons).
    val_prob = clf.predict_proba(X[is_val])[:, 1]
    prob_by_s1: dict[str, list[tuple[str, float]]] = defaultdict(list)
    val_q = [q for q, v in zip(q_id, is_val) if v]
    val_t = [t for t, v in zip(t_id, is_val) if v]
    for q, t, p in zip(val_q, val_t, val_prob):
        prob_by_s1[q].append((t, float(p)))
    val_entities = sorted(val_set)
    thr, val_f = tune_threshold(prob_by_s1, gt, val_entities)

    # Entity-level (macro) precision / recall / F0.5 at the tuned threshold --
    # this is exactly how the leaderboard scores you.
    pred_map = {s1: {tid for tid, p in prob_by_s1.get(s1, []) if p >= thr}
                for s1 in val_entities}
    ent = macro_scores(pred_map, gt, val_entities)
    # Pair-level (micro) classifier metrics at the tuned threshold.
    y_val = y[is_val]
    pair = pairwise_scores(y_val, (val_prob >= thr).astype(int))

    log(f"[train] tuned threshold = {thr:.2f}")
    log(f"[train] VAL entity-macro : precision={ent['precision']:.4f}  "
        f"recall={ent['recall']:.4f}  F0.5={ent['f_beta']:.4f}")
    log(f"[train] VAL pair-level   : precision={pair['precision']:.4f}  "
        f"recall={pair['recall']:.4f}  accuracy={pair['accuracy']:.4f}  "
        f"(tp={pair['tp']} fp={pair['fp']} fn={pair['fn']})")
    try:
        imp = sorted(zip(X.columns, clf.feature_importances_), key=lambda t: -t[1])
        log("[train] top features   : " + ", ".join(f"{k}={int(v)}" for k, v in imp[:6]))
    except Exception:  # noqa: BLE001
        pass

    # --- Checkpoint: persist matcher + threshold + metrics -------------------
    if not args.no_cache:
        art = Path(args.artifacts_dir)
        art.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, art / "matcher_lgbm.joblib")
        (art / "matcher_meta.json").write_text(json.dumps({
            "threshold": thr,
            "feature_cols": list(X.columns),
            "val_entity_macro": ent,
            "val_pairwise": {k: pair[k] for k in ("precision", "recall", "accuracy",
                                                  "tp", "fp", "fn", "tn")},
            "n_pairs": int(len(y)), "n_positives": int(y.sum()),
        }, indent=2))
        log(f"[ckpt] saved matcher -> {art / 'matcher_lgbm.joblib'}")

    return clf, thr, list(X.columns)


def predict_split(test_data: dict, clf, threshold: float, feature_cols: list[str], args):
    """Score test candidates and apply the tuned threshold."""
    reranker = _maybe_reranker(args)
    q_id, t_id, X = build_pairs_and_features(
        test_data["s1"], test_data["tgt"], test_data["candidates"],
        test_data["name_vec"], test_data["comb_vec"],
        test_data["s1_name_tfidf"], test_data["tgt_name_tfidf"],
        test_data["s1_comb_tfidf"], test_data["tgt_comb_tfidf"],
        test_data["s1_emb"], test_data["tgt_emb"],
        reranker, args.use_emb,
    )
    matches: dict[str, list[str]] = defaultdict(list)
    if len(q_id) > 0:
        X = X.reindex(columns=feature_cols, fill_value=0.0)  # align to training columns
        prob = clf.predict_proba(X)[:, 1]
        for q, t, p in zip(q_id, t_id, prob):
            if p >= threshold:
                matches[q].append(t)
    return {q: sorted(set(v)) for q, v in matches.items()}


def _maybe_reranker(args) -> Reranker | None:
    if getattr(args, "_reranker_cache", None) is not None:
        return args._reranker_cache
    rr = None
    if not args.no_reranker:
        rr = Reranker(model_name=args.rerank_model, batch_size=args.rerank_batch)
    args._reranker_cache = rr
    return rr


def _read_ids(path: Path) -> list[str]:
    return pd.read_parquet(path, columns=["entity_id"])["entity_id"].astype(str).tolist()


def _ensure_all(df: pd.DataFrame, all_ids: list[str], col: str) -> pd.DataFrame:
    """Keep only known S1 ids, add missing ones as empty rows, order by all_ids."""
    have = set(df["source1_entity_id"])
    extra = [{"source1_entity_id": s, col: ""} for s in all_ids if s not in have]
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    order = {s: i for i, s in enumerate(all_ids)}
    df = df[df["source1_entity_id"].isin(order)].copy()
    df["__o"] = df["source1_entity_id"].map(order)
    return df.sort_values("__o")[["source1_entity_id", col]].reset_index(drop=True)


def merge_parts(output_dir: Path, processed_dir: Path) -> int:
    """Concatenate per-shard part files into the final submission TSVs."""
    parts = output_dir / "parts"
    m_files = sorted(parts.glob("matching_part_*.tsv"))
    c_files = sorted(parts.glob("candidate_part_*.tsv"))
    if not m_files:
        log(f"[merge] no matching_part_*.tsv in {parts}"); return 1
    log(f"[merge] combining {len(m_files)} matching + {len(c_files)} candidate part(s)")

    def _combine(files, col):
        if not files:
            return pd.DataFrame(columns=["source1_entity_id", col])
        df = pd.concat([pd.read_csv(f, sep="\t", dtype=str) for f in files], ignore_index=True)
        return df.fillna("").drop_duplicates("source1_entity_id")

    test_s1 = _read_ids(processed_dir / "clean_test_source1.parquet")
    m = _ensure_all(_combine(m_files, "matched_entity_ids"), test_s1, "matched_entity_ids")
    c = _ensure_all(_combine(c_files, "candidate_entity_ids"), test_s1, "candidate_entity_ids")

    output_dir.mkdir(parents=True, exist_ok=True)
    m.to_csv(output_dir / "matching_results.tsv", sep="\t", index=False)
    c.to_csv(output_dir / "candidate_pairs.tsv", sep="\t", index=False)
    log(f"[merge] wrote final outputs ({len(m):,} rows)")

    matching = {r["source1_entity_id"]: [x for x in r["matched_entity_ids"].split(",") if x]
                for _, r in m.iterrows()}
    cand = {r["source1_entity_id"]: [x for x in r["candidate_entity_ids"].split(",") if x]
            for _, r in c.iterrows()}
    valid_targets = (set(_read_ids(processed_dir / "clean_test_source2.parquet"))
                     | set(_read_ids(processed_dir / "clean_test_source3.parquet")))
    validate_submission(matching, cand, test_s1, valid_targets)
    return 0


def main() -> int:
    args = parse_args()
    args.use_emb = not args.no_embeddings
    args._reranker_cache = None
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    art = Path(args.artifacts_dir)
    model_path, meta_path = art / "matcher_lgbm.joblib", art / "matcher_meta.json"

    log("Amazon ML - Business Entity Resolution :: matching pipeline")
    log(f"stage={args.stage} shards={args.num_shards} shard_id={args.shard_id} "
        f"emb={'(off)' if not args.use_emb else args.emb_model} "
        f"rerank={'(off)' if args.no_reranker else args.rerank_model}")

    # ---------------- MERGE ----------------
    if args.stage == "merge":
        return merge_parts(output_dir, processed_dir)

    emb_model = EmbeddingModel(model_name=args.emb_model) if args.use_emb else None

    # ---------------- TRAIN (or resume from cached matcher) ----------------
    if args.stage in ("all", "train"):
        if args.use_cached_model and model_path.exists() and meta_path.exists():
            clf = joblib.load(model_path)
            meta = json.loads(meta_path.read_text())
            threshold = float(meta["threshold"]); feature_cols = meta["feature_cols"]
            log("[resume] loaded cached matcher; skipping training.")
        else:
            gt = load_ground_truth(Path(args.ground_truth))
            log(f"ground truth: {len(gt):,} S1 entities, "
                f"{sum(len(v) for v in gt.values()):,} match links")
            train_data = generate_split(processed_dir, "train", emb_model, args)
            clf, threshold, feature_cols = train_matcher(train_data, gt, args)
        if args.stage == "train":
            log("[train] complete -- model saved. Next: --stage predict")
            return 0
    else:  # --stage predict: matcher must already be trained + cached
        if not (model_path.exists() and meta_path.exists()):
            log(f"[ERROR] no cached matcher in {art}. Run '--stage train' first.")
            return 1
        clf = joblib.load(model_path)
        meta = json.loads(meta_path.read_text())
        threshold = float(meta["threshold"]); feature_cols = meta["feature_cols"]
        log(f"[predict] loaded matcher (threshold={threshold:.2f})")

    # ---------------- PREDICT (optionally sharded) ----------------
    test_s1_all = sorted(_read_ids(processed_dir / "clean_test_source1.parquet"))
    sharded = args.num_shards > 1
    if sharded:
        if not (0 <= args.shard_id < args.num_shards):
            log(f"[ERROR] --shard-id must be in [0, {args.num_shards})."); return 1
        shard_ids = list(np.array_split(np.array(test_s1_all), args.num_shards)[args.shard_id])
        parts_dir = output_dir / "parts"
        m_part = parts_dir / f"matching_part_{args.shard_id:04d}.tsv"
        c_part = parts_dir / f"candidate_part_{args.shard_id:04d}.tsv"
        if m_part.exists() and c_part.exists() and not args.force:
            log(f"[predict] shard {args.shard_id} already done -> skip (--force to redo).")
            return 0
        log(f"[predict] shard {args.shard_id + 1}/{args.num_shards}: "
            f"{len(shard_ids):,} of {len(test_s1_all):,} S1 entities")
        s1_filter: set[str] | None = set(shard_ids)
    else:
        shard_ids, s1_filter = test_s1_all, None

    test_data = generate_split(processed_dir, "test", emb_model, args, s1_filter=s1_filter)
    s1_ids = list(test_data["s1"]["entity_id"].to_numpy())
    candidates_out = {q: sorted(v) for q, v in test_data["candidates"].items()}
    matches_out = predict_split(test_data, clf, threshold, feature_cols, args)
    candidates_out = {s1: candidates_out.get(s1, []) for s1 in s1_ids}
    matches_out = {s1: matches_out.get(s1, []) for s1 in s1_ids}

    if sharded:
        parts_dir.mkdir(parents=True, exist_ok=True)
        write_submission(c_part, s1_ids, candidates_out, "candidate_entity_ids")
        write_submission(m_part, s1_ids, matches_out, "matched_entity_ids")
        log(f"[predict] shard {args.shard_id} done. After all shards run: --stage merge")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_submission(output_dir / "candidate_pairs.tsv", s1_ids, candidates_out, "candidate_entity_ids")
    write_submission(output_dir / "matching_results.tsv", s1_ids, matches_out, "matched_entity_ids")
    valid_targets = set(test_data["tgt"]["entity_id"].to_numpy())
    validate_submission(matches_out, candidates_out, s1_ids, valid_targets)
    n_matched = sum(1 for s1 in s1_ids if matches_out.get(s1))
    log(f"[done] {n_matched:,}/{len(s1_ids):,} test S1 entities matched "
        f"(threshold={threshold:.2f})")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entity Resolution matching pipeline.")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--ground-truth", default="data/train_ground_truth.tsv")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--artifacts-dir", default="artifacts/er",
                   help="Checkpoint dir: cached embeddings, candidates, trained matcher.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable all checkpointing (embeddings / candidates / model).")
    p.add_argument("--use-cached-model", action="store_true",
                   help="Load a previously trained matcher and skip training.")
    p.add_argument("--stage", choices=["all", "train", "predict", "merge"], default="all",
                   help="all = train+predict; split into stages for Kaggle-style batching.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split test Source-1 entities into N batches for --stage predict.")
    p.add_argument("--shard-id", type=int, default=0,
                   help="Which 0-based batch to run (used with --num-shards).")
    p.add_argument("--force", action="store_true",
                   help="Recompute a shard even if its part files already exist.")
    p.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B",
                   help="HF model id, or 'hashing' to force the CPU fallback.")
    p.add_argument("--rerank-model", default="Qwen/Qwen3-Reranker-8B")
    p.add_argument("--no-embeddings", action="store_true", help="Skip the embedding stage.")
    p.add_argument("--no-reranker", action="store_true", help="Skip the reranker feature.")
    p.add_argument("--k-tfidf", type=int, default=20,
                   help="Top-k candidates per blocking pass (smaller = leaner candidate set).")
    p.add_argument("--k-emb", type=int, default=50)
    p.add_argument("--max-df", type=int, default=50_000,
                   help="Drop tokens appearing in more than this many documents "
                        "(keeps blocking posting lists short; absolute count).")
    p.add_argument("--rerank-batch", type=int, default=16)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())
