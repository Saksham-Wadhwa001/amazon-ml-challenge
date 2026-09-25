#!/usr/bin/env python3
"""
preprocess.py
=============

Production-grade, highly vectorized preprocessing pipeline for the
**Amazon ML Business Entity Resolution Challenge**, built on Polars.

The script ingests tab-separated (``.tsv``) source files with the columns::

    entity_id    business_name    business_address    country

and produces, for each source, a single unified Parquet file containing the
**original columns untouched** plus four appended, cleaned columns:

    clean_name          normalized business name
    clean_address       normalized business address (landmarks/states handled)
    extracted_pincode   5- or 6-digit postal code lifted from the raw address
    normalized_country  2-letter country code (IN / US / FR / <fallback>)

Design guarantees
-----------------
* **No rows are ever dropped.** Nulls/empties are filled with ``""`` before
  processing, so every ``entity_id`` present in the input survives to the
  output (a hard requirement for entity resolution recall).
* **Chunked + checkpointed.** Each source is streamed in row chunks of
  ``CHUNK_SIZE`` and every processed chunk is written to its own
  ``part_{idx:04d}.parquet`` checkpoint. Existing checkpoints are skipped, so a
  killed / re-queued job resumes exactly where it stopped (idempotent).
* **Memory-safe.** Input is read in a single streaming pass (never fully
  materialized), and checkpoints are concatenated into the unified file via a
  lazy streaming sink.
* **Live-monitorable.** Progress is printed with ``flush=True`` so logs can be
  followed with ``tail -f``.

Assumptions
-----------
* Each ``.tsv`` has a **header row** with the four documented columns, in the
  documented order. If your files are headerless, set ``HAS_HEADER = False``.
* Column *values* are read as strings (``infer_schema_length=0``) so numeric
  looking ``entity_id`` values keep their exact original form (e.g. leading
  zeros are never lost).

Usage
-----
    python scripts/preprocess.py                       # uses ./data
    python scripts/preprocess.py --data-dir /path/tsv  # custom input dir
    DATA_DIR=/path/tsv python scripts/preprocess.py    # via environment

Everything (input dir, output dir, checkpoint dir, chunk size) is configurable
through CLI flags or environment variables.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
import warnings
from collections.abc import Iterator
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Configurable chunk size (rows per checkpoint). Overridable via --chunk-size
# or the CHUNK_SIZE environment variable.
CHUNK_SIZE = 100_000

# Canonical schema. Values are read as strings to preserve IDs exactly.
TSV_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# Set to False if your TSV files do NOT contain a header row.
HAS_HEADER = True

# Source files. Train + test entities span the US, India and France.
TRAIN_SOURCES = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]
TEST_SOURCES = ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]

# CSV/TSV reader options shared by the row-count pass and the batched reader.
# These are valid for both `scan_csv` and `read_csv_batched`.
CSV_KWARGS: dict = dict(
    separator="\t",
    has_header=HAS_HEADER,
    new_columns=TSV_COLUMNS,       # enforce canonical names regardless of header spelling
    infer_schema_length=0,         # read EVERY column as String (never lose leading zeros / ids)
    quote_char=None,               # TSV: treat quotes literally -> avoids row-merge corruption
    encoding="utf8-lossy",         # never crash the job on a stray invalid byte
    truncate_ragged_lines=True,    # tolerate malformed over-long rows instead of failing
    empty_string_is_null=False,    # keep empty fields as "" (max fidelity); we still fill_null defensively
    eol_char="\n",
)

# ---------------------------------------------------------------------------
# Cleaning vocabulary (compiled as Rust-regex strings used by Polars)
# ---------------------------------------------------------------------------

# Postal codes: first standalone 5- or 6-digit token (US ZIP / IN PIN / FR code).
# `\b` boundaries prevent matching inside longer numbers (e.g. phone numbers).
PINCODE_PATTERN = r"(\b\d{5,6}\b)"

# Address landmark noise, longest / multi-word alternatives first.
LANDMARK_PATTERN = r"\b(?:in front of|opposite|opp|behind|beside|near)\b"

# Company legal-form suffixes (English + French), longest-first for clarity.
LEGAL_SUFFIX_PATTERN = (
    r"\b(?:corporation|private|limited|sarl|eurl|snc|sci|sas|"
    r"corp|pvt|ltd|inc|llc|sa)\b"
)

# Indian state-code -> full name. Applied to the address only.
STATE_ABBREVIATIONS: dict[str, str] = {
    r"\bhr\b": "haryana",
    r"\bup\b": "uttar pradesh",
    r"\bmh\b": "maharashtra",
    r"\bdl\b": "delhi",
    r"\bka\b": "karnataka",
    r"\btn\b": "tamil nadu",
}

# Country label -> canonical 2-letter code.
COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "IN": ("india", "in", "ind"),
    "US": ("usa", "us", "united states"),
    "FR": ("france", "fr"),
}

# Generic cleanup patterns.
PUNCT_PATTERN = r"[^\w\s]"   # any non-word, non-space char -> replaced by a single space
WHITESPACE_PATTERN = r"\s+"  # runs of whitespace -> single space (run last)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def log(message: str) -> None:
    """Timestamped, flushed print so logs stream live under ``tail -f``."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Vectorized cleaning expressions
#
# Each builder returns a pure Polars expression, so the whole pipeline executes
# in Polars' native engine (no Python-level per-row work). Field scoping:
#   * legal suffixes  -> name   (company legal forms belong to names)
#   * landmarks/states -> address
#   * lowercase / punctuation / whitespace -> both
# ---------------------------------------------------------------------------

def _clean_name_expr() -> pl.Expr:
    """business_name -> clean_name."""
    return (
        pl.col("business_name")
        .fill_null("")                                    # null/empty -> ""
        .str.to_lowercase()                               # lowercase everything
        .str.replace_all(LEGAL_SUFFIX_PATTERN, " ")       # drop legal suffixes
        .str.replace_all(PUNCT_PATTERN, " ")              # strip punctuation -> space
        .str.replace_all(WHITESPACE_PATTERN, " ")         # collapse whitespace
        .str.strip_chars()                                # trim (run last)
        .alias("clean_name")
    )


def _clean_address_expr() -> pl.Expr:
    """business_address -> clean_address."""
    expr = (
        pl.col("business_address")
        .fill_null("")                                    # null/empty -> ""
        .str.to_lowercase()                               # lowercase everything
        .str.replace_all(LANDMARK_PATTERN, " ")           # remove landmark noise
    )
    # Expand Indian state abbreviations (word-boundary matched).
    for pattern, full_name in STATE_ABBREVIATIONS.items():
        expr = expr.str.replace_all(pattern, full_name)
    return (
        expr
        .str.replace_all(PUNCT_PATTERN, " ")              # strip punctuation -> space
        .str.replace_all(WHITESPACE_PATTERN, " ")         # collapse whitespace
        .str.strip_chars()                                # trim (run last)
        .alias("clean_address")
    )


def _pincode_expr() -> pl.Expr:
    """
    business_address -> extracted_pincode.

    Extracted from the RAW (null-filled) address *before* any punctuation
    stripping, exactly as required. Missing -> "".
    """
    return (
        pl.col("business_address")
        .fill_null("")
        .str.extract(PINCODE_PATTERN, 1)
        .fill_null("")
        .alias("extracted_pincode")
    )


def _country_expr() -> pl.Expr:
    """country -> normalized_country (IN / US / FR, else uppercased fallback)."""
    canon = (
        pl.col("country")
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(PUNCT_PATTERN, " ")     # "U.S.A." -> "u s a "
        .str.replace_all(WHITESPACE_PATTERN, " ")
        .str.strip_chars()
    )
    return (
        pl.when(canon.is_in(list(COUNTRY_ALIASES["IN"]))).then(pl.lit("IN"))
        .when(canon.is_in(list(COUNTRY_ALIASES["US"]))).then(pl.lit("US"))
        .when(canon.is_in(list(COUNTRY_ALIASES["FR"]))).then(pl.lit("FR"))
        .otherwise(canon.str.to_uppercase())     # keep unknown labels rather than blanking
        .alias("normalized_country")
    )


def clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply the full cleaning pipeline to a chunk.

    The four original columns are retained untouched; four cleaned columns are
    appended. No rows are added or removed.
    """
    return df.with_columns(
        _clean_name_expr(),
        _clean_address_expr(),
        _pincode_expr(),
        _country_expr(),
    )


# ---------------------------------------------------------------------------
# Chunked streaming I/O
# ---------------------------------------------------------------------------

def count_rows(path: Path) -> int | None:
    """
    Count data rows via a cheap lazy scan (single pass). Returns ``None`` if the
    count cannot be computed, in which case progress percentages are disabled
    but processing still proceeds.
    """
    try:
        n = pl.scan_csv(str(path), **CSV_KWARGS).select(pl.len()).collect().item()
        return int(n)
    except Exception as exc:  # noqa: BLE001 - progress is best-effort only
        log(f"[warn] Could not count rows for {path.name}: {exc}. Progress % disabled.")
        return None


def _iter_raw_batches(path: Path, chunk_size: int) -> Iterator[pl.DataFrame]:
    """
    Yield raw DataFrame batches from a TSV in a single sequential pass.

    Prefers the mature ``read_csv_batched`` reader (stable behaviour; deprecation
    notice silenced) and falls back to the modern ``scan_csv().collect_batches()``
    streaming generator on Polars builds where the batched reader was removed.
    """
    if hasattr(pl, "read_csv_batched"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            reader = pl.read_csv_batched(str(path), batch_size=chunk_size, **CSV_KWARGS)
        while True:
            batches = reader.next_batches(1)
            if not batches:          # None or [] -> exhausted
                break
            yield from batches
        return

    # Fallback: modern streaming batch generator (Polars >= 1.37).
    yield from pl.scan_csv(str(path), **CSV_KWARGS).collect_batches(chunk_size=chunk_size)


def iter_exact_chunks(path: Path, chunk_size: int) -> Iterator[pl.DataFrame]:
    """
    Re-window the underlying (approximately-sized) batches into deterministic
    chunks of *exactly* ``chunk_size`` rows (the final chunk may be smaller).

    Exact, content-defined boundaries make ``part_{idx:04d}.parquet`` checkpoints
    perfectly reproducible across runs, which is what makes resume safe.
    """
    buffer: pl.DataFrame | None = None
    for batch in _iter_raw_batches(path, chunk_size):
        buffer = batch if buffer is None else pl.concat([buffer, batch])
        while buffer.height >= chunk_size:
            yield buffer.head(chunk_size)
            buffer = buffer.slice(chunk_size)
    if buffer is not None and buffer.height > 0:
        yield buffer


def _empty_output_schema() -> dict[str, pl.DataType]:
    """Schema of the unified output: originals + appended cleaned columns."""
    schema: dict[str, pl.DataType] = {c: pl.Utf8 for c in TSV_COLUMNS}
    schema.update(
        clean_name=pl.Utf8,
        clean_address=pl.Utf8,
        extracted_pincode=pl.Utf8,
        normalized_country=pl.Utf8,
    )
    return schema


def _write_empty_unified(path: Path) -> None:
    """Write an empty, correctly-typed unified file (for empty sources)."""
    pl.DataFrame(schema=_empty_output_schema()).write_parquet(str(path))


def _log_progress(
    action: str,
    stem: str,
    chunk_no: int,
    n_chunks: int | None,
    rows_done: int,
    total_rows: int | None,
) -> None:
    """Emit a live progress line, e.g. ``Processing chunk 3/15 (20.0%)``."""
    if n_chunks:
        pct = 100.0 * chunk_no / n_chunks
        head = f"Processing chunk {chunk_no}/{n_chunks} ({pct:.1f}%)"
    else:
        head = f"Processing chunk {chunk_no}"
    rows = f"{rows_done:,}/{total_rows:,}" if total_rows else f"{rows_done:,}"
    tail = "checkpoint exists -> skip" if action == "skip" else "written"
    log(f"  [{stem}] {head} | rows {rows} | {tail}")


# ---------------------------------------------------------------------------
# Per-source driver
# ---------------------------------------------------------------------------

def process_source(
    filename: str,
    data_dir: Path,
    checkpoint_root: Path,
    output_dir: Path,
    chunk_size: int,
) -> None:
    """Process one TSV source end-to-end: chunk -> checkpoint -> unify."""
    src_path = data_dir / filename
    stem = Path(filename).stem                       # e.g. "train_source1"
    unified_path = output_dir / f"clean_{stem}.parquet"
    checkpoint_dir = checkpoint_root / stem

    log(f"=== Source: {filename} ===")

    if not src_path.exists():
        log(f"  [skip] {src_path} not found.")
        return
    if unified_path.exists():
        log(f"  [skip] Unified output already exists: {unified_path.name} "
            f"(delete to reprocess).")
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_rows = count_rows(src_path)
    if total_rows == 0:
        _write_empty_unified(unified_path)
        log(f"  [done] Empty source -> wrote empty {unified_path.name}")
        return
    n_chunks = math.ceil(total_rows / chunk_size) if total_rows else None
    log(f"  rows={total_rows:,} chunk_size={chunk_size:,} chunks={n_chunks}"
        if total_rows else f"  rows=unknown chunk_size={chunk_size:,}")

    # --- Stage 1: chunk -> per-chunk cleaned checkpoints (resumable) ---------
    t0 = time.perf_counter()
    rows_done = 0
    for idx, chunk in enumerate(iter_exact_chunks(src_path, chunk_size)):
        part_path = checkpoint_dir / f"part_{idx:04d}.parquet"
        rows_done += chunk.height

        if part_path.exists():
            _log_progress("skip", stem, idx + 1, n_chunks, rows_done, total_rows)
            continue

        cleaned = clean_dataframe(chunk)
        # Write to a temp file then atomically rename so a crash never leaves a
        # half-written checkpoint that a resume would wrongly trust.
        tmp_path = part_path.with_name(part_path.name + ".tmp")
        cleaned.write_parquet(str(tmp_path))
        tmp_path.replace(part_path)
        _log_progress("done", stem, idx + 1, n_chunks, rows_done, total_rows)

    # --- Stage 2: concatenate checkpoints -> unified file --------------------
    parts = sorted(checkpoint_dir.glob("part_*.parquet"))
    if not parts:
        _write_empty_unified(unified_path)
        log(f"  [warn] No checkpoints produced; wrote empty {unified_path.name}")
        return

    log(f"  Concatenating {len(parts)} checkpoint(s) -> {unified_path.name}")
    tmp_unified = unified_path.with_name(unified_path.name + ".tmp")
    part_strs = [str(p) for p in parts]
    try:
        # Lazy streaming concat -> no full in-memory materialization.
        pl.scan_parquet(part_strs).sink_parquet(str(tmp_unified))
    except Exception as exc:  # noqa: BLE001 - robust fallback for older engines
        log(f"  [warn] streaming sink failed ({exc}); using in-memory concat.")
        pl.concat([pl.read_parquet(p) for p in part_strs]).write_parquet(str(tmp_unified))
    tmp_unified.replace(unified_path)

    # --- Verification: row count must be preserved (no entity dropped) -------
    out_rows = int(pl.scan_parquet(str(unified_path)).select(pl.len()).collect().item())
    elapsed = time.perf_counter() - t0
    log(f"  [done] {filename}: {out_rows:,} rows -> {unified_path.name} in {elapsed:,.1f}s")
    if total_rows is not None and out_rows != total_rows:
        log(f"  [WARN] Row-count mismatch: input={total_rows:,} output={out_rows:,}")
    else:
        log(f"  [ok] Row count preserved ({out_rows:,}) - no entities dropped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polars preprocessing for the Amazon ML Entity Resolution Challenge.",
    )
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"),
                        help="Directory containing the input .tsv files (default: ./data).")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "data/processed"),
                        help="Directory for unified clean_*.parquet outputs.")
    parser.add_argument("--checkpoint-dir", default=os.environ.get("CHECKPOINT_DIR", "data/checkpoints"),
                        help="Directory for per-chunk parquet checkpoints.")
    parser.add_argument("--chunk-size", type=int,
                        default=int(os.environ.get("CHUNK_SIZE", CHUNK_SIZE)),
                        help=f"Rows per chunk / checkpoint (default: {CHUNK_SIZE}).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoint_root = Path(args.checkpoint_dir)
    chunk_size = args.chunk_size

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    log("Amazon ML - Business Entity Resolution :: preprocessing")
    log(f"polars={pl.__version__} | data_dir={data_dir} | output_dir={output_dir} "
        f"| checkpoints={checkpoint_root} | chunk_size={chunk_size:,}")

    sources = TRAIN_SOURCES + TEST_SOURCES
    failures: list[str] = []
    t0 = time.perf_counter()

    for filename in sources:
        try:
            process_source(filename, data_dir, checkpoint_root, output_dir, chunk_size)
        except Exception as exc:  # noqa: BLE001 - isolate per-source failures
            log(f"[ERROR] Failed to process {filename}: {exc}")
            traceback.print_exc()
            failures.append(filename)

    log(f"All sources handled in {time.perf_counter() - t0:,.1f}s")
    if failures:
        log(f"[FAIL] {len(failures)} source(s) failed: {failures}")
        return 1
    log("[SUCCESS] Preprocessing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
