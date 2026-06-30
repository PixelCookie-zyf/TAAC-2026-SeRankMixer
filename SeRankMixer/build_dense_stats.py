"""Build dense_stats.json — per-dim std for user_dense + item_dense, computed
AFTER signed_log1p, for the long-tail / large-magnitude dense fids only.

Run ONCE on the platform as a TRAINING-side job (reads TRAIN_DATA_PATH); it
writes dense_stats.json. Commit that JSON into baseline_rd2/dense_stats.json so
train/infer just LOAD it (no per-run recompute). train.py also copies it into
the checkpoint dir so infer can read it from MODEL_OUTPUT_PATH.

What it does
------------
For each dense group (user_dense, item_dense):
  1. Scan rows, track a Welford mean/std of signed_log1p(x) (the post-transform
     distribution the model will see) and per-dim raw |max| (for logging only).
  2. A fid is "long-tail" iff it is on the hardcoded whitelist
     ``_DEFAULT_LONGTAIL_FIDS`` (override via ``--longtail_fids`` /
     ``$LONGTAIL_FIDS``). The whitelist replaces the old ``absmax > threshold``
     auto-selection, which was too coarse: it pulled in bounded / near-constant
     dense (e.g. fid 120, max≈2) whose tiny std hit STD_FLOOR and got blown up
     ~100x by the /std step, destabilising training. The whitelist is the
     genuinely count / heavy-tail fids only:
       user_dense: 62,63,64,65,66,118,121,131,132
       item_dense: 124,129
     and deliberately EXCLUDES 61/87/89/90/91/120/123 (and item 127/128). 130 is
     held out for a separate +130 ablation.
  3. Long-tail fids get log1p_mask=True and std = std(signed_log1p(x)) (floored
     at STD_FLOOR). Other fids get log1p_mask=False, std=1.0 (runtime no-op).

The runtime (dataset.py) then, per dim where log1p_mask: x = sign(x)*log1p(|x|),
then x /= std (no mean subtraction = scale_only; preserves "0 = absent").

Output JSON (small, ~100-200KB):
  {
    "format": "rd2_dense_stats_v1",
    "dense_transform": "signed_log1p",
    "longtail_selection": "whitelist",
    "longtail_fids_requested": [...],
    "std_floor": 0.01,
    "n_samples": <int>,
    "build_data_dir": <abspath>, "build_time": <iso>,
    "user_dense": {"total_dim", "fid_layout": [[fid,dim],...],
                   "log1p_mask": [bool,...], "std": [float,...],
                   "longtail_fids": [...]},
    "item_dense": {... same ...}
  }

Env vars:
  TRAIN_DATA_PATH                 data dir (*.parquet)
  SCHEMA_PATH                     schema.json (defaults to <data>/schema.json)
  DENSE_STATS_OUTPUT_DIR          where to write dense_stats.json (default: cwd
                                  or $TRAIN_LOG_PATH)
  ANALYZE_SAMPLE_FILES            row-group cap (0 = full scan; default 0)
  LONGTAIL_FIDS                   csv of fids to transform; empty = the hardcoded
                                  _DEFAULT_LONGTAIL_FIDS whitelist
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STD_FLOOR = 0.01

# Hardcoded whitelist of fids to apply signed_log1p + per-dim /std to. These are
# the genuinely count / heavy-tail / wide-scale dense fids; everything else
# (pretrained 61/87, signed pairs 89/90/91, bounded 120, stable stats 123, item
# 127/128) is left raw so its geometry is not distorted. user_dense and
# item_dense fids are disjoint, so a single flat set is unambiguous. 130 is held
# out for a separate +130 ablation (its main open question is effective dim
# 259 vs 263, not the transform).
_DEFAULT_LONGTAIL_FIDS = [62, 63, 64, 65, 66, 118, 121, 131, 132, 124, 129]


def build_weighted_pairs(
    user_int_schema: Any,
    user_dense_schema: Any,
    fids_csv: str,
) -> List[Tuple[int, int, int]]:
    """Resolve a CSV of user fids to ``[(int_fid_idx, dense_offset, length), ...]``
    for the (id, weight)-pair pooling. A fid is kept only if present in BOTH the
    user_int and user_dense schemas with equal per-row length (ids and weights
    are then position-aligned). ``''`` -> ``[]`` (feature off).

    ``user_int_schema`` / ``user_dense_schema`` are ``FeatureSchema``-like objects
    exposing ``.entries = [(fid, offset, length), ...]``. Shared by train.py and
    infer.py so both reconstruct the identical pooling map.
    """
    fids_csv = (fids_csv or "").strip()
    if not fids_csv:
        return []
    int_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(user_int_schema.entries)}
    int_len = {fid: length for fid, _, length in user_int_schema.entries}
    dense_entry = {fid: (off, length) for fid, off, length in user_dense_schema.entries}
    pairs: List[Tuple[int, int, int]] = []
    for tok in fids_csv.split(','):
        tok = tok.strip()
        if not tok:
            continue
        fid = int(tok)
        if fid not in int_fid_to_idx:
            logging.warning(f"weighted_pair_pool_fids: fid {fid} not in user_int schema; skipping")
            continue
        if fid not in dense_entry:
            logging.warning(f"weighted_pair_pool_fids: fid {fid} not in user_dense schema; skipping")
            continue
        d_off, d_len = dense_entry[fid]
        if int_len[fid] != d_len:
            logging.warning(
                f"weighted_pair_pool_fids: fid {fid} int_len={int_len[fid]} != "
                f"dense_len={d_len}; not position-aligned, skipping")
            continue
        pairs.append((int_fid_to_idx[fid], d_off, d_len))
        logging.info(
            f"weighted_pair_pool: fid {fid} -> fid_idx={int_fid_to_idx[fid]}, "
            f"dense_offset={d_off}, length={d_len}")
    return pairs

# Must match dataset.py's _USER_DENSE_DIM_OVERRIDE so the per-dim arrays line up
# with the runtime user_dense layout.
_USER_DENSE_DIM_OVERRIDE: Dict[int, int] = {130: 259}


def log(msg: str) -> None:
    logging.info(msg)
    print(msg, flush=True)


def _resolve_files(data_dir: str) -> List[str]:
    if os.path.isdir(data_dir):
        files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        if files:
            return files
    if data_dir.endswith(".parquet") and os.path.exists(data_dir):
        return [data_dir]
    raise FileNotFoundError(f"{data_dir!r} is not a directory or single parquet")


def _load_layout(schema_path: str, group: str,
                 dim_caps: Optional[Dict[int, int]] = None) -> List[List[int]]:
    """Return [[fid, dim], ...] for a dense group from schema.json."""
    with open(schema_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    layout = [list(e) for e in raw.get(group, [])]
    if dim_caps:
        for entry in layout:
            fid = entry[0]
            if fid in dim_caps and entry[1] > dim_caps[fid]:
                entry[1] = dim_caps[fid]
    return layout


def _list_row_groups(files: List[str], sample_rgs: Optional[int]
                     ) -> Tuple[List[Tuple[str, int]], int]:
    rg_list: List[Tuple[str, int]] = []
    total_rows = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_list.append((f, i))
            total_rows += pf.metadata.row_group(i).num_rows
    if sample_rgs and 0 < sample_rgs < len(rg_list):
        # Evenly-strided sample across ALL row groups (not the first N) so the
        # std estimate is not biased by file/time ordering (e.g. dense features
        # that drift over time, like item_dense_129).
        stride = max(1, len(rg_list) // sample_rgs)
        sampled = rg_list[::stride][:sample_rgs]
        return sampled, total_rows
    return rg_list, total_rows


def _pad_varlen_float(arrow_col, max_dim: int, B: int) -> np.ndarray:
    """Head-slice + zero-pad a ListArray of floats to (B, max_dim).

    Mirrors dataset.py._ragged_to_dense_2d (head slice keeps the first
    min(len, max_dim) values).
    """
    offsets = arrow_col.offsets.to_numpy()
    values = arrow_col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    out = np.zeros((B, max_dim), dtype=np.float32)
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        n = e - s
        if n <= 0:
            continue
        use = min(n, max_dim)
        out[i, :use] = values[s:s + use]
    return out


def _accumulate_group(files: List[str], rg_list: List[Tuple[str, int]],
                      group: str, layout: List[List[int]],
                      longtail_set: set) -> Dict:
    total_dim = sum(d for _, d in layout)
    columns = [f"{group}_feats_{fid}" for fid, _ in layout]

    n = 0
    mean = np.zeros(total_dim, dtype=np.float64)
    m2 = np.zeros(total_dim, dtype=np.float64)
    raw_absmax = np.zeros(total_dim, dtype=np.float64)

    by_file: Dict[str, List[int]] = {}
    for f, rg in rg_list:
        by_file.setdefault(f, []).append(rg)

    for f, rgs in by_file.items():
        try:
            pf = pq.ParquetFile(f)
        except OSError:
            continue
        avail = set(pf.schema_arrow.names)
        use_cols = [c for c in columns if c in avail]
        if not use_cols:
            continue
        for rg in rgs:
            try:
                table = pf.read_row_group(rg, columns=use_cols)
            except (KeyError, pa.ArrowInvalid):
                continue
            B = table.num_rows
            if B == 0:
                continue
            chunk = np.zeros((B, total_dim), dtype=np.float32)
            offset = 0
            for fid, dim in layout:
                col_name = f"{group}_feats_{fid}"
                if col_name in table.column_names:
                    col = table.column(col_name).combine_chunks()
                    chunk[:, offset:offset + dim] = _pad_varlen_float(col, dim, B)
                offset += dim
            # raw |max| per dim (selection signal, on raw values)
            np.maximum(raw_absmax, np.abs(chunk).max(axis=0), out=raw_absmax)
            # Welford on signed_log1p(x)
            transformed = (np.sign(chunk) * np.log1p(np.abs(chunk))).astype(np.float64)
            B64 = transformed.shape[0]
            n_new = n + B64
            delta = transformed - mean
            mean = mean + delta.sum(axis=0) / n_new
            m2 = m2 + (delta * (transformed - mean)).sum(axis=0)
            n = n_new

    raw_std = np.sqrt(m2 / max(n - 1, 1))

    # Per-fid long-tail selection by hardcoded whitelist (membership), NOT by an
    # absmax threshold. raw_absmax is still computed and logged per fid so the
    # decision stays auditable (and so an over-large std-floor blow-up is
    # visible), but it no longer drives selection.
    log1p_mask = np.zeros(total_dim, dtype=bool)
    std = np.ones(total_dim, dtype=np.float64)
    longtail_fids: List[int] = []
    offset = 0
    for fid, dim in layout:
        fid_absmax = float(raw_absmax[offset:offset + dim].max()) if dim > 0 else 0.0
        selected = int(fid) in longtail_set
        if selected:
            log1p_mask[offset:offset + dim] = True
            std[offset:offset + dim] = np.maximum(raw_std[offset:offset + dim], STD_FLOOR)
            longtail_fids.append(int(fid))
        log(f"  [{group}] fid {fid}: dim={dim}, raw|max|={fid_absmax:.4g}, "
            f"{'LOG1P+SCALE' if selected else 'raw'}")
        offset += dim

    log(f"  [{group}] total_dim={total_dim}, n_rows={n:,}, "
        f"long-tail fids (whitelist): {longtail_fids}")
    return {
        "total_dim": int(total_dim),
        "fid_layout": [[int(fid), int(dim)] for fid, dim in layout],
        "log1p_mask": log1p_mask.tolist(),
        "std": [float(s) for s in std],
        "longtail_fids": longtail_fids,
        "n_samples": int(n),
    }


def build_dense_stats_json(
    data_dir: str,
    schema_path: str,
    output_path: str,
    sample_rgs: Optional[int] = None,
    longtail_fids: Optional[List[int]] = None,
    schema_dim_caps: Optional[Dict[int, int]] = None,
) -> Dict:
    t0 = time.time()
    files = _resolve_files(data_dir)
    rg_list, total_rows = _list_row_groups(files, sample_rgs)
    longtail_list = list(_DEFAULT_LONGTAIL_FIDS if longtail_fids is None else longtail_fids)
    longtail_set = {int(f) for f in longtail_list}
    log("dense_stats build starting")
    log(f"  data_dir   = {data_dir}")
    log(f"  schema     = {schema_path}")
    log(f"  files={len(files)}, row_groups(scanned)={len(rg_list)}, "
        f"total_rows(all)={total_rows:,}")
    log(f"  longtail_fids (whitelist) = {sorted(longtail_set)}")

    # Effective per-fid dim for the dense layout MUST match dataset.py's
    # _load_schema exactly (the dense_stats validation compares total_dim /
    # fid_layout against the runtime schema). For user_dense: an explicit
    # schema_dim_caps entry supersedes the 130->259 override; other fids keep
    # the override. _load_layout truncates shrink-only, so merging override +
    # caps (caps last, so they win) reproduces dataset.py's precedence.
    caps = dict(schema_dim_caps or {})
    if caps:
        log(f"  schema_dim_caps = {caps}")
    user_caps = dict(_USER_DENSE_DIM_OVERRIDE)
    user_caps.update(caps)  # explicit caps win over the default override
    user_layout = _load_layout(schema_path, "user_dense", user_caps)
    item_layout = _load_layout(schema_path, "item_dense", caps or None)

    payload: Dict = {
        "format": "rd2_dense_stats_v1",
        "dense_transform": "signed_log1p",
        "longtail_selection": "whitelist",
        "longtail_fids_requested": sorted(longtail_set),
        "std_floor": STD_FLOOR,
        "build_data_dir": os.path.abspath(data_dir),
        "build_sample_rgs": sample_rgs,
        "build_time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    if user_layout:
        payload["user_dense"] = _accumulate_group(files, rg_list, "user_dense", user_layout, longtail_set)
    if item_layout:
        payload["item_dense"] = _accumulate_group(files, rg_list, "item_dense", item_layout, longtail_set)
    payload["n_samples"] = max(
        payload.get("user_dense", {}).get("n_samples", 0),
        payload.get("item_dense", {}).get("n_samples", 0),
    )

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_kb = os.path.getsize(output_path) / 1024
    log("")
    log("=" * 60)
    log(f"  dense_stats written -> {output_path}  ({size_kb:.1f} KB)")
    log(f"  total wallclock: {time.time() - t0:.1f}s")
    log("=" * 60)
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
    data_dir = os.environ.get("TRAIN_DATA_PATH") or os.environ.get("EVAL_DATA_PATH")
    if not data_dir:
        raise RuntimeError("TRAIN_DATA_PATH (or EVAL_DATA_PATH) env var not set")
    schema_path = os.environ.get("SCHEMA_PATH") or os.path.join(data_dir, "schema.json")
    output_dir = (os.environ.get("DENSE_STATS_OUTPUT_DIR")
                  or os.environ.get("TRAIN_LOG_PATH") or os.getcwd())
    output_path = os.path.join(output_dir, "dense_stats.json")
    sample_env = os.environ.get("ANALYZE_SAMPLE_FILES", "0")
    sample_rgs = int(sample_env) if sample_env else 0
    fids_env = os.environ.get("LONGTAIL_FIDS", "").strip()
    longtail_fids = [int(t) for t in fids_env.split(",") if t.strip()] if fids_env else None
    build_dense_stats_json(
        data_dir=data_dir,
        schema_path=schema_path,
        output_path=output_path,
        sample_rgs=sample_rgs if sample_rgs > 0 else None,
        longtail_fids=longtail_fids,
    )


if __name__ == "__main__":
    main()
