"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ───────────────────────── Vectorized padding ────────────────────────────────


def _ragged_to_dense_2d(
    offsets: "npt.NDArray",
    values: "npt.NDArray",
    max_len: int,
    B: int,
    dtype: Any,
) -> Tuple["npt.NDArray", "npt.NDArray[np.int64]"]:
    """Vectorized ragged -> dense padding (replaces per-row Python loops).

    Given an Arrow ``ListArray``'s ``offsets`` (length >= ``B + 1``, holding
    absolute indices into the flat child ``values`` buffer) and that ``values``
    buffer, build a ``[B, max_len]`` array whose row ``i`` holds the first
    ``min(len_i, max_len)`` elements of list ``i``, right-padded with zeros.

    This reproduces the previous loop exactly:
    - head slice ``values[start:start+use_len]`` keeps the newest tokens for the
      NEWEST->OLDEST sequence storage convention,
    - absolute offsets are used as-is (so sliced/filtered Arrow arrays behave
      identically to the old code).

    Returns ``(padded, used_lengths)`` where
    ``used_lengths[i] = min(len_i, max_len)``. The caller is responsible for any
    ``<= 0`` masking (kept out of here because dense float columns must preserve
    legitimate zero/negative values).
    """
    offsets = np.asarray(offsets)
    starts = offsets[:B].astype(np.int64)
    ends = offsets[1:B + 1].astype(np.int64)
    raw_len = ends - starts
    np.clip(raw_len, 0, None, out=raw_len)
    use = np.minimum(raw_len, max_len)

    out = np.zeros((B, max_len), dtype=dtype)
    total = int(use.sum())
    if total > 0:
        row_start = np.zeros(B, dtype=np.int64)
        if B > 1:
            np.cumsum(use[:-1], out=row_start[1:])
        col_pos = np.arange(total, dtype=np.int64) - np.repeat(row_start, use)
        row_idx = np.repeat(np.arange(B, dtype=np.int64), use)
        src = np.repeat(starts, use) + col_pos
        out[row_idx, col_pos] = values[src]
    return out, use


# ─────────────────────────── schema_dim_caps ─────────────────────────────────

def parse_schema_dim_caps(spec: Any) -> Dict[int, int]:
    """Parse ``"fid:cap,fid:cap,..."`` into ``{fid: cap}``.

    Caps only ever SHRINK a fid's schema dim (applied as ``min(dim, cap)`` when
    the schema is loaded), truncating the rarely-used long tail of multi-value
    lists / dense vectors. Accepts an already-parsed dict (passed through) or
    ``''`` / ``None`` (-> ``{}``). Shared by train.py (for build_dense_stats),
    dataset.py, and indirectly infer.py so all three apply the identical caps.
    """
    if not spec:
        return {}
    if isinstance(spec, dict):
        return {int(k): int(v) for k, v in spec.items()}
    out: Dict[int, int] = {}
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok:
            continue
        fid_s, cap_s = tok.split(':')
        out[int(fid_s)] = int(cap_s)
    return out


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1



class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        seed: int = 42,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        timestamp_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        keep_fraction: float = 1.0,
        longtail_transform: str = 'none',
        dense_stats_path: Optional[str] = None,
        schema_dim_caps: str = '',
        time_decay_alpha: float = 0.0,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            seed: random seed for deterministic shuffle in _flush_buffer.
            ddp_rank: DDP rank (passed from the main process; worker subprocesses inherit).
            ddp_world_size: DDP world_size.
            timestamp_range: optional ``(lo, hi)`` to filter rows by timestamp.
                Only rows with ``lo < ts <= hi`` are kept. Use None for open ends.
            keep_fraction: global fraction of rows that survive ``timestamp_range``
                (rows_kept / rows_total). Used ONLY to scale the progress/ETA
                batch-count estimate (``progress_num_batches``); it does not affect
                iteration. Defaults to 1.0 (no filtering).
            time_decay_alpha: exponential sample-weight decay in days from the
                latest timestamp in the parquet files. 0.0 disables weighting.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self._seed = seed
        self._flush_count = 0
        self._ts_range = timestamp_range
        # Clamp to (0, 1]; only used to scale the progress-bar estimate.
        self._keep_fraction = min(1.0, max(1e-6, float(keep_fraction)))
        # Long-tail dense transform: 'none' = raw (frozen base), or
        # 'log1p_scale_only' = signed_log1p + per-dim /std on long-tail dims,
        # driven by dense_stats.json. Arrays populated by _load_dense_stats.
        self.longtail_transform = longtail_transform
        self.dense_stats_path = dense_stats_path
        self.time_decay_alpha: float = float(time_decay_alpha)
        if self.time_decay_alpha < 0:
            raise ValueError(
                f"time_decay_alpha must be >= 0, got {time_decay_alpha}")
        self._t_reference: float = 0.0
        # Per-fid schema dim caps ({fid: cap}); shrink-only, applied in
        # _load_schema before any buffers / plans are built. Truncates the
        # rarely-used long tail of multi-value int lists and dense vectors.
        self._schema_dim_caps: Dict[int, int] = parse_schema_dim_caps(schema_dim_caps)
        self._ud_log1p_mask: Optional[np.ndarray] = None   # (user_dense_dim,) bool
        self._ud_std_inv: Optional[np.ndarray] = None      # (1, user_dense_dim) f32
        self._id_log1p_mask: Optional[np.ndarray] = None   # (item_dense_dim,) bool
        self._id_std_inv: Optional[np.ndarray] = None      # (1, item_dense_dim) f32
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        # DDP rank-level sharding: greedy allocation by row count to balance across ranks.
        if ddp_world_size > 1:
            sorted_rgs = sorted(self._rg_list, key=lambda x: x[2], reverse=True)
            rank_buckets = [[] for _ in range(ddp_world_size)]
            rank_rows = [0] * ddp_world_size
            for rg in sorted_rgs:
                min_rank = min(range(ddp_world_size), key=lambda r: rank_rows[r])
                rank_buckets[min_rank].append(rg)
                rank_rows[min_rank] += rg[2]
            self._rg_list = rank_buckets[ddp_rank]
            logging.info(f"DDP shard: rank={ddp_rank}, world_size={ddp_world_size}, "
                         f"row_groups={len(self._rg_list)}, rows={rank_rows[ddp_rank]} "
                         f"(all ranks: {rank_rows})")

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # Load dense_stats.json (per-dim log1p_mask + std for the long-tail
        # transform). Needs the user/item dense schema dims, so call it after
        # _load_schema. No-op when longtail_transform == 'none'.
        #   'log1p'            = signed_log1p on the whitelist dims only (NO /std).
        #   'log1p_scale_only' = signed_log1p + per-dim /std on the whitelist.
        # 'log1p' matches the SeFCN reference, which never z-scores a dim that
        # feeds the dense projection (its /std only ever touched weight-only pair
        # fids, where it is a no-op). /std on feature dims amplifies their
        # low-variance components (worst case x100 at STD_FLOOR), so it is opt-in.
        if self.longtail_transform not in ('none', 'log1p', 'log1p_scale_only'):
            raise ValueError(
                f"longtail_transform must be 'none', 'log1p' or "
                f"'log1p_scale_only', got {self.longtail_transform!r}")
        if self.longtail_transform != 'none':
            if not (self.dense_stats_path and os.path.exists(self.dense_stats_path)):
                raise FileNotFoundError(
                    f"longtail_transform={self.longtail_transform!r} requires a "
                    f"dense_stats.json; dense_stats_path={self.dense_stats_path!r} "
                    f"is missing. Build it once with build_dense_stats.py.")
            self._load_dense_stats(self.dense_stats_path)

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # Time-decay sample weight (Temporal Importance Factor).
        if self.time_decay_alpha > 0:
            ts_ci = self._col_idx.get('timestamp')
            if ts_ci is None:
                raise ValueError(
                    "time_decay_alpha > 0 requires a 'timestamp' column "
                    "in the parquet schema")
            max_ts = 0
            for f in self._parquet_files:
                _pf = pq.ParquetFile(f)
                for rg_idx in range(_pf.metadata.num_row_groups):
                    stats = _pf.metadata.row_group(rg_idx).column(ts_ci).statistics
                    if stats is not None and getattr(stats, 'has_min_max', False):
                        rg_max = int(stats.max)
                        if rg_max > max_ts:
                            max_ts = rg_max
            if max_ts == 0:
                raise ValueError(
                    "time_decay_alpha > 0 requested but no timestamp "
                    "statistics were found in any parquet row-group metadata")
            self._t_reference = float(max_ts)
            half_life_days = float(np.log(2.0) / self.time_decay_alpha)
            logging.info(
                f"PCVRParquetDataset: time-decay sample weight enabled, "
                f"alpha={self.time_decay_alpha:.3f}, "
                f"t_reference={int(max_ts)} (latest ts in parquet); "
                f"half-life ~= {half_life_days:.2f} days")

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_item_dense = np.zeros((B, self.item_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        self._item_dense_plan = []
        offset = 0
        for fid, dim in self._item_dense_cols:
            ci = self._col_idx.get(f'item_dense_feats_{fid}')
            self._item_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # schema_dim_caps: shrink-only per-fid truncation of the head-slice
        # length. Multi-value int lists are stored newest-first (head = newest;
        # for the verified (id, weight) pairs head = highest weight), so capping
        # keeps the most informative head and drops a mostly-padding tail.
        caps = self._schema_dim_caps

        def _cap(fid: int, dim: int) -> int:
            c = caps.get(fid)
            if c is not None and c < dim:
                logging.info(f"schema_dim_caps: fid {fid} dim {dim} -> {c}")
                return c
            return dim

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = [
            [fid, vs, _cap(fid, dim)] for fid, vs, dim in raw['user_int']
        ]
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = [
            [fid, vs, _cap(fid, dim)] for fid, vs, dim in raw['item_int']
        ]
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        # Default override for features whose effective length < schema max_len.
        # An explicit schema_dim_caps entry SUPERSEDES this override (so e.g.
        # 130:263 wins over the 259 default), keeping caps the single source of
        # truth for any fid the caller names.
        _USER_DENSE_DIM_OVERRIDE = {
            130: 259,  # effective dim=259, tail is unrelated stats
        }

        def _cap_user_dense(fid: int, dim: int) -> int:
            if fid in caps:
                return _cap(fid, dim)
            return _USER_DENSE_DIM_OVERRIDE.get(fid, dim)

        self._user_dense_cols: List[List[int]] = [
            [fid, _cap_user_dense(fid, dim)] for fid, dim in raw['user_dense']
        ]
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense: [[fid, dim], ...] ----
        self._item_dense_cols: List[List[int]] = [
            [fid, _cap(fid, dim)] for fid, dim in raw.get('item_dense', [])
        ]
        self.item_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._item_dense_cols:
            self.item_dense_schema.add(fid, dim)

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def _load_dense_stats(self, path: str) -> None:
        """Load dense_stats.json (built by build_dense_stats.py) and populate the
        per-dim ``log1p_mask`` / ``std_inv`` arrays for user_dense and item_dense.

        Validates that each group's ``total_dim`` and ``fid_layout`` match this
        dataset's runtime schema (post schema_dim_caps + the 130->259 default
        override), so the per-dim arrays line up with the flattened dense
        buffers. ``std_inv = 1/std`` (std=1.0 on
        non-long-tail dims => the runtime ``x *= std_inv`` is a no-op there).
        """
        with open(path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        logging.info(f"Loading dense_stats from {path} "
                     f"(format={stats.get('format')}, transform={stats.get('dense_transform')})")

        def _resolve(group_name: str, runtime_cols: List[List[int]]):
            block = stats.get(group_name)
            if not block:
                return None, None
            runtime_layout = [[int(fid), int(dim)] for fid, dim in runtime_cols]
            runtime_dim = sum(d for _, d in runtime_layout)
            built_layout = [[int(fid), int(dim)] for fid, dim in block['fid_layout']]
            if block['total_dim'] != runtime_dim or built_layout != runtime_layout:
                raise ValueError(
                    f"dense_stats[{group_name}] layout mismatch: built "
                    f"total_dim={block['total_dim']}, fid_layout={built_layout} vs "
                    f"runtime total_dim={runtime_dim}, fid_layout={runtime_layout}. "
                    f"Rebuild dense_stats.json from the same schema.")
            mask = np.asarray(block['log1p_mask'], dtype=bool)
            std = np.asarray(block['std'], dtype=np.float32)
            std_inv = (1.0 / std).reshape(1, -1).astype(np.float32)
            n_sel = int(mask.sum())
            logging.info(f"  dense_stats[{group_name}]: {runtime_dim} dims, "
                         f"{n_sel} long-tail (log1p+scale), fids={block.get('longtail_fids')}")
            return mask, std_inv

        self._ud_log1p_mask, self._ud_std_inv = _resolve('user_dense', self._user_dense_cols)
        if self.item_dense_schema.total_dim > 0:
            self._id_log1p_mask, self._id_std_inv = _resolve('item_dense', self._item_dense_cols)

        # 'log1p' mode: keep the log1p masks but drop the /std step entirely, so
        # no dim that feeds the dense projection is z-scored (SeFCN-faithful).
        # _convert_batch already guards each /std multiply on std_inv is not None.
        if self.longtail_transform == 'log1p':
            self._ud_std_inv = None
            self._id_std_inv = None
            logging.info("longtail_transform='log1p': /std disabled (log1p only)")

    def estimated_num_batches(self) -> int:
        """Estimate batch count for logging/progress display only."""
        return (self.num_rows + self.batch_size - 1) // self.batch_size

    def progress_num_batches(self) -> int:
        """Batch-count denominator for the progress bar / ETA (display only).

        The estimate must match how ``__iter__`` actually yields batches, which
        differs by whether the shuffle buffer re-packs:

        * No re-pack (``buffer_batches<=1``): each raw Arrow batch is filtered
          in-place and yielded as-is.
          Row Groups are NOT time-ordered in rd2, so a ``--split_by_time`` filter
          empties almost no raw batch — the batch COUNT stays at the raw per-Row-
          Group ceiling sum (only the rows-per-batch shrink). So ``keep_fraction``
          must NOT scale it here; doing so under-counts ~1/keep_fraction-fold and
          starves the progress prints. This sum is exact for no-repack loaders.

        * Re-pack (``buffer_batches>1``; train and optionally valid): the flush
          re-slices kept rows into full batches, so the count scales with the
          kept fraction and row count rather than the raw Row Group count. Worker
          flush remainders can add a few tail batches, so this remains a display
          estimate, not a training control value.
        """
        bs = self.batch_size
        repacks = self.buffer_batches > 1
        if repacks and self._keep_fraction < 1.0:
            kept_rows = round(self.num_rows * self._keep_fraction)
            return max(1, (kept_rows + bs - 1) // bs)
        if repacks:
            return max(1, (self.num_rows + bs - 1) // bs)
        return sum((n + bs - 1) // bs for _, _, n in self._rg_list)

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count
        # when no timestamp filter is applied. See progress_num_batches() for the
        # filter-aware estimate used by the trainer's progress display.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        self._flush_count = 0  # Reset each epoch

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if batch_dict is None:
                    continue
                if self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self._seed + self._flush_count)
            self._flush_count += 1
            rand_idx = torch.randperm(total_rows, generator=g)
        else:
            rand_idx = torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded, lengths = _ragged_to_dense_2d(offsets, values, max_len, B, np.int64)
        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded, _ = _ragged_to_dense_2d(offsets, values, max_dim, B, np.float32)
        return padded

    def _convert_batch(self, batch: "pa.RecordBatch") -> Optional[Dict[str, Any]]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""

        # ---- timestamp range filter ----
        if self._ts_range is not None:
            lo, hi = self._ts_range
            ts_col = batch.column(self._col_idx['timestamp'])
            if lo is not None and hi is not None:
                mask = pc.and_(pc.greater(ts_col, lo), pc.less_equal(ts_col, hi))
            elif hi is not None:
                mask = pc.less_equal(ts_col, hi)
            else:
                mask = pc.greater(ts_col, lo)
            batch = batch.filter(mask)
            if batch.num_rows == 0:
                return None

        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        sample_weight = None
        if self.time_decay_alpha > 0:
            days_to_ref = (self._t_reference - timestamps.astype(np.float64)) / 86400.0
            sample_weight = np.exp(-self.time_decay_alpha * days_to_ref).astype(np.float32)
        # user_id is only consumed at inference time (predictions.json keys).
        # During training/eval it is unused, so skip the per-batch Python-list
        # build (.to_pylist boxes B ints every batch in the worker).
        if self.is_training:
            user_ids = None
        else:
            user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        # Long-tail transform: signed_log1p on the masked dims, then /std.
        # user_dense_feats_raw is snapshotted AFTER log1p but BEFORE /std — this
        # is the non-negative weight stream the (id, weight) pair pooling reads.
        # Only produced when the transform is active (frozen base => omitted).
        user_dense_raw = None
        if self._ud_log1p_mask is not None:
            if self._ud_log1p_mask.any():
                sub = user_dense[:, self._ud_log1p_mask]
                user_dense[:, self._ud_log1p_mask] = np.sign(sub) * np.log1p(np.abs(sub))
            user_dense_raw = user_dense.copy()
            if self._ud_std_inv is not None:
                user_dense *= self._ud_std_inv

        # ---- item_dense ----
        if self.item_dense_schema.total_dim > 0:
            item_dense = self._buf_item_dense[:B]
            item_dense[:] = 0
            for ci, dim, offset in self._item_dense_plan:
                col = batch.column(ci)
                padded = self._pad_varlen_float_column(col, dim, B)
                item_dense[:, offset:offset + dim] = padded
            if self._id_log1p_mask is not None and self._id_log1p_mask.any():
                sub = item_dense[:, self._id_log1p_mask]
                item_dense[:, self._id_log1p_mask] = np.sign(sub) * np.log1p(np.abs(sub))
            if self._id_std_inv is not None:
                item_dense *= self._id_std_inv
            item_dense_tensor = torch.from_numpy(item_dense.copy())
        else:
            item_dense_tensor = torch.zeros(B, 0, dtype=torch.float32)

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': item_dense_tensor,
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }
        if sample_weight is not None:
            result['sample_weight'] = torch.from_numpy(sample_weight)
        if user_dense_raw is not None:
            result['user_dense_feats_raw'] = torch.from_numpy(user_dense_raw)

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for di, domain in enumerate(self.seq_domains):
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                padded_c, use_c = _ragged_to_dense_2d(offs, vals, max_len, B, out.dtype)
                out[:, c, :] = padded_c
                np.maximum(lengths, use_c, out=lengths)

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                ts_padded, _ = _ragged_to_dense_2d(ts_offs, ts_vals, max_len, B, np.int64)

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    split_by_time: bool = False,
    longtail_transform: str = 'none',
    dense_stats_path: Optional[str] = None,
    schema_dim_caps: str = '',
    time_decay_alpha: float = 0.0,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    Split modes:
      - ``split_by_time=False`` (default): split by Row Group position.
        The last ``valid_ratio`` fraction of Row Groups becomes validation.
      - ``split_by_time=True``: split by timestamp. Scan the timestamp column
        to find a cutoff so that the last ``valid_ratio`` of rows (by time
        order) become validation.

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    use_cuda = torch.cuda.is_available()

    if split_by_time:
        # ---- Time-based split: find timestamp cutoff at (1 - valid_ratio) quantile ----
        from concurrent.futures import ThreadPoolExecutor

        def _read_ts(fpath):
            return pq.read_table(fpath, columns=['timestamp']).column('timestamp').to_numpy()

        with ThreadPoolExecutor(max_workers=32) as pool:
            all_ts = list(pool.map(_read_ts, pq_files))
        all_ts = np.concatenate(all_ts)
        total_rows = len(all_ts)

        cutoff = int(np.percentile(all_ts, (1.0 - valid_ratio) * 100))
        train_rows = int((all_ts <= cutoff).sum())
        valid_rows = total_rows - train_rows

        logging.info(f"Time-based split: cutoff_ts={cutoff}, "
                     f"train={train_rows} rows, valid={valid_rows} rows, "
                     f"total={total_rows}")
        del all_ts

        train_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=shuffle_train,
            buffer_batches=buffer_batches,
            clip_vocab=clip_vocab,
            seed=seed,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            timestamp_range=(None, cutoff),
            keep_fraction=(train_rows / total_rows) if total_rows else 1.0,
            longtail_transform=longtail_transform,
            dense_stats_path=dense_stats_path,
            schema_dim_caps=schema_dim_caps,
            time_decay_alpha=time_decay_alpha,
        )

        _train_kw = {}
        if num_workers > 0:
            _train_kw['persistent_workers'] = True
            _train_kw['prefetch_factor'] = 2

        train_loader = DataLoader(
            train_dataset, batch_size=None,
            num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
        )

        valid_buffer_batches = buffer_batches if buffer_batches > 1 else 0
        valid_num_workers = min(4, max(0, int(num_workers)))
        valid_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=False,
            buffer_batches=valid_buffer_batches,
            clip_vocab=clip_vocab,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            timestamp_range=(cutoff, None),
            keep_fraction=(valid_rows / total_rows) if total_rows else 1.0,
            longtail_transform=longtail_transform,
            dense_stats_path=dense_stats_path,
            schema_dim_caps=schema_dim_caps,
            time_decay_alpha=time_decay_alpha,
        )
        _valid_kw = {}
        if valid_num_workers > 0:
            _valid_kw['prefetch_factor'] = 2
        valid_loader = DataLoader(
            valid_dataset, batch_size=None,
            num_workers=valid_num_workers, pin_memory=use_cuda, **_valid_kw,
        )

        logging.info(f"Parquet (time split): train={train_rows}, valid={valid_rows}, "
                     f"batch_size={batch_size}, buffer_batches={buffer_batches}, "
                     f"valid_buffer_batches={valid_buffer_batches}, "
                     f"valid_num_workers={valid_num_workers}")

        return train_loader, valid_loader, train_dataset

    # ---- Row Group position-based split (original behavior) ----
    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
        seed=seed,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        longtail_transform=longtail_transform,
        dense_stats_path=dense_stats_path,
        schema_dim_caps=schema_dim_caps,
        time_decay_alpha=time_decay_alpha,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_buffer_batches = buffer_batches if buffer_batches > 1 else 0
    valid_num_workers = min(4, max(0, int(num_workers)))
    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=valid_buffer_batches,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        longtail_transform=longtail_transform,
        dense_stats_path=dense_stats_path,
        schema_dim_caps=schema_dim_caps,
        time_decay_alpha=time_decay_alpha,
    )
    _valid_kw = {}
    if valid_num_workers > 0:
        _valid_kw['prefetch_factor'] = 2
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=valid_num_workers, pin_memory=use_cuda, **_valid_kw,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}, "
                 f"valid_buffer_batches={valid_buffer_batches}, "
                 f"valid_num_workers={valid_num_workers}")

    return train_loader, valid_loader, train_dataset
