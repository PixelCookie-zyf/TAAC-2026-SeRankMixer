"""PCVRRankMixer inference script (uploaded by the contestant into the
evaluation container).

Model construction mirrors ``train.py``: we rebuild the model from
``schema.json`` + ``train_config.json``. All model
hyperparameters are resolved first from the ckpt directory's
``train_config.json`` (written by ``trainer.py`` when saving a checkpoint),
falling back to ``_FALLBACK_MODEL_CFG`` below (which must stay consistent
with the CLI defaults in ``train.py``).

Only the Parquet data format is supported.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (points at the ``global_step``
                       sub-directory containing ``model.pt`` / ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import os
import json
import logging
import contextlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRRankMixer, ModelInput
from utils import configure_sdp_kernels
from build_dense_stats import build_weighted_pairs


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# Fallback values used only when ``train_config.json`` is missing from the
# ckpt directory.
#
# These MUST match the argparse defaults in ``train.py``; otherwise once the
# fallback path is actually taken the built model will shape-mismatch the
# saved state_dict.
#
# Special note on ``num_time_buckets``: this value is strictly determined by
# ``dataset.BUCKET_BOUNDARIES`` and is NOT an independent hyperparameter.
# When the feature is enabled we therefore use the constant exposed by the
# dataset module; ``0`` means disabled.
_FALLBACK_MODEL_CFG = {
    'd_model': 64,
    'emb_dim': 64,
    'hidden_mult': 4,
    'dropout_rate': 0.01,
    'action_num': 1,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'emb_skip_threshold': 0,
    'seq_id_threshold': 10000,
    # CSV of user fids with log1p-weighted (id, weight)-pair pooling; '' = off.
    # Resolved to a (fid_idx, dense_offset, length) map at model-build time.
    'weighted_pair_pool_fids': '',
    # SeFCN-backbone hyperparams (round-tripped from train_config.json).
    'pair_pool': 'mean',
    'concat_senet_dims': '1024,256',
    'concat_mlp_dims': '1024,512,256,128',
    'two_stream_mlp': 0,
    'two_stream_dims': '512,128',
    'two_stream_input': 'senet',
    'two_stream_fusion': 'sum',
    'bilinear_groups': 8,
    # RankMixer cross hyperparams (cross_arch=rankmixer).
    'rankmixer_tokens': 16,
    'rankmixer_dim': 128,
    'rankmixer_heads': 0,
    'rankmixer_layers': 2,
    'rankmixer_expansion': 4,
    'rankmixer_pool': 'mean',
    'rankmixer_ffn': 'pswiglu',
    'rankmixer_down_init_gain': 0.01,
    'rankmixer_moe_experts': 4,
    'rankmixer_moe_top_k': 2,
    'rankmixer_remoe_l1_coeff': 1e-08,
    'rankmixer_remoe_l1_multiplier': 1.2,
    'rankmixer_norm': 'ln',
    'rankmixer_tokenize': 'semantic',
    'rankmixer_group_tokens': '',
    'rankmixer_seq_per_token': 0,
    'use_senet': 1,
    'use_input_bn': 1,
    'seq_dual_din_domains': '',
    'dense_per_fid': False,
    'user_dense_groups': '',
    'item_dense_groups': '',
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRRankMixer``.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``feature_specs = [(vocab_size, offset, length), ...]`` in the
    order of ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse a string like ``'seq_a:256,seq_b:256,...'`` into a dict."""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory.

    Returns an empty dict (which triggers fallback resolution) if the file is
    not present.
    """
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults. "
        f"Shape mismatch may occur if training used non-default hyperparameters.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from ``train_config``; missing keys fall
    back to ``_FALLBACK_MODEL_CFG``.

    Special handling for ``num_time_buckets``: it is not exposed on the CLI
    as an independent hyperparameter; the bucket count is uniquely determined
    by the length of ``dataset.BUCKET_BOUNDARIES``. Resolution order:

      1) ``train_config`` contains ``num_time_buckets`` directly (legacy ckpt)
         -> use that value;
      2) ``train_config`` contains ``use_time_buckets`` (new-style training)
         -> derive as ``NUM_TIME_BUCKETS`` or ``0``;
      3) neither is present -> fall back to ``_FALLBACK_MODEL_CFG[...]``.
    """
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == 'num_time_buckets':
            if 'num_time_buckets' in train_config:
                cfg[key] = train_config['num_time_buckets']
            elif 'use_time_buckets' in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config['use_time_buckets'] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    f"train_config missing both 'num_time_buckets' and 'use_time_buckets', "
                    f"using fallback = {cfg[key]}")
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}")
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    device: str = 'cpu',
) -> nn.Module:
    """Construct the RankMixer model (``PCVRRankMixer``) from the dataset
    schema and a resolved ``model_cfg`` dict.

    Args:
        dataset: a ``PCVRParquetDataset`` providing the feature schema.
        model_cfg: resolved model hyperparameters, typically the output of
            ``resolve_model_cfg``.
        device: torch device.
    """
    # Feature specs.
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    # Resolve the (id, weight)-pair pooling map from the saved CSV flag. Popped
    # from model_cfg because the models take the resolved list/map, not the CSV.
    model_cfg = dict(model_cfg)
    weighted_pair_pool_fids = model_cfg.pop('weighted_pair_pool_fids', '')
    user_weighted_pairs = build_weighted_pairs(
        dataset.user_int_schema, dataset.user_dense_schema, weighted_pair_pool_fids)
    if user_weighted_pairs:
        logging.info(f"Inference weighted-pair pooling fids: {weighted_pair_pool_fids!r} "
                     f"-> {len(user_weighted_pairs)} pair(s)")

    # SeFCN backbone. Pop the SeFCN hyperparams from model_cfg (the rest are the
    # shared base args the backbone's __init__ accepts and del's as needed).
    pair_pool = model_cfg.pop('pair_pool', 'mean')
    concat_senet_dims = model_cfg.pop('concat_senet_dims', '1024,256')
    concat_mlp_dims = model_cfg.pop('concat_mlp_dims', '1024,512,256,128')
    two_stream_mlp = model_cfg.pop('two_stream_mlp', 0)
    two_stream_dims = model_cfg.pop('two_stream_dims', '512,128')
    two_stream_input = model_cfg.pop('two_stream_input', 'senet')
    two_stream_fusion = model_cfg.pop('two_stream_fusion', 'sum')
    bilinear_groups = model_cfg.pop('bilinear_groups', 8)
    rankmixer_tokens = model_cfg.pop('rankmixer_tokens', 16)
    rankmixer_dim = model_cfg.pop('rankmixer_dim', 128)
    rankmixer_heads = model_cfg.pop('rankmixer_heads', 0)
    rankmixer_layers = model_cfg.pop('rankmixer_layers', 2)
    rankmixer_expansion = model_cfg.pop('rankmixer_expansion', 4)
    rankmixer_pool = model_cfg.pop('rankmixer_pool', 'mean')
    rankmixer_ffn = model_cfg.pop('rankmixer_ffn', 'pswiglu')
    rankmixer_down_init_gain = model_cfg.pop('rankmixer_down_init_gain', 0.01)
    rankmixer_moe_experts = model_cfg.pop('rankmixer_moe_experts', 4)
    rankmixer_moe_top_k = model_cfg.pop('rankmixer_moe_top_k', 2)
    rankmixer_remoe_l1_coeff = model_cfg.pop('rankmixer_remoe_l1_coeff', 1e-8)
    rankmixer_remoe_l1_multiplier = model_cfg.pop('rankmixer_remoe_l1_multiplier', 1.2)
    rankmixer_norm = model_cfg.pop('rankmixer_norm', 'ln')
    rankmixer_tokenize = model_cfg.pop('rankmixer_tokenize', 'semantic')
    rankmixer_group_tokens = model_cfg.pop('rankmixer_group_tokens', '')
    rankmixer_seq_per_token = model_cfg.pop('rankmixer_seq_per_token', 0)
    use_senet = model_cfg.pop('use_senet', 1)
    use_input_bn = model_cfg.pop('use_input_bn', 1)
    seq_dual_din_domains = model_cfg.pop('seq_dual_din_domains', '')
    dense_per_fid = model_cfg.pop('dense_per_fid', False)
    user_dense_groups = model_cfg.pop('user_dense_groups', '')
    item_dense_groups = model_cfg.pop('item_dense_groups', '')

    user_dense_dim = dataset.user_dense_schema.total_dim

    # Rebuild the identical pair-offset map + kept-dense complement as train.py.
    pair_weighted_dense_offsets = {
        int(fid_idx): int(d_off) for fid_idx, d_off, _l in (user_weighted_pairs or [])
    }
    paired_ranges = sorted(
        (int(d_off), int(d_off) + int(d_len))
        for _, d_off, d_len in (user_weighted_pairs or []))
    user_dense_keep_offsets: List[Tuple[int, int]] = []
    cursor = 0
    for s, e in paired_ranges:
        if s > cursor:
            user_dense_keep_offsets.append((cursor, s - cursor))
        cursor = max(cursor, e)
    if cursor < user_dense_dim:
        user_dense_keep_offsets.append((cursor, user_dense_dim - cursor))

    # Per-fid layout for --dense_per_fid (rebuild identically to train.py): kept
    # user_dense fields (pair fids removed) + full item_dense schema, ordered.
    _pair_fids = {
        int(t) for t in str(weighted_pair_pool_fids).split(',') if str(t).strip()
    }
    user_dense_fid_layout = [
        (fid, length)
        for fid, _off, length in dataset.user_dense_schema.entries
        if fid not in _pair_fids
    ]
    item_dense_fid_layout = [
        (fid, length)
        for fid, _off, length in dataset.item_dense_schema.entries
    ]

    def _parse_int_csv(s):
        vals = [int(t) for t in str(s).split(',') if str(t).strip()]
        return vals or None

    logging.info(
        f"Building PCVRRankMixer "
        f"(use_senet={bool(int(use_senet))}, use_input_bn={bool(int(use_input_bn))}) "
        f"with cfg: {model_cfg}")
    model = PCVRRankMixer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=user_dense_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        pair_weighted_dense_offsets=pair_weighted_dense_offsets,
        pair_pool=pair_pool,
        user_dense_keep_offsets=user_dense_keep_offsets,
        concat_senet_dims=_parse_int_csv(concat_senet_dims),
        concat_mlp_dims=_parse_int_csv(concat_mlp_dims),
        two_stream_mlp=bool(int(two_stream_mlp)),
        two_stream_dims=_parse_int_csv(two_stream_dims),
        two_stream_input=two_stream_input,
        two_stream_fusion=two_stream_fusion,
        bilinear_groups=int(bilinear_groups),
        rankmixer_tokens=int(rankmixer_tokens),
        rankmixer_dim=int(rankmixer_dim),
        rankmixer_heads=int(rankmixer_heads),
        rankmixer_layers=int(rankmixer_layers),
        rankmixer_expansion=int(rankmixer_expansion),
        rankmixer_pool=rankmixer_pool,
        rankmixer_ffn=rankmixer_ffn,
        rankmixer_down_init_gain=float(rankmixer_down_init_gain),
        rankmixer_moe_experts=int(rankmixer_moe_experts),
        rankmixer_moe_top_k=int(rankmixer_moe_top_k),
        rankmixer_remoe_l1_coeff=float(rankmixer_remoe_l1_coeff),
        rankmixer_remoe_l1_multiplier=float(rankmixer_remoe_l1_multiplier),
        rankmixer_norm=rankmixer_norm,
        rankmixer_tokenize=rankmixer_tokenize,
        rankmixer_group_tokens=_parse_int_csv(rankmixer_group_tokens),
        rankmixer_seq_per_token=bool(int(rankmixer_seq_per_token)),
        seq_dual_din_domains=[s.strip() for s in str(seq_dual_din_domains).split(',') if s.strip()] or None,
        dense_per_fid=bool(dense_per_fid),
        user_dense_fid_layout=user_dense_fid_layout,
        item_dense_fid_layout=item_dense_fid_layout,
        user_dense_groups=str(user_dense_groups) or None,
        item_dense_groups=str(item_dense_groups) or None,
        use_senet=bool(int(use_senet)),
        use_input_bn=bool(int(use_input_bn)),
        **model_cfg,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load ``state_dict``; any missing/unexpected key fails fast
    with a diagnostic message.
    """
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters.")
        raise e


def get_ckpt_path() -> Optional[str]:
    """Locate the first ``*.pt`` file inside the directory pointed at by
    ``$MODEL_OUTPUT_PATH``. Returns ``None`` if no checkpoint is found.
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ``ModelInput``, handling dynamic seq domains."""
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        user_dense_feats_raw=device_batch.get('user_dense_feats_raw'),
    )


def _autocast_ctx(device: str, use_amp: bool, amp_dtype: str):
    """Return inference autocast context from train_config runtime flags."""
    if not use_amp:
        return contextlib.nullcontext()
    if amp_dtype not in ('bfloat16', 'float16'):
        logging.warning("Unknown amp_dtype=%r; falling back to bfloat16", amp_dtype)
        amp_dtype = 'bfloat16'
    dtype = torch.bfloat16 if amp_dtype == 'bfloat16' else torch.float16
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    return torch.autocast(device_type=device_type, dtype=dtype)


def main() -> None:
    # ---- Read environment variables ----
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Schema: prefer the one from model_dir (to exactly match training);
    #      fall back to the one in data_dir if missing. ----
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json (single source of truth for all hyperparams) ----
    train_config = load_train_config(model_dir)
    configure_sdp_kernels(
        enable_flash=bool(train_config.get('enable_flash_sdp', True)),
        enable_mem_efficient=bool(train_config.get('enable_mem_efficient_sdp', True)),
        enable_math=bool(train_config.get('enable_math_sdp', True)),
    )
    infer_use_amp = bool(train_config.get('use_amp', False))
    infer_amp_dtype = str(train_config.get('amp_dtype', 'bfloat16'))
    logging.info(f"Inference AMP: use_amp={infer_use_amp}, amp_dtype={infer_amp_dtype}")

    # ---- Parse seq_max_lens ----
    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- Data loading: reuse batch_size / num_workers from training config ----
    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))

    # Long-tail dense transform: apply the SAME transform as training, reading
    # dense_stats.json from the checkpoint dir (trainer.py copies it there).
    longtail_transform = str(train_config.get('longtail_transform', 'none'))
    dense_stats_path = None
    if longtail_transform != 'none':
        dense_stats_path = os.path.join(model_dir, 'dense_stats.json')
        if not os.path.exists(dense_stats_path):
            raise FileNotFoundError(
                f"longtail_transform={longtail_transform!r} but dense_stats.json "
                f"is missing from MODEL_OUTPUT_PATH={model_dir!r}. It must be "
                f"copied into the checkpoint dir at training time.")
        logging.info(f"Inference long-tail transform: {longtail_transform} "
                     f"(stats={dense_stats_path})")

    # schema_dim_caps: reapply the identical per-fid truncation used at training
    # so the dataset layout (and the weighted-pair offsets / model shapes derived
    # from it) match the checkpoint exactly.
    schema_dim_caps = str(train_config.get('schema_dim_caps', '') or '')
    if schema_dim_caps:
        logging.info(f"Inference schema_dim_caps: {schema_dim_caps!r}")

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
        longtail_transform=longtail_transform,
        dense_stats_path=dense_stats_path,
        schema_dim_caps=schema_dim_caps,
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ---- Build model: every structural hyperparameter is resolved from train_config ----
    model_cfg = resolve_model_cfg(train_config)

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        device=device,
    )

    # ---- Strictly load weights ----
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}. "
            "This typically means the training job wrote only the sidecar "
            "files (schema.json / train_config.json) for this step but did "
            "not persist model.pt — a symptom of a race between "
            "_remove_old_best_dirs and EarlyStopping.save_checkpoint."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )

    all_probs = []
    all_user_ids = []
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])

            with _autocast_ctx(device, infer_use_amp, infer_amp_dtype):
                logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1).float()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"  Processed {(batch_idx + 1) * batch_size} samples")

    logging.info(f"Inference complete: {len(all_probs)} predictions")

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
