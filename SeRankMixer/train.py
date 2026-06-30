"""PCVRRankMixer training entry point (self-contained baseline).

Supports single-GPU and multi-GPU DDP training (via torchrun).

Usage:
    # Single GPU
    python train.py [--num_epochs 10] [--batch_size 256] ...
    # Multi-GPU
    torchrun --nproc_per_node=N train.py [--num_epochs 10] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist

from utils import set_seed, EarlyStopping, create_logger, configure_sdp_kernels
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS, parse_schema_dim_caps
from model import PCVRRankMixer
from trainer import PCVRRankingTrainer
from build_dense_stats import (
    build_dense_stats_json, build_weighted_pairs,
    _DEFAULT_LONGTAIL_FIDS)

# Default dense fids that get --longtail_transform (signed_log1p + per-dim /std).
# Sourced from build_dense_stats so there is one whitelist, but exposed here as
# the --longtail_fids argparse default so it shows up in --help / train_config
# and can be overridden from an experiment script without editing code.
_DEFAULT_LONGTAIL_FIDS_STR = ','.join(str(f) for f in _DEFAULT_LONGTAIL_FIDS)


# ─────────────────────────── DDP Helpers ──────────────────────────────────


def setup_ddp():
    """Initialize DDP. Detects torchrun environment variables.

    Returns:
        (rank, local_rank, world_size). Non-DDP returns (0, 0, 1).
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # Bind this rank to its GPU BEFORE init_process_group, and pass
        # device_id so NCCL eagerly maps rank -> GPU. This silences the
        # "device used by this process is currently unknown" / "No device id
        # provided" warnings and removes the potential rank<->GPU mismatch hang.
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            device_id=torch.device('cuda', local_rank),
        )
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


# ─────────────────────────────────────────────────────────────────────────


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRRankMixer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=2,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='Save an extra checkpoint after each epoch '
                             'validation, in addition to the best checkpoint.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--split_by_time', action='store_true',
                        help='Split train/valid by timestamp instead of Row Group position')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--log_every_n_steps', type=int, default=100,
                        help='Cadence (in steps) for printing train progress and '
                             'writing the TensorBoard Loss/train scalar. Each log '
                             'forces one host sync (loss.item()), so smaller = finer '
                             'curve but more syncs. 1 = per-step (the old behavior).')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')
    parser.add_argument('--seq_dual_din_domains', type=str, default='',
                        help='CSV of seq domains to apply DUAL-DIN to: the '
                             'newest-first head[:L] slice is split into a recent '
                             'half [0:L/2] and an older half [L/2:L], each pooled '
                             'by its own DIN, then concat + projected to d_model '
                             '(T / rankmixer_dim unchanged). Set the domain L via '
                             "--seq_max_lens (e.g. seq_d:1024 -> half=512). '' = off.")
    parser.add_argument('--dense_per_fid', action='store_true',
                        help='Per-fid dense re-tokenization: expand user_dense / '
                             'item_dense from 2 blob tokens into one RankMixer '
                             'token PER dense field (or per --user/item_dense_groups '
                             'cluster), so token-mixing sees each dense field as its '
                             'own token. rankmixer_group_tokens then supplies ONLY '
                             'the head counts [user_emb,item_emb,seq...]; each dense '
                             'sub-group = 1 token. OFF => byte-identical to base. '
                             'Note H=T => total tokens must divide rankmixer_dim.')
    parser.add_argument('--user_dense_groups', type=str, default='',
                        help="user_dense per-fid grouping spec for --dense_per_fid: "
                             "'|' separates sub-groups, ',' merges fids into one "
                             "token, e.g. '61|87|89,90,91,118|120|121|123|130|131|132'. "
                             "Must cover the kept user_dense fids in order. '' = one "
                             "token per kept fid.")
    parser.add_argument('--item_dense_groups', type=str, default='',
                        help="item_dense per-fid grouping spec for --dense_per_fid "
                             "(same format), e.g. '124|127,128|129'. '' = one token "
                             "per item_dense fid.")

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    # ── SeRankMixer backbone hyperparameters ──
    # This is the SeRankMixer line: per-fid pooled embeddings + DIN-pooled
    # sequences -> input BN -> ChannelSENET -> RankMixer cross -> MLP, on the
    # baseline_rd2 rd2 pipeline. The HyFormer line lives in baseline_rd2/.
    # Data flags (weighted-pair / longtail) are shared.
    parser.add_argument('--pair_pool', type=str, default='mean',
                        choices=['mean', 'sum'],
                        help='(din_concat_senet_dcn) pooling for (id, weight)-pair '
                             'fids in the field tokenizer. mean = weighted average.')
    parser.add_argument('--concat_senet_dims', type=str, default='1024,256',
                        help='(din_concat_senet_dcn) ChannelSENET reduction dims, '
                             'CSV. Empty = SENET off.')
    parser.add_argument('--concat_mlp_dims', type=str, default='1024,512,256,128',
                        help='(din_concat_senet_dcn) final MLP hidden dims, CSV.')
    parser.add_argument('--two_stream_mlp', type=int, default=0, choices=[0, 1],
                        help='Enable a Wide&Deep-style side MLP logit branch on the '
                             'concat feature vector, parallel to RankMixer and ADDED '
                             'at the logit. Zero-initialized, so training starts '
                             'equivalent to the base (OFF = byte-identical).')
    parser.add_argument('--two_stream_dims', type=str, default='512,128',
                        help='Hidden dims for the --two_stream_mlp side branch, CSV.')
    parser.add_argument('--two_stream_input', type=str, default='senet',
                        choices=['raw', 'bn', 'senet'],
                        help='Which concat representation feeds the side MLP: '
                             'raw, post-input-BN, or post-SENET.')
    parser.add_argument('--two_stream_fusion', type=str, default='sum',
                        choices=['sum', 'bilinear'],
                        help="How the two streams combine. 'sum' = deep_logit + "
                             "wide_logit (default, current best). 'bilinear' = "
                             "ALSO add FinalMLP's group-wise bilinear interaction "
                             "of the two stream hidden vectors (arXiv 2304.00902) "
                             "as a zero-init residual (final = deep + wide + "
                             "bilinear(deep_h, wide_h)); sum-equivalent at init. "
                             "REQUIRES --two_stream_mlp 1.")
    parser.add_argument('--bilinear_groups', type=int, default=8,
                        help='(--two_stream_fusion bilinear) number of group-wise '
                             'subspaces for the bilinear weight; must divide BOTH '
                             'stream hidden dims (deep=concat_mlp_dims[-1], '
                             'wide=two_stream_dims[-1]). Larger = lower-rank / '
                             'fewer params / stronger regularization.')
    # ── RankMixer cross (cross_arch=rankmixer; paper arXiv 2507.15551) ──
    parser.add_argument('--rankmixer_tokens', type=int, default=16,
                        help='(cross_arch rankmixer, CONTIGUOUS tokenize only) '
                             'number of equal slices T the post-SENET vector is '
                             'split into (Eq.2). Ignored by semantic tokenize, '
                             'which uses --rankmixer_group_tokens.')
    parser.add_argument('--rankmixer_dim', type=int, default=128,
                        help='(cross_arch rankmixer) per-token dimension D. Must '
                             'be divisible by the head count.')
    parser.add_argument('--rankmixer_heads', type=int, default=0,
                        help='(cross_arch rankmixer) token-mixing heads H. '
                             '0 = use H=T (paper default, shape-preserving).')
    parser.add_argument('--rankmixer_layers', type=int, default=2,
                        help='(cross_arch rankmixer) number of stacked '
                             'RankMixer blocks L.')
    parser.add_argument('--rankmixer_expansion', type=int, default=4,
                        help='(cross_arch rankmixer) PFFN hidden expansion k '
                             '(hidden = k*D).')
    parser.add_argument('--rankmixer_pool', type=str, default='mean',
                        choices=['mean', 'flatten'],
                        help='(cross_arch rankmixer) final aggregation over '
                             'tokens. mean = paper mean-pool -> D; flatten -> T*D.')
    parser.add_argument('--rankmixer_ffn', type=str, default='pswiglu',
                        choices=['pswiglu', 'remoe'],
                        help='(cross_arch rankmixer) block FFN. pswiglu = current '
                             'clean baseline (0.823886); remoe = per-token ReMoE '
                             'with a single ReLU router (ICLR25 arXiv 2412.14711).')
    parser.add_argument('--rankmixer_down_init_gain', type=float, default=0.01,
                        help='(cross_arch rankmixer, pswiglu/remoe) xavier gain for '
                             'SwiGLU W_down (Down-Matrix Small Init, paper 0.01 = '
                             'the 0.822706 value). Small => the FFN starts ~0 '
                             '(approx identity) but its W_up/W_gate grads are also '
                             'suppressed at init; set 1.0 for standard xavier.')
    parser.add_argument('--rankmixer_moe_experts', type=int, default=4,
                        help='(cross_arch rankmixer, ffn=remoe) number of SwiGLU '
                             'experts (per-token expert banks). Ignored by pswiglu.')
    parser.add_argument('--rankmixer_moe_top_k', type=int, default=2,
                        help='(cross_arch rankmixer, ffn=remoe) target average '
                             'active experts/token (ReMoE target_active), held by '
                             'the adaptive L1 routing reg. Ignored by pswiglu.')
    parser.add_argument('--rankmixer_remoe_l1_coeff', type=float, default=1e-8,
                        help='(cross_arch rankmixer, ffn=remoe) initial coefficient '
                             'for ReLU infer-router L1 sparsity regularization.')
    parser.add_argument('--rankmixer_remoe_l1_multiplier', type=float, default=1.2,
                        help='(cross_arch rankmixer, ffn=remoe) per-step multiplier '
                             'used to adapt the L1 coefficient toward the target '
                             'sparsity implied by --rankmixer_moe_top_k.')
    parser.add_argument('--rankmixer_norm', type=str, default='ln',
                        choices=['ln', 'rmsnorm'],
                        help='(cross_arch rankmixer) norm used in BOTH block norms '
                             '(post-norm: after token-mixing and after the FFN). '
                             'ln = LayerNorm (paper default = the 0.822706 value); '
                             'rmsnorm = RMSNorm (no mean-centering / no bias, learned '
                             'RMS gain only). Single-variable ablation off 0.822706.')
    parser.add_argument('--rankmixer_tokenize', type=str, default='semantic',
                        choices=['semantic', 'contiguous'],
                        help='(cross_arch rankmixer) tokenization. semantic = '
                             'split at feature-group boundaries (user_emb/item_emb/'
                             'seq/user_dense/item_dense), each group its own Proj '
                             '-> tokens (paper domain-knowledge grouping); '
                             'contiguous = equal slices of the flat vector (Eq.2).')
    parser.add_argument('--rankmixer_group_tokens', type=str, default='',
                        help='(cross_arch rankmixer, semantic) REQUIRED per-group '
                             'token counts as a CSV of 5 ints in the order '
                             'user_emb,item_emb,seq,user_dense,item_dense '
                             '(e.g. "6,4,2,2,2"). Empty groups (dim 0, e.g. '
                             'item_dense when absent) are forced to 0. There is no '
                             'auto-split: semantic tokenize errors if this is '
                             'empty. Total T = sum; needs token_dim %% T == 0 when '
                             '--rankmixer_heads 0.')
    parser.add_argument('--rankmixer_seq_per_token', type=int, default=0,
                        choices=[0, 1],
                        help='(cross_arch rankmixer, semantic) 1 = give EACH seq '
                             'token its own Linear (one FFN per behavior domain: '
                             'Linear(d_model -> D) x n_seq_tokens) instead of '
                             'one shared Linear over the whole seq group '
                             '(contiguous per-domain seq slices). Default 0.')
    # NOTE: input-BN + ChannelSENET are CONFIRMED-OPTIMAL and FIXED (BN+SENET on).
    # They are no longer CLI-tunable — the model defaults bake them in. (Dropping
    # BN+SENET or using
    # plain PFFN both tested worse.)
    parser.add_argument('--compile_mode', type=str, default='none',
                        choices=['none', 'default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode for the model. '
                             'none = disabled. default = safest first speedup. '
                             'reduce-overhead uses CUDA graphs and prefers static '
                             'shapes. max-autotune has the longest warmup.')
    parser.add_argument('--float32_matmul_precision', type=str, default='high',
                        choices=['highest', 'high', 'medium'],
                        help='Precision policy for float32 matmul. high enables '
                             'TF32 tensor cores on NVIDIA GPUs where available; '
                             'highest keeps stricter fp32 behavior.')
    parser.set_defaults(use_amp=True)
    parser.add_argument('--use_amp', dest='use_amp', action='store_true',
                        help='Enable mixed-precision forward/loss via torch.autocast '
                             '(default on for rd2).')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                        help='Disable mixed precision and run forward/loss in fp32.')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16'],
                        help='AMP autocast dtype. bfloat16 is recommended on H20.')
    parser.set_defaults(
        enable_flash_sdp=True,
        enable_mem_efficient_sdp=True,
        enable_math_sdp=True,
    )
    parser.add_argument('--enable_flash_sdp', dest='enable_flash_sdp',
                        action='store_true',
                        help='Allow Flash Attention SDPA kernels (default on).')
    parser.add_argument('--disable_flash_sdp', dest='enable_flash_sdp',
                        action='store_false',
                        help='Disallow Flash Attention SDPA kernels.')
    parser.add_argument('--enable_mem_efficient_sdp', dest='enable_mem_efficient_sdp',
                        action='store_true',
                        help='Allow memory-efficient SDPA kernels (default on).')
    parser.add_argument('--disable_mem_efficient_sdp', dest='enable_mem_efficient_sdp',
                        action='store_false',
                        help='Disallow memory-efficient SDPA kernels.')
    parser.add_argument('--enable_math_sdp', dest='enable_math_sdp',
                        action='store_true',
                        help='Allow math SDPA fallback kernels (default on).')
    parser.add_argument('--disable_math_sdp', dest='enable_math_sdp',
                        action='store_false',
                        help='Disallow math SDPA fallback kernels. Risky if Flash '
                             'cannot handle a particular mask/layout.')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce',
                        choices=['bce', 'bce_pos_weight', 'focal'],
                        help='Loss type: bce = BCEWithLogits, '
                             'bce_pos_weight = BCEWithLogits with positive-class '
                             'weighting (--pos_weight), focal = Focal Loss.')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--pos_weight', type=float, default=1.0,
                        help='Positive-class BCE weight for '
                             '--loss_type bce_pos_weight '
                             '(multiplies the loss of positive samples; '
                             '1.0 = no-op).')
    parser.add_argument('--time_decay_alpha', type=float, default=0.0,
                        help='Exponential time-decay sample weight: '
                             'weight = exp(-alpha * (t_ref - sample_ts) / 86400). '
                             't_ref is auto-detected as the max timestamp in '
                             'the parquet files. alpha=0 disables time decay.')
    parser.add_argument('--use_sam', action='store_true', default=False,
                        help='Enable dense-only SAM. Each training step runs two '
                             'forward/backward passes and perturbs AdamW-managed '
                             'dense parameters only.')
    parser.add_argument('--sam_rho', type=float, default=0.05,
                        help='SAM neighborhood radius rho; only used with --use_sam.')
    parser.add_argument('--sam_start_epoch', type=int, default=1,
                        help='Epoch at which SAM activates (1 = from the start = '
                             'current behavior). Set 2 to let epoch 1 train with '
                             'plain Muon and begin the flat-minimum ascent only '
                             'once the tokenizer / dense per-fid have aligned — '
                             'mirrors the SWA window (which also starts at epoch '
                             '2). Only used with --use_sam.')
    parser.add_argument('--sam_matrix_only', action='store_true', default=False,
                        help='Restrict the SAM ascent step to weight MATRICES '
                             '(the Muon-routed set: ndim>=2 with both trailing '
                             'dims >1), excluding norm/BN/LN/RMSNorm scale, biases '
                             'and scalar/gate params. Concentrates the sharpness '
                             'regularization on the interaction matrices and '
                             'leaves BN/calibration params un-perturbed. Only '
                             'used with --use_sam.')
    parser.add_argument('--use_swa', action='store_true', default=False,
                        help='Enable dense-only Stochastic Weight Averaging '
                             '(equal-weight running mean of snapshots). Per epoch '
                             '>= --swa_start_epoch it averages the weights, '
                             'recomputes BatchNorm stats, and saves a parallel '
                             '`.swa` checkpoint to A/B against the regular model.')
    parser.add_argument('--swa_start_epoch', type=int, default=2,
                        help='First epoch (1-indexed) to start folding SWA '
                             'snapshots; only used with --use_swa.')
    parser.add_argument('--swa_collect_every', type=int, default=0,
                        help='Fold an intra-epoch SWA snapshot every N optimizer '
                             'steps (0 = end-of-epoch only). Many snapshots from '
                             'the constant-LR trajectory = the faithful SWA '
                             'recipe; only used with --use_swa.')
    parser.add_argument('--swa_bn_batches', type=int, default=50,
                        help='Number of training batches used to recompute '
                             'BatchNorm running stats for the averaged weights '
                             'before each SWA eval/save; only used with --use_swa.')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'muon'],
                        help="Dense optimizer. 'adamw' = base (byte-identical). "
                             "'muon' = scaled Muon (Moonlight/Kimi) on the 2D "
                             'hidden matrices + AdamW on the rest (norms/biases/'
                             'head); sparse Adagrad path is unchanged either way.')
    parser.add_argument('--muon_lr', type=float, default=1e-4,
                        help='Muon LR for the 2D matrices. The scaled variant '
                             'RMS-matches AdamW so this lives in the AdamW range; '
                             'sweep e.g. 1e-4 / 3e-4 / 5e-4. Only used by '
                             '--optimizer muon.')
    parser.add_argument('--muon_momentum', type=float, default=0.95,
                        help='Muon Nesterov-momentum coefficient.')
    parser.add_argument('--muon_weight_decay', type=float, default=0.01,
                        help='Muon decoupled weight decay (matches the dense '
                             "AdamW default so it's a clean optimizer A/B).")
    parser.add_argument('--muon_ns_steps', type=int, default=5,
                        help='Newton-Schulz iteration steps for orthogonalization.')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart regularization to reduce '
                             'overfitting; which embeddings get reset is controlled by '
                             '--reinit_cardinality_threshold, default 0 = all)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold for the re-init strategy: '
                             'Embeddings whose vocab_size is STRICTLY GREATER than '
                             'this value are reset at each epoch end (from '
                             '--reinit_sparse_after_epoch onward). The check is '
                             'vocab_size > threshold, so 0 = reset ALL embeddings '
                             '(every vocab_size > 0) every epoch -- the most '
                             'aggressive cold restart, and the current default. To '
                             'reset nothing, set this at or above the largest '
                             'vocab_size. time_embedding and features dropped by '
                             '--emb_skip_threshold are never reset.')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    # (NS-groups tokenizer removed: it was del'd by the model and unused.)
    parser.add_argument('--weighted_pair_pool_fids', type=str, default='',
                        help='CSV of user fids whose multi-value embedding '
                             'pooling is weighted by their aligned user_dense '
                             '(id, weight) pair instead of mean pooling, e.g. '
                             '"62,63,64,65,66,118,121". A fid is used only if '
                             'present in both user_int and user_dense with equal '
                             'per-row length; those fids are also EXCLUDED from '
                             'the user_dense feature projection (weight-only). '
                             'Weights come from user_dense_feats_raw, so pair it '
                             'with --longtail_transform log1p_scale_only for '
                             'log-compressed weights. Empty (default) = pure mean '
                             'pooling (frozen base).')
    parser.add_argument('--longtail_transform', type=str, default='none',
                        choices=['none', 'log1p', 'log1p_scale_only'],
                        help='Dense long-tail transform on the --longtail_fids '
                             'whitelist. none (default) = raw (frozen base). '
                             'log1p = signed_log1p only (SeFCN-faithful: no dim '
                             'that feeds the dense projection is z-scored). '
                             'log1p_scale_only = signed_log1p + per-dim /std '
                             '(z-scores feature dims too; can amplify their '
                             'low-variance components, so prefer log1p). Both '
                             'are driven by --dense_stats_path; pretrained / '
                             'non-whitelist dims are left raw.')
    parser.add_argument('--dense_stats_path', type=str, default=None,
                        help='Path to a prebuilt dense_stats.json. When omitted '
                             'and --longtail_transform != none, train.py builds '
                             'it inline (rank 0) into the checkpoint dir and '
                             'copies it into each saved checkpoint so infer.py '
                             'reads the identical stats from MODEL_OUTPUT_PATH. '
                             'No download/commit of the JSON is needed.')
    parser.add_argument('--dense_stats_sample_rgs', type=int, default=800,
                        help='Row groups sampled for the inline dense_stats '
                             'build (per-dim std estimate). 0 = full scan. '
                             'Row groups are not time-ordered, so a few hundred '
                             'is a representative sample (~minute-scale).')
    parser.add_argument('--longtail_fids', type=str, default=_DEFAULT_LONGTAIL_FIDS_STR,
                        help='CSV of dense fids to apply --longtail_transform to '
                             '(signed_log1p + per-dim /std). Default is the '
                             'whitelist (user_dense 62,63,64,65,66,118,121,131,132 '
                             '+ item_dense 124,129), shown here so it can be edited '
                             'from an experiment script. This replaces the old '
                             'absmax>10 auto-selection (which wrongly pulled in '
                             'bounded fid 120, whose tiny std blew up x100 under '
                             '/std). Edit for ablations, e.g. add 130; empty string '
                             'disables the transform entirely (no dim selected). '
                             'Only used when --longtail_transform != none.')
    parser.add_argument('--schema_dim_caps', type=str, default='',
                        help='CSV of "fid:cap" pairs that SHRINK a feature\'s '
                             'per-row head-slice length, e.g. '
                             '"62:3,63:4,64:6,65:10,66:14,130:263". Applies to '
                             'multi-value int lists (newest-first, so the head '
                             'is kept) and dense vectors. For the verified '
                             '(id, weight) pairs (62-66) the head is the highest '
                             'weight, so truncation is near-lossless. An explicit '
                             'cap on 130 supersedes the built-in 259 override. '
                             'Adds no params (pooling is param-free); int caps '
                             'do not change any tensor shape, only 130/other '
                             'non-paired user_dense caps resize user_dense_proj. '
                             'Caps are saved to train_config.json so infer.py '
                             'reapplies the identical layout. Empty = no caps.')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    # ── DDP initialization ──
    rank, local_rank, world_size = setup_ddp()
    ddp_enabled = world_size > 1

    args = parse_args()

    # DDP: override device to local_rank.
    if ddp_enabled:
        args.device = f'cuda:{local_rank}'

    # Create output directories (only rank 0).
    if is_main_process():
        Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        if args.tf_events_dir:
            Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # DDP barrier: wait for rank 0 to finish creating directories.
    if ddp_enabled:
        dist.barrier()

    # Initialize logger and RNG.
    set_seed(args.seed + rank)  # Different seed per rank for data diversity.

    if is_main_process():
        create_logger(os.path.join(args.log_dir, 'train.log'))
    else:
        logging.basicConfig(level=logging.WARNING)

    logging.info(f"DDP: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    logging.info(f"Args: {vars(args)}")
    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    logging.info(
        "float32 matmul precision set to %r",
        args.float32_matmul_precision,
    )
    configure_sdp_kernels(
        enable_flash=args.enable_flash_sdp,
        enable_mem_efficient=args.enable_mem_efficient_sdp,
        enable_math=args.enable_math_sdp,
    )

    writer = None
    if is_main_process() and args.tf_events_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse schema_dim_caps once; reused by the inline dense_stats build (so the
    # per-dim std arrays match the capped runtime layout) and the data loaders.
    schema_dim_caps_dict = parse_schema_dim_caps(args.schema_dim_caps)
    if schema_dim_caps_dict:
        logging.info(f"schema_dim_caps: {schema_dim_caps_dict}")

    # ---- dense_stats.json resolution (long-tail transform) ----
    # Platform can't download files, so we don't commit/upload the JSON. Use an
    # explicitly-provided file if it exists, otherwise build it inline on rank 0
    # into the checkpoint dir. trainer.py copies it into each saved checkpoint so
    # infer.py reads the identical stats from MODEL_OUTPUT_PATH.
    if args.longtail_transform != 'none':
        if args.dense_stats_path and os.path.exists(args.dense_stats_path):
            logging.info(f"Using provided dense_stats.json: {args.dense_stats_path}")
        else:
            built_path = os.path.join(args.ckpt_dir, 'dense_stats.json')
            if is_main_process():
                sample = args.dense_stats_sample_rgs
                logging.info(
                    f"Building dense_stats.json inline -> {built_path} "
                    f"(sample_rgs={sample or 'full'})")
                # The arg is authoritative: parsed list (possibly empty to
                # disable). Never None, so build_dense_stats does not silently
                # fall back to its own constant.
                longtail_fids = [int(t) for t in args.longtail_fids.split(',') if t.strip()]
                build_dense_stats_json(
                    data_dir=args.data_dir,
                    schema_path=schema_path,
                    output_path=built_path,
                    sample_rgs=sample if sample and sample > 0 else None,
                    longtail_fids=longtail_fids,
                    schema_dim_caps=schema_dim_caps_dict,
                )
            if ddp_enabled and dist.is_initialized():
                dist.barrier()  # other ranks wait for rank-0 to finish writing
            args.dense_stats_path = built_path

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed + rank,
        seq_max_lens=seq_max_lens,
        ddp_rank=rank,
        ddp_world_size=world_size,
        split_by_time=args.split_by_time,
        longtail_transform=args.longtail_transform,
        dense_stats_path=args.dense_stats_path,
        schema_dim_caps=args.schema_dim_caps,
        time_decay_alpha=args.time_decay_alpha,
    )

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    user_weighted_pairs = build_weighted_pairs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_dense_schema,
        args.weighted_pair_pool_fids)
    if user_weighted_pairs and args.longtail_transform == 'none':
        logging.warning(
            "weighted_pair_pool_fids is set but longtail_transform=none: pooling "
            "weights will be the RAW (un-log1p'd) dense counts, so the largest "
            "count dominates each pool. Use --longtail_transform log1p_scale_only "
            "for log-compressed weights.")

    user_dense_dim = pcvr_dataset.user_dense_schema.total_dim
    # Constructor args for the model (PCVRRankMixer == PCVRDINConcatSENETDCN;
    # it del's a couple of legacy args it no longer uses, e.g. user_dense_dim).
    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": user_dense_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
    }

    # SeFCN backbone. Reuse the SAME weighted-pair config as the HyFormer line:
    # pair fids -> {fid_idx: dense_offset} pooling weights, and the kept user
    # dense = the complement of the paired (weight-only) ranges, identical to
    # what the HyFormer line feeds its user_dense projection.
    pair_weighted_dense_offsets = {
        int(fid_idx): int(d_off)
        for fid_idx, d_off, _d_len in (user_weighted_pairs or [])
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

    # Per-fid layout for --dense_per_fid: the kept user_dense fields (pair fids
    # removed, so the fid order matches _slice_kept_dense's memory layout) and
    # the full item_dense schema, each as ordered [(fid, capped_dim), ...].
    _pair_fids = {
        int(t) for t in args.weighted_pair_pool_fids.split(',') if t.strip()
    }
    user_dense_fid_layout = [
        (fid, length)
        for fid, _off, length in pcvr_dataset.user_dense_schema.entries
        if fid not in _pair_fids
    ]
    item_dense_fid_layout = [
        (fid, length)
        for fid, _off, length in pcvr_dataset.item_dense_schema.entries
    ]

    def _parse_int_csv(s: str) -> "Optional[List[int]]":
        vals = [int(t) for t in s.split(',') if t.strip()]
        return vals or None

    model = PCVRRankMixer(
        **model_args,
        pair_weighted_dense_offsets=pair_weighted_dense_offsets,
        pair_pool=args.pair_pool,
        user_dense_keep_offsets=user_dense_keep_offsets,
        concat_senet_dims=_parse_int_csv(args.concat_senet_dims),
        concat_mlp_dims=_parse_int_csv(args.concat_mlp_dims),
        two_stream_mlp=bool(args.two_stream_mlp),
        two_stream_dims=_parse_int_csv(args.two_stream_dims),
        two_stream_input=args.two_stream_input,
        two_stream_fusion=args.two_stream_fusion,
        bilinear_groups=args.bilinear_groups,
        rankmixer_tokens=args.rankmixer_tokens,
        rankmixer_dim=args.rankmixer_dim,
        rankmixer_heads=args.rankmixer_heads,
        rankmixer_layers=args.rankmixer_layers,
        rankmixer_expansion=args.rankmixer_expansion,
        rankmixer_pool=args.rankmixer_pool,
        rankmixer_ffn=args.rankmixer_ffn,
        rankmixer_down_init_gain=args.rankmixer_down_init_gain,
        rankmixer_moe_experts=args.rankmixer_moe_experts,
        rankmixer_moe_top_k=args.rankmixer_moe_top_k,
        rankmixer_remoe_l1_coeff=args.rankmixer_remoe_l1_coeff,
        rankmixer_remoe_l1_multiplier=args.rankmixer_remoe_l1_multiplier,
        rankmixer_norm=args.rankmixer_norm,
        rankmixer_tokenize=args.rankmixer_tokenize,
        rankmixer_group_tokens=_parse_int_csv(args.rankmixer_group_tokens),
        rankmixer_seq_per_token=bool(args.rankmixer_seq_per_token),
        seq_dual_din_domains=[s.strip() for s in args.seq_dual_din_domains.split(',') if s.strip()] or None,
        dense_per_fid=bool(args.dense_per_fid),
        user_dense_fid_layout=user_dense_fid_layout,
        item_dense_fid_layout=item_dense_fid_layout,
        user_dense_groups=args.user_dense_groups or None,
        item_dense_groups=args.item_dense_groups or None,
        # use_input_bn / use_senet are CONFIRMED-OPTIMAL and FIXED by the model
        # defaults (BN+SENET on) -- no longer CLI args.
    ).to(args.device)
    logging.info(
        f"Built PCVRRankMixer(use_senet={model.use_senet}, "
        f"use_input_bn={model.use_input_bn}, "
        f"rankmixer_seq_per_token={bool(args.rankmixer_seq_per_token)}, "
        f"rankmixer_ffn={args.rankmixer_ffn}, "
        f"moe_experts={args.rankmixer_moe_experts}, "
        f"moe_top_k={args.rankmixer_moe_top_k}): "
        f"pair_offsets={pair_weighted_dense_offsets}, "
        f"user_dense_keep_offsets={user_dense_keep_offsets}, input_dim={model.input_dim}")

    # Compile before DDP wrapping. Keep state_dict/load_state_dict routed to
    # the original module so checkpoints stay loadable by plain infer.py.
    if args.compile_mode != 'none':
        logging.info(f"Applying torch.compile(mode={args.compile_mode!r})")
        uncompiled_model = model
        model = torch.compile(model, mode=args.compile_mode)
        model.state_dict = uncompiled_model.state_dict
        model.load_state_dict = uncompiled_model.load_state_dict

    # ── DDP wrapping ──
    if ddp_enabled:
        # find_unused_parameters stays OFF: input_bn always runs before SENET, so
        # every built parameter gets a grad each step (no unused-param scan needed).
        find_unused = False
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=find_unused,
        )
        logging.info(
            f"Model wrapped with DDP on device cuda:{local_rank} "
            f"(find_unused_parameters={find_unused})")

    # Log model sizing info.
    raw_model = model.module if ddp_enabled else model
    sizing_model = getattr(raw_model, '_orig_mod', raw_model)
    num_ns = sizing_model.num_ns
    logging.info(f"PCVRRankMixer model created: num_ns={num_ns}, d_model={args.d_model}, "
                 f"rankmixer_dim={args.rankmixer_dim}, "
                 f"rankmixer_layers={args.rankmixer_layers}, "
                 f"rankmixer_ffn={args.rankmixer_ffn}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.rankmixer_layers,
        "head": args.rankmixer_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        pos_weight=args.pos_weight,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        optimizer=args.optimizer,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        muon_ns_steps=args.muon_ns_steps,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        dense_stats_path=args.dense_stats_path if args.longtail_transform != 'none' else None,
        eval_every_n_steps=args.eval_every_n_steps,
        log_every_n_steps=args.log_every_n_steps,
        train_config=vars(args),
        save_every_epoch=args.save_every_epoch,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        use_sam=args.use_sam,
        sam_rho=args.sam_rho,
        sam_start_epoch=args.sam_start_epoch,
        sam_matrix_only=args.sam_matrix_only,
        use_swa=args.use_swa,
        swa_start_epoch=args.swa_start_epoch,
        swa_collect_every=args.swa_collect_every,
        swa_bn_batches=args.swa_bn_batches,
    )

    trainer.train()

    if writer:
        writer.close()

    logging.info("Training complete!")
    cleanup_ddp()


if __name__ == "__main__":
    main()
