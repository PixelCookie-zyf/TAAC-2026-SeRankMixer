"""PCVRRankMixer pointwise trainer (binary-classification, AUC-monitored).

Supports single-GPU and multi-GPU DDP training.

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.algorithms.join import Join
from sklearn.metrics import roc_auc_score

from utils import (
    sigmoid_focal_loss,
    EarlyStopping,
    _format_duration,
    _format_rate,
    _format_train_progress_line,
)
from model import ModelInput


def is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def is_ddp() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


class _nullcontext:
    """Python 3.6 compatible null context manager."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def _weighted_mean(
    per_sample: torch.Tensor,
    sample_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    """Sample-weighted mean reduction; uniform mean when sample_weight is None."""
    if sample_weight is None:
        return per_sample.mean()
    w = sample_weight.view(-1).to(per_sample.dtype)
    return (per_sample.view(-1) * w).sum() / w.sum().clamp(min=1e-8)


class SWA:
    """Stochastic Weight Averaging: an *equal-weight* running mean over a subset
    of model parameters (Izmailov et al. 2018, arXiv 1803.05407).

    Unlike :class:`EMA`'s exponential decay, every collected snapshot gets the
    same weight ``1/n``. With a constant learning rate the SGD/Muon iterates
    orbit the basin rim and their mean lands in a flatter, better-generalizing
    point. Dense-only (sparse Embeddings excluded) -- mirrors :class:`EMA`
    exactly so the two are a clean A/B on the *averaging rule* alone.

    NOTE: the averaged weights need their BatchNorm running statistics
    recomputed (the per-snapshot stats are meaningless for the mean); the
    trainer does that before every SWA eval/save.
    """

    def __init__(
        self,
        model: nn.Module,
        exclude_ptrs: Optional[set] = None,
    ) -> None:
        exclude_ptrs = exclude_ptrs or set()
        self._items: List[Tuple[str, nn.Parameter]] = []
        self.avg: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.data_ptr() in exclude_ptrs:
                continue
            self._items.append((name, p))
            self.avg[name] = p.detach().clone()
        self.n: int = 0  # snapshots folded so far (avg is meaningful once n>=1)
        self._backup: Dict[str, torch.Tensor] = {}

    @property
    def num_params(self) -> int:
        return sum(a.numel() for a in self.avg.values())

    @torch.no_grad()
    def update(self) -> None:
        """Fold the model's current weights into the equal-weight running mean."""
        n1 = self.n + 1
        for name, p in self._items:
            # avg <- avg + (p - avg) / n1  == running arithmetic mean
            self.avg[name].add_(p.data - self.avg[name], alpha=1.0 / n1)
        self.n = n1

    @torch.no_grad()
    def apply_shadow(self) -> None:
        if self._backup:
            raise RuntimeError(
                "SWA.apply_shadow() called twice without restore() in between")
        for name, p in self._items:
            self._backup[name] = p.data.clone()
            p.data.copy_(self.avg[name])

    @torch.no_grad()
    def restore(self) -> None:
        if not self._backup:
            return
        for name, p in self._items:
            backup = self._backup.get(name)
            if backup is not None:
                p.data.copy_(backup)
        self._backup = {}


@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Orthogonalize the last two dims of ``G`` via the quintic Newton-Schulz.

    Supports a single matrix ``[m, n]`` OR a BATCH of matrices ``[..., m, n]``
    (e.g. the per-token FFN bank ``[T, d_in, d_out]`` = T independent matrices,
    each orthogonalized on its own). Returns an approximation of the orthogonal
    polar factor U V^T per matrix (singular values driven toward 1). Each matrix
    is L2-normalized (over its last two dims) first, so it is scale-invariant:
    the update magnitude is set downstream, not by ``G``. fp32 (runs --no_amp).
    Quintic coefficients from Keller Jordan's Muon.
    """
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = False
    if X.size(-2) > X.size(-1):  # keep rows <= cols for the X @ X^T products
        X = X.transpose(-2, -1)
        transposed = True
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X


class Muon(torch.optim.Optimizer):
    """Scaled Muon (Moonlight/Kimi variant) for 2D weight matrices.

    Update = NewtonSchulz(Nesterov-momentum gradient), RMS-matched to AdamW by
    ``0.2 * sqrt(max(d_out, d_in))`` so the per-element update magnitude is
    comparable to AdamW's — letting the matrices reuse an LR in the AdamW range
    and AdamW-style decoupled weight decay. Intended ONLY for hidden 2D
    matrices; biases / norms / the final head and the (sparse) embeddings are
    optimized elsewhere.
    """

    def __init__(self, params, lr=1e-4, momentum=0.95, weight_decay=0.01,
                 ns_steps=5, nesterov=True):
        if lr < 0:
            raise ValueError(f"Muon lr must be >= 0, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Muon momentum must be in [0,1), got {momentum}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            mom = group['momentum']
            wd = group['weight_decay']
            ns = group['ns_steps']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    raise RuntimeError(
                        f"Muon received a {g.ndim}D param (shape {tuple(p.shape)}); "
                        "route 1D/scalar params to AdamW.")
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if nesterov else buf
                # Orthogonalize each matrix (last two dims). A >2D param is a
                # BANK of independent matrices (per-token FFN [T, d_in, d_out]),
                # each orthogonalized + scaled on its own.
                ortho = _zeropower_via_newtonschulz5(upd, ns)
                # per-element RMS of ortho ~ 1/sqrt(max(m,n)); * 0.2*sqrt(max) -> ~0.2
                m, n = p.shape[-2], p.shape[-1]
                scale = 0.2 * (max(m, n) ** 0.5)
                if wd != 0:
                    p.mul_(1.0 - lr * wd)            # decoupled (AdamW-style) WD
                p.add_(ortho.to(p.dtype), alpha=-lr * scale)
        return loss


class _CombinedOptimizer:
    """Presents several optimizers (each owning a DISJOINT param set) as one.

    Used to drive the dense params with Muon (2D matrices) + AdamW (everything
    else) behind the single ``self.dense_optimizer`` handle the trainer expects.
    Only the methods the trainer touches are implemented (no state_dict — the
    trainer checkpoints model weights only, never optimizer state).
    """

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self):
        groups = []
        for o in self.optimizers:
            groups += o.param_groups
        return groups

    @property
    def state(self):
        merged = {}
        for o in self.optimizers:
            merged.update(o.state)
        return merged

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for o in self.optimizers:
            o.step()


class PCVRRankingTrainer:
    """PCVRRankMixer trainer for pointwise binary classification (supports DDP).

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        pos_weight: float = 1.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        optimizer: str = 'adamw',
        muon_lr: float = 1e-4,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.01,
        muon_ns_steps: int = 5,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        dense_stats_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        log_every_n_steps: int = 100,
        train_config: Optional[Dict[str, Any]] = None,
        save_every_epoch: bool = False,
        use_amp: bool = True,
        amp_dtype: str = 'bfloat16',
        use_sam: bool = False,
        sam_rho: float = 0.05,
        sam_start_epoch: int = 1,
        sam_matrix_only: bool = False,
        use_swa: bool = False,
        swa_start_epoch: int = 2,
        swa_collect_every: int = 0,
        swa_bn_batches: int = 50,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        self.schema_path: Optional[str] = schema_path
        self.dense_stats_path: Optional[str] = dense_stats_path

        # Get the raw model (unwrap DDP if needed).
        self.raw_model = model.module if hasattr(model, 'module') else model
        self._last_model_aux_loss: Optional[torch.Tensor] = None

        self.optimizer_name: str = optimizer
        if self.optimizer_name not in ('adamw', 'muon'):
            raise ValueError(
                f"optimizer must be 'adamw' or 'muon', got {optimizer!r}")
        # NOTE: --use_sam IS compatible with Muon. SAM is optimizer-agnostic — it
        # perturbs params along their gradient and recomputes the gradient at the
        # perturbed point; the optimizer (Muon+AdamW split) then steps with that
        # gradient unchanged. _sam_first/second_step iterate
        # dense_optimizer.param_groups, which already aggregates the Muon and
        # AdamW groups, so all dense params are perturbed. (Re-enabled to test
        # SAM+SWA: SWA was a big win, and SAM is the explicit flat-minimum method
        # the SWA result re-opened — see EXPERIMENTS.)

        # Dual optimizer: Adagrad for sparse Embeddings; dense params go to AdamW
        # OR (optimizer=muon) a split of Muon(2D hidden matrices) + AdamW(rest:
        # norms, biases, the [*,1] head). The sparse Adagrad path is IDENTICAL
        # in both cases (cold-restarted each epoch), so muon is a clean A/B on
        # the dense optimizer only.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(self.raw_model, 'get_sparse_params'):
            sparse_params = self.raw_model.get_sparse_params()
            dense_params = self.raw_model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            if self.optimizer_name == 'muon':
                # Muon: every weight MATRIX (2D Linears) and BANK of matrices
                # (>=3D: the per-token FFN [T, d_in, d_out] = T matrices), i.e.
                # ndim>=2 with both trailing dims > 1. Everything else stays on
                # AdamW: 1D norms/biases, scalars, and the [*,1]/[1,*] head.
                def _is_muon(p):
                    return p.ndim >= 2 and min(p.shape[-2], p.shape[-1]) > 1
                muon_params = [p for p in dense_params if _is_muon(p)]
                adamw_params = [p for p in dense_params if not _is_muon(p)]
                muon_count = sum(p.numel() for p in muon_params)
                adamw_count = sum(p.numel() for p in adamw_params)
                logging.info(
                    f"Dense params split for Muon: {len(muon_params)} matrices "
                    f"({muon_count:,} params) -> scaled Muon "
                    f"(lr={muon_lr}, momentum={muon_momentum}, "
                    f"wd={muon_weight_decay}, ns_steps={muon_ns_steps}); "
                    f"{len(adamw_params)} non-matrix tensors ({adamw_count:,} "
                    f"params: norms/biases/head) -> AdamW (lr={lr})")
                opts = []
                muon_opt = Muon(muon_params, lr=muon_lr, momentum=muon_momentum,
                                weight_decay=muon_weight_decay,
                                ns_steps=muon_ns_steps)
                opts.append(muon_opt)
                if adamw_params:
                    opts.append(torch.optim.AdamW(
                        adamw_params, lr=lr, betas=(0.9, 0.98)))
                self.dense_optimizer = _CombinedOptimizer(opts)
            else:
                logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
                self.dense_optimizer = torch.optim.AdamW(
                    dense_params, lr=lr, betas=(0.9, 0.98)
                )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                self.raw_model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.pos_weight: float = float(pos_weight)
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.log_every_n_steps: int = max(1, int(log_every_n_steps))
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.save_every_epoch: bool = save_every_epoch
        self.use_amp: bool = use_amp
        if amp_dtype not in ('bfloat16', 'float16'):
            raise ValueError(
                f"amp_dtype must be 'bfloat16' or 'float16', got {amp_dtype!r}"
            )
        self.amp_dtype: torch.dtype = (
            torch.bfloat16 if amp_dtype == 'bfloat16' else torch.float16
        )
        self._autocast_device_type: str = 'cuda' if 'cuda' in str(device) else 'cpu'
        if self.use_amp:
            logging.info(
                "AMP enabled: autocast(device_type=%r, dtype=%r); "
                "Embedding tables and optimizer state stay fp32",
                self._autocast_device_type,
                amp_dtype,
            )
        else:
            logging.info("AMP disabled: training/eval forward runs in fp32")

        # Stochastic Weight Averaging (dense-only, equal-weight; mirrors EMA's
        # scope so it is a clean A/B on the averaging rule). Snapshots are folded
        # from epoch >= swa_start_epoch (the model has reached the good basin),
        # optionally intra-epoch every swa_collect_every steps. BatchNorm running
        # stats are recomputed for the averaged weights over swa_bn_batches train
        # batches before each SWA eval/save. Train-only: model.pt is weights-only
        # (averaged weights + recomputed BN baked into buffers) -> infer.py is
        # unchanged. Mutually exclusive with the other weight-averagers.
        self.use_swa: bool = bool(use_swa)
        self.swa_start_epoch: int = int(swa_start_epoch)
        self.swa_collect_every: int = int(swa_collect_every)
        self.swa_bn_batches: int = int(swa_bn_batches)
        self.swa: Optional[SWA] = None
        # Replay buffer of raw CPU training batches for the SWA BatchNorm
        # recompute (filled during the first training epoch). Replaying these
        # avoids opening a SECOND DataLoader iterator during the SWA eval: the
        # train_loader uses persistent_workers + pin_memory, so re-iterating it
        # at the epoch boundary restarts its pin_memory thread right as
        # evaluate() spins up the valid loader's pin_memory thread -> two
        # concurrent cudaHostRegister calls on one device race ("CUDA error:
        # invalid argument", surfaced async at the next CUDA op = valid
        # pin_memory). Replaying cached batches removes that extra pin thread.
        self._swa_bn_cache: List[Any] = []
        if self.use_swa:
            if self.swa_start_epoch < 1:
                raise ValueError(
                    f"--swa_start_epoch must be >= 1, got {swa_start_epoch!r}")
            if self.swa_bn_batches < 1:
                raise ValueError(
                    "--swa_bn_batches must be >= 1: the averaged weights need "
                    "their BatchNorm running stats recomputed, else eval/infer "
                    f"use stale per-snapshot stats. Got {swa_bn_batches!r}")
            sparse_ptrs = set()
            if hasattr(self.raw_model, 'get_sparse_params'):
                sparse_ptrs = {p.data_ptr() for p in self.raw_model.get_sparse_params()}
            self.swa = SWA(self.raw_model, exclude_ptrs=sparse_ptrs)
            _mode = ('intra-epoch every %d steps + epoch boundary'
                     % self.swa_collect_every) if self.swa_collect_every > 0 \
                else 'end-of-epoch only'
            logging.info(
                f"SWA enabled (dense-only): start_epoch={self.swa_start_epoch}, "
                f"collect={_mode}, bn_recompute_batches={self.swa_bn_batches}; "
                f"tracking {len(self.swa.avg)} tensors / {self.swa.num_params:,} "
                f"params; sparse Embeddings ({len(sparse_ptrs)} tensors) excluded.")

        self.use_sam: bool = bool(use_sam)
        self.sam_rho: float = float(sam_rho)
        self.sam_start_epoch: int = int(sam_start_epoch)
        self.sam_matrix_only: bool = bool(sam_matrix_only)
        self._sam_perturbations: Dict[int, torch.Tensor] = {}
        # Ids of the weight MATRICES among the dense params (ndim>=2, both
        # trailing dims >1 — the same structural predicate Muon routes by,
        # see _is_muon above). Populated only when sam_matrix_only: the SAM
        # ascent step is then confined to this interaction-matrix subspace,
        # leaving norm/BN/LN/RMSNorm scale, biases and scalar/gate params
        # un-perturbed. Param identities are stable across training, so the
        # id() set built once here stays valid (same pattern as
        # _sam_perturbations keying by id(p)).
        self._sam_matrix_ids: set = set()
        if self.use_sam:
            if self.sam_rho <= 0:
                raise ValueError(
                    f"sam_rho must be positive when use_sam is True, got {sam_rho!r}")
            if self.sam_matrix_only:
                self._sam_matrix_ids = {
                    id(p)
                    for group in self.dense_optimizer.param_groups
                    for p in group['params']
                    if p.ndim >= 2 and min(p.shape[-2], p.shape[-1]) > 1
                }
            dense_n = sum(
                p.numel()
                for group in self.dense_optimizer.param_groups
                for p in group['params']
            )
            sam_n = sum(
                p.numel()
                for group in self.dense_optimizer.param_groups
                for p in group['params']
                if not self.sam_matrix_only or id(p) in self._sam_matrix_ids
            )
            logging.info(
                f"SAM enabled: rho={self.sam_rho}, start_epoch={self.sam_start_epoch}, "
                f"matrix_only={self.sam_matrix_only}, perturbing {sam_n:,}/{dense_n:,} "
                f"dense params ("
                f"{'Muon-routed matrices only' if self.sam_matrix_only else 'all dense; under Muon = the Muon+AdamW split'}"
                f"). Each SAM step runs two forward+backward passes (~2x train "
                f"cost) and is skipped before epoch {self.sam_start_epoch}. "
                f"Composes with SWA/Muon.")

        logging.info(f"PCVRRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"pos_weight={self.pos_weight}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"use_sam={self.use_sam}, use_swa={self.use_swa}")

    def _world_size(self) -> int:
        return dist.get_world_size() if is_ddp() else 1

    def _current_lr(self) -> float:
        if not self.dense_optimizer.param_groups:
            return 0.0
        return float(self.dense_optimizer.param_groups[0].get('lr', 0.0))

    def _gpu_memory_gb(self) -> Optional[float]:
        if not torch.cuda.is_available() or 'cuda' not in str(self.device):
            return None
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

    def _epoch_start_line(self, epoch: int) -> str:
        cfg = self.train_config or {}
        dataset = getattr(self.train_loader, 'dataset', None)
        batch_size = cfg.get('batch_size', getattr(dataset, 'batch_size', '?'))
        world_size = self._world_size()
        eff_bs = batch_size * world_size if isinstance(batch_size, int) else '?'
        mode = 'time-split' if cfg.get('split_by_time') else 'row-group'
        workers = cfg.get('num_workers', '?')
        compile_mode = cfg.get('compile_mode', 'none')
        amp = 'off'
        if self.use_amp:
            amp = cfg.get('amp_dtype') or str(self.amp_dtype).replace('torch.', '')
        return (
            f"[epoch] ep={epoch}/{self.num_epochs} mode={mode} "
            f"bs/rank={batch_size} world={world_size} eff_bs={eff_bs} "
            f"workers={workers} amp={amp} compile={compile_mode}"
        )

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name."""
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _build_epoch_dir_name(self, epoch: int, global_step: int) -> str:
        """Build a non-best epoch checkpoint sub-directory name."""
        parts = [f"global_step{global_step}", f"epoch{epoch}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        return ".".join(parts)

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``."""
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        # Copy dense_stats.json so infer.py can apply the identical long-tail
        # transform (read from MODEL_OUTPUT_PATH).
        if self.dense_stats_path and os.path.exists(self.dense_stats_path):
            shutil.copy2(self.dense_stats_path, ckpt_dir)

        if self.train_config:
            import json
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(self.train_config, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save model.pt plus sidecar files (only on rank 0)."""
        if not is_main_process():
            return ""
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(
                self.raw_model.state_dict(),
                os.path.join(ckpt_dir, "model.pt"),
            )
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _save_epoch_checkpoint(
        self, epoch: int, global_step: int, suffix: str = ''
    ) -> str:
        """Save a regular epoch checkpoint (only on rank 0).

        ``suffix`` (e.g. ``'swa'``) appends a ``.swa`` tag to the directory so
        the SWA checkpoint sits alongside the same-epoch regular one without
        clobbering it. When called for SWA the averaged weights are already
        swapped into ``raw_model``, so ``state_dict()`` captures the SWA model.
        """
        if not is_main_process():
            return ""
        dir_name = self._build_epoch_dir_name(epoch, global_step)
        if suffix:
            dir_name = f"{dir_name}.{suffix}"
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.raw_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        _tag = f" [{suffix}]" if suffix else ""
        logging.info(f"Saved epoch {epoch}{_tag} checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories."""
        if not is_main_process():
            return
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in batch to self.device."""
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically (only on rank 0)."""
        if not is_main_process():
            return

        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            self.early_stopping(val_auc, self.raw_model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.raw_model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def _broadcast_early_stop(self) -> bool:
        """Broadcast rank 0's early_stop flag to all ranks. Returns whether to stop."""
        if not is_ddp():
            return self.early_stopping.early_stop

        flag = torch.tensor(
            [1 if (is_main_process() and self.early_stopping.early_stop) else 0],
            dtype=torch.long, device=self.device
        )
        dist.broadcast(flag, src=0)
        return flag.item() == 1

    def train(self) -> None:
        """Main training loop with DDP support."""
        if is_main_process():
            print("Start training (PCVRRankMixer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            if is_main_process():
                print(self._epoch_start_line(epoch))
            # Accumulate loss on-device; sync to host only at the log cadence so
            # the per-step .item() CUDA sync no longer stalls the input pipeline.
            loss_sum = torch.zeros((), device=self.device)
            window_loss_sum = torch.zeros((), device=self.device)
            epoch_steps = 0  # actual batches this epoch (for avg loss denominator)
            window_steps = 0
            epoch_rows = 0
            epoch_t0 = time.time()

            # Join: ensures ranks that finish their data early participate in
            # empty all-reduce calls, preventing DDP deadlock with IterableDataset.
            join_ctx = Join([self.model]) if is_ddp() else _nullcontext()
            with join_ctx:
                for step, batch in enumerate(self.train_loader):
                    # Stash the first swa_bn_batches CPU batches so the per-epoch
                    # SWA BatchNorm recompute can replay them without opening a
                    # second (pin_memory) DataLoader iterator during the SWA eval
                    # (see self._swa_bn_cache note). Captured pristine, before
                    # _train_step (which moves a COPY to device and never mutates
                    # this dict). Cheap: a length check + append for ~50 steps.
                    if (self.swa is not None
                            and len(self._swa_bn_cache) < self.swa_bn_batches):
                        self._swa_bn_cache.append(batch)
                    loss = self._train_step(batch, epoch)  # detached 0-dim tensor on device
                    total_step += 1
                    epoch_steps += 1
                    window_steps += 1
                    epoch_rows += int(batch['label'].shape[0])
                    loss_sum += loss
                    window_loss_sum += loss

                    # SWA: fold an intra-epoch snapshot into the equal-weight
                    # running mean every swa_collect_every steps (once past the
                    # warm-up epoch). Side-effect-free on the live params /
                    # optimizer, so the training trajectory is unchanged.
                    if (self.swa is not None
                            and epoch >= self.swa_start_epoch
                            and self.swa_collect_every > 0
                            and total_step % self.swa_collect_every == 0):
                        self.swa.update()

                    if is_main_process() and (step + 1) % self.log_every_n_steps == 0:
                        # One host sync per log_every_n_steps (was every step);
                        # TB Loss/train is written at this same cadence.
                        loss_val = loss.item()
                        avg_window_loss = (window_loss_sum / max(window_steps, 1)).item()
                        avg_epoch_loss = (loss_sum / max(epoch_steps, 1)).item()
                        elapsed = time.time() - epoch_t0
                        step_speed = (step + 1) / elapsed if elapsed > 0 else 0.0
                        row_speed = (epoch_rows * self._world_size() / elapsed
                                     if elapsed > 0 else 0.0)
                        if self.writer:
                            self.writer.add_scalar('Loss/train', loss_val, total_step)
                        train_line = _format_train_progress_line(
                            epoch=epoch,
                            step=step + 1,
                            total_step=total_step,
                            loss=loss_val,
                            avg_window_loss=avg_window_loss,
                            avg_epoch_loss=avg_epoch_loss,
                            window_steps=window_steps,
                            step_per_sec=step_speed,
                            row_per_sec=row_speed,
                            elapsed_sec=elapsed,
                            lr=self._current_lr(),
                            mem_gb=self._gpu_memory_gb(),
                        )
                        remoe_stats = self._model_aux_stats()
                        if remoe_stats:
                            remoe_active = remoe_stats.get('avg_active', 0.0)
                            remoe_sparsity = remoe_stats.get('sparsity', 0.0)
                            remoe_l1 = remoe_stats.get('l1_coeff', 0.0)
                            remoe_target_active = remoe_stats.get(
                                'target_active', 0.0)
                            remoe_target_sparsity = remoe_stats.get(
                                'target_sparsity', 0.0)
                            remoe_aux = 0.0
                            if self._last_model_aux_loss is not None:
                                remoe_aux = float(
                                    self._last_model_aux_loss.detach().float().cpu())
                            train_line += (
                                f" | remoe_active={remoe_active:.2f} "
                                f"remoe_target_active={remoe_target_active:.2f} "
                                f"remoe_sparsity={remoe_sparsity:.3f} "
                                f"remoe_target_sparsity={remoe_target_sparsity:.3f} "
                                f"remoe_aux={remoe_aux:.3e} "
                                f"remoe_l1={remoe_l1:.1e}"
                            )
                            # Per-layer, per-expert ReLU-gate load (fraction of
                            # positions each expert slot is active) -> the MoE
                            # load-balance view. Compact in the text log; full
                            # per-expert series go to TensorBoard below.
                            per_layer = remoe_stats.get('per_layer', [])
                            for _li, _ls in enumerate(per_layer):
                                _loads = _ls.get('per_expert_load', [])
                                if _loads:
                                    _ls_str = ",".join(f"{x:.2f}" for x in _loads)
                                    train_line += f" | L{_li}_load=[{_ls_str}]"
                            if self.writer:
                                self.writer.add_scalar(
                                    'Loss/remoe_aux', remoe_aux, total_step)
                                self.writer.add_scalar(
                                    'ReMoE/avg_active', remoe_active, total_step)
                                self.writer.add_scalar(
                                    'ReMoE/target_active',
                                    remoe_target_active, total_step)
                                self.writer.add_scalar(
                                    'ReMoE/sparsity', remoe_sparsity, total_step)
                                self.writer.add_scalar(
                                    'ReMoE/target_sparsity',
                                    remoe_target_sparsity, total_step)
                                self.writer.add_scalar(
                                    'ReMoE/l1_coeff', remoe_l1, total_step)
                                # Per-layer / per-expert series: avg active
                                # experts, each expert slot's load fraction and
                                # mean ReLU gate weight. Lets TensorBoard show
                                # whether the experts stay balanced or collapse.
                                for _li, _ls in enumerate(per_layer):
                                    _la = _ls.get('avg_active')
                                    if _la is not None:
                                        self.writer.add_scalar(
                                            f'ReMoE/layer{_li}/avg_active',
                                            _la, total_step)
                                    for _ei, _lv in enumerate(
                                            _ls.get('per_expert_load', [])):
                                        self.writer.add_scalar(
                                            f'ReMoE/layer{_li}/expert{_ei}_load',
                                            _lv, total_step)
                                    for _ei, _wv in enumerate(
                                            _ls.get('per_expert_weight', [])):
                                        self.writer.add_scalar(
                                            f'ReMoE/layer{_li}/expert{_ei}_weight',
                                            _wv, total_step)
                        # logging.info already streams to stdout (StreamHandler)
                        # AND the train.log file; a separate print() duplicated
                        # every train line on captured stdout.
                        logging.info(train_line)
                        window_loss_sum.zero_()
                        window_steps = 0

                    # Step-level validation (only when eval_every_n_steps > 0).
                    if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                        logging.info(f"Evaluating at step {total_step}")
                        val_auc, val_logloss = self.evaluate(epoch=epoch)
                        self.model.train()

                        if is_main_process():
                            logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")
                            if self.writer:
                                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                        self._handle_validation_result(total_step, val_auc, val_logloss)

                        should_stop = self._broadcast_early_stop()
                        if should_stop:
                            logging.info(f"Early stopping at step {total_step}")
                            return

            train_elapsed = time.time() - epoch_t0
            if is_main_process():
                avg_epoch_loss = (loss_sum / max(epoch_steps, 1)).item()
                logging.info(f"Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}, "
                             f"train time: {train_elapsed/60:.1f}min")

            eval_t0 = time.time()
            val_auc, val_logloss = self.evaluate(epoch=epoch)
            eval_elapsed = time.time() - eval_t0
            self.model.train()

            if is_main_process():
                print(f"[valid] ep={epoch} gstep={total_step} | "
                      f"auc={val_auc:.6f} logloss={val_logloss:.4f} | "
                      f"train={_format_duration(train_elapsed)} "
                      f"eval={_format_duration(eval_elapsed)}")
                logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")
                if self.writer:
                    self.writer.add_scalar('AUC/valid', val_auc, total_step)
                    self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)
            if self.save_every_epoch:
                self._save_epoch_checkpoint(epoch, total_step)

            # SWA: fold the end-of-epoch snapshot, then (averaged-weights) eval +
            # save a parallel `.swa` checkpoint so the user can A/B the SWA model
            # against the same-epoch regular model on the identical holdout. Runs
            # BEFORE the early-stop break so the final epoch's SWA model is saved
            # too. Does not feed the early-stopper (that tracks the live model).
            if self.swa is not None and epoch >= self.swa_start_epoch:
                self.swa.update()
                self._run_swa_eval_and_save(epoch, total_step)

            should_stop = self._broadcast_early_stop()
            if should_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # Reinitialize high-cardinality sparse params (cold restart).
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.raw_model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                # DDP: reinit uses random init, broadcast from rank 0 to sync all ranks.
                if is_ddp():
                    for p in self.raw_model.parameters():
                        dist.broadcast(p.data, src=0)
                sparse_params = self.raw_model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ModelInput NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        seq_timestamps: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            if f'{domain}_timestamps' in device_batch:
                seq_timestamps[domain] = device_batch[f'{domain}_timestamps']
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            user_dense_feats_raw=device_batch.get('user_dense_feats_raw'),
            timestamp=device_batch.get('timestamp'),
            seq_timestamps=seq_timestamps or None,
        )

    def _bn_modules(self) -> List[nn.Module]:
        """All BatchNorm modules (input BN + per-MLP-layer BN) in the model."""
        from torch.nn.modules.batchnorm import _BatchNorm
        return [m for m in self.raw_model.modules() if isinstance(m, _BatchNorm)]

    @torch.no_grad()
    def _recompute_bn_for_swa(self) -> Tuple[List[Any], int]:
        """Reset + recompute BatchNorm running stats for the currently-loaded
        (SWA-averaged) weights over up to ``swa_bn_batches`` CACHED training
        batches (replayed from ``self._swa_bn_cache``, filled during the first
        epoch -- see that attribute's note for why we must NOT re-iterate the
        DataLoader here).

        The mean of N snapshots has its own activation statistics that none of
        the per-snapshot BN buffers describe, so without this the SWA model
        evaluates/infers with stale stats and looks falsely bad. Uses the
        inference forward (``predict`` => dropout OFF) so the recomputed stats
        match eval/inference conditions. Per-rank, no DDP collectives (matches
        how the live plain-BatchNorm stats are handled). Returns
        ``(backup, n_seen)`` for the caller to restore the live BN buffers.
        """
        bns = self._bn_modules()
        if not self._swa_bn_cache:
            # Can't happen in practice (the cache fills from epoch-1 step 1, well
            # before the first SWA eval), but if it ever did, skip the recompute
            # WITHOUT resetting -> keep the live BN stats rather than zeroing them.
            logging.warning(
                "SWA BN-recompute: batch cache empty; skipping recompute and "
                "keeping live BatchNorm stats.")
            return [], 0
        backup: List[Any] = []
        for m in bns:
            backup.append((
                m,
                None if m.running_mean is None else m.running_mean.clone(),
                None if m.running_var is None else m.running_var.clone(),
                None if m.num_batches_tracked is None
                else m.num_batches_tracked.clone(),
                m.momentum,
                m.training,
            ))
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average over the recompute pass
        # Whole model to eval (dropout / aux off), then re-enable ONLY the BN
        # modules so they accumulate running stats on the inference-path forward.
        was_training = self.raw_model.training
        self.raw_model.eval()
        for m in bns:
            m.train()
        n_seen = 0
        # Replay the CACHED CPU batches (no DataLoader iterator -> no second
        # pin_memory thread to race the valid loader's during the SWA eval).
        for batch in self._swa_bn_cache:
            device_batch = self._batch_to_device(batch)
            model_input = self._make_model_input(device_batch)
            with self._autocast_ctx():
                self.raw_model.predict(model_input)
            n_seen += 1
            if n_seen >= self.swa_bn_batches:
                break
        # Drain any aux-loss buffer the forwards populated (ReMoE etc.) so it
        # cannot leak into the next real training step.
        self._consume_model_aux_loss()
        if was_training:
            self.raw_model.train()
        return backup, n_seen

    @torch.no_grad()
    def _restore_bn(self, backup: List[Any]) -> None:
        """Restore the live BN running stats saved by ``_recompute_bn_for_swa``."""
        for (m, rmean, rvar, nbt, momentum, training) in backup:
            if rmean is not None:
                m.running_mean.copy_(rmean)
            if rvar is not None:
                m.running_var.copy_(rvar)
            if nbt is not None:
                m.num_batches_tracked.copy_(nbt)
            m.momentum = momentum
            m.train(training)

    def _run_swa_eval_and_save(self, epoch: int, total_step: int) -> None:
        """Eval + save the SWA-averaged model for this epoch, then restore the
        live weights/BN exactly. The global RNG is snapshot/restored around the
        whole block so the training data + dropout stream is byte-identical to a
        non-SWA run (the SWA eval/BN-recompute forwards consume no RNG budget the
        live trajectory can see)."""
        if self.swa is None or self.swa.n <= 0:
            return
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        try:
            self.swa.apply_shadow()
            bn_backup, n_seen = self._recompute_bn_for_swa()
            swa_auc, swa_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            if is_main_process():
                print(f"[swa] ep={epoch} gstep={total_step} n_avg={self.swa.n} "
                      f"bn_batches={n_seen} | auc={swa_auc:.6f} "
                      f"logloss={swa_logloss:.4f}")
                logging.info(
                    f"Epoch {epoch} SWA Validation | n_avg={self.swa.n} "
                    f"AUC: {swa_auc}, LogLoss: {swa_logloss}")
                if self.writer:
                    self.writer.add_scalar('AUC/valid_swa', swa_auc, total_step)
                    self.writer.add_scalar(
                        'LogLoss/valid_swa', swa_logloss, total_step)
            if self.save_every_epoch:
                self._save_epoch_checkpoint(epoch, total_step, suffix='swa')
            self._restore_bn(bn_backup)
            self.swa.restore()
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        self.model.train()

    def _autocast_ctx(self) -> Any:
        """Return the configured AMP autocast context or a no-op context."""
        if not self.use_amp:
            return _nullcontext()
        return torch.autocast(
            device_type=self._autocast_device_type,
            dtype=self.amp_dtype,
        )

    def _aux_model(self) -> nn.Module:
        return getattr(self.raw_model, '_orig_mod', self.raw_model)

    def _consume_model_aux_loss(self) -> Optional[torch.Tensor]:
        model = self._aux_model()
        if not hasattr(model, 'consume_aux_loss'):
            return None
        return model.consume_aux_loss()

    def _update_model_aux_state(self) -> None:
        model = self._aux_model()
        if hasattr(model, 'update_remoe_l1_coeff'):
            model.update_remoe_l1_coeff()

    def _model_aux_stats(self) -> Dict[str, float]:
        model = self._aux_model()
        if not hasattr(model, 'remoe_router_stats'):
            return {}
        return model.remoe_router_stats()

    def _compute_loss(
        self,
        model_input: ModelInput,
        label: torch.Tensor,
        sample_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass + loss assembly for both vanilla and SAM steps."""
        with self._autocast_ctx():
            logits = self.model(model_input).squeeze(-1)
            if self.loss_type == 'focal':
                loss = sigmoid_focal_loss(
                    logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
            elif self.loss_type in ('bce', 'bce_pos_weight'):
                pw = None
                if self.loss_type == 'bce_pos_weight' and self.pos_weight != 1.0:
                    pw = torch.full_like(label, self.pos_weight, dtype=logits.dtype)
                bce_per = F.binary_cross_entropy_with_logits(
                    logits, label, reduction='none', pos_weight=pw)
                loss = _weighted_mean(bce_per, sample_weight)
            else:
                raise ValueError(f"Unknown loss_type={self.loss_type!r}")
            aux_loss = self._consume_model_aux_loss()
            self._last_model_aux_loss = None
            if aux_loss is not None:
                self._last_model_aux_loss = aux_loss.detach()
                loss = loss + aux_loss
        return loss

    @torch.no_grad()
    def _sam_first_step(self) -> None:
        """Perturb dense params along the current dense-gradient direction.

        With ``sam_matrix_only`` the ascent step is confined to the weight
        MATRICES (``self._sam_matrix_ids``): both the gradient norm and the
        perturbation exclude norm/bias/scalar params, so the rho-radius ball
        lives entirely in the interaction-matrix subspace. With it off
        (default) every dense param is perturbed exactly as before.
        """
        grad_norm_sq = torch.zeros(1, device=self.device)
        for group in self.dense_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.sam_matrix_only and id(p) not in self._sam_matrix_ids:
                    continue
                grad_norm_sq = grad_norm_sq + p.grad.detach().pow(2).sum()
        grad_norm = grad_norm_sq.sqrt()
        scale = self.sam_rho / (grad_norm + 1e-12)

        self._sam_perturbations.clear()
        for group in self.dense_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.sam_matrix_only and id(p) not in self._sam_matrix_ids:
                    continue
                e_w = (p.grad.detach() * scale).to(p.dtype)
                self._sam_perturbations[id(p)] = e_w
                p.add_(e_w)

    @torch.no_grad()
    def _sam_second_step(self) -> None:
        """Restore dense params after the second SAM backward pass."""
        for group in self.dense_optimizer.param_groups:
            for p in group['params']:
                e_w = self._sam_perturbations.get(id(p))
                if e_w is not None:
                    p.sub_(e_w)
        self._sam_perturbations.clear()

    def _clip_grad(self, foreach: bool) -> None:
        """Clip the global grad-norm to 1.0 over params with DENSE grads only.

        Sparse grads (e.g. MSN per-token value Embeddings) are skipped: torch's
        vector/foreach norm has no SparseCPU/CUDA kernel, and those tables are
        handled by Adagrad (which doesn't need grad clipping). When NO param has
        a sparse grad (the default / MSN off), this clips exactly the same set as
        ``clip_grad_norm_(self.model.parameters())`` — clip_grad_norm_ already
        skips ``grad is None`` — so the base trajectory is byte-identical.
        """
        params = [p for p in self.model.parameters()
                  if p.grad is not None and not p.grad.is_sparse]
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, foreach=foreach)

    def _train_step(self, batch: Dict[str, Any], epoch: int = 1) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()
        sample_weight = device_batch.get('sample_weight')
        if sample_weight is not None:
            sample_weight = sample_weight.float()

        model_input = self._make_model_input(device_batch)

        # Delayed-SAM gate: epochs before sam_start_epoch train with the plain
        # single-pass step (let the tokenizer/dense per-fid align first), then
        # the flat-minimum two-pass ascent kicks in. start_epoch=1 -> always on
        # from epoch 1 = the original behavior (byte-identical).
        use_sam_now = self.use_sam and epoch >= self.sam_start_epoch
        if use_sam_now:
            self.dense_optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()
            loss = self._compute_loss(model_input, label, sample_weight)
            loss.backward()
            reported_loss = loss.detach()

            self._sam_first_step()

            self.dense_optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()
            loss2 = self._compute_loss(model_input, label, sample_weight)
            loss2.backward()

            self._sam_second_step()
            self._clip_grad(foreach=False)
            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()
            self._update_model_aux_state()
            return reported_loss

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()
        loss = self._compute_loss(model_input, label, sample_weight)
        loss.backward()
        # Clip dense grads only (skips MSN sparse value-table grads, which torch's
        # foreach norm can't process and which Adagrad doesn't need clipped). When
        # MSN is off, no grad is sparse -> identical clip set as before.
        self._clip_grad(foreach=True)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()
        self._update_model_aux_state()

        # Return the detached loss tensor (no .item()): the caller accumulates
        # on-device and only syncs to host at the logging cadence.
        return loss.detach()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation and return (AUC, logloss).

        In DDP mode, gathers results from all ranks and computes metrics on rank 0.
        Other ranks receive (0.0, 0.0).
        """
        if is_main_process():
            logging.info("Start Evaluation (PCVRRankMixer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        # Progress denominator (see progress_num_batches). The valid loader does
        # NOT re-pack (shuffle=False / buffer_batches=0): it filters each raw Arrow
        # batch in place and yields it, so the batch COUNT is the raw per-row-group
        # ceiling sum regardless of --split_by_time (only rows-per-batch shrink).
        # progress_num_batches returns exactly that for the no-repack path, so the
        # 25/50/75/100% prints land correctly. (Scaling it by keep_fraction would
        # under-count ~10x under time split and suppress the prints.)
        num_batches = self.valid_loader.dataset.progress_num_batches()
        # Print eval progress only ~4 times (at 25/50/75/100% of num_batches).
        eval_print_interval = max(1, num_batches // 4)
        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            eval_t0 = time.time()
            for step, batch in enumerate(self.valid_loader):
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())
                if (
                    is_main_process()
                    and (step + 1) % eval_print_interval == 0
                    and (step + 1) <= num_batches
                ):
                    elapsed = time.time() - eval_t0
                    print(f"[eval] ep={epoch} step={step+1} | "
                          f"elapsed={_format_duration(elapsed)}")

        if all_logits_list:
            all_logits = torch.cat(all_logits_list, dim=0).float()
            all_labels = torch.cat(all_labels_list, dim=0).long()
        else:
            # This rank's valid shard is empty. Under DDP the valid row groups
            # are split across ranks by greedy row-count bucketing (dataset.py),
            # so when valid has FEWER row groups than ranks the non-owning ranks
            # get zero batches. The prime case is full-data training, which
            # floors valid to a single RG (--valid_ratio 0) -> only one rank
            # owns it. Emit correctly-shaped empty tensors so the collective
            # _gather_eval_results below (which already pads variable-length
            # shards) still runs on EVERY rank and rank 0 computes metrics over
            # the non-empty shard, instead of crashing here on torch.cat([]).
            all_logits = torch.empty(0, dtype=torch.float32)
            all_labels = torch.empty(0, dtype=torch.long)

        # DDP: gather eval results from all ranks to rank 0.
        if is_ddp():
            all_logits, all_labels = self._gather_eval_results(all_logits, all_labels)

        # Only rank 0 computes metrics; other ranks return placeholder.
        if not is_main_process():
            return 0.0, 0.0

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _gather_eval_results(
        self, local_logits: torch.Tensor, local_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDP: gather eval results from all ranks to rank 0.

        Uses pad + all_gather to handle variable-length data from IterableDataset.
        """
        device = torch.device(self.device)
        world_size = dist.get_world_size()

        # 1. Gather each rank's sample count.
        local_size = torch.tensor([local_logits.shape[0]], dtype=torch.long, device=device)
        size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        sizes = [int(s.item()) for s in size_list]
        max_size = max(sizes)

        # 2. Pad to uniform length and all_gather.
        def _pad_and_gather(tensor, max_sz):
            padded = torch.zeros(max_sz, dtype=tensor.dtype, device=device)
            padded[:tensor.shape[0]] = tensor.to(device)
            gathered = [torch.zeros(max_sz, dtype=tensor.dtype, device=device) for _ in range(world_size)]
            dist.all_gather(gathered, padded)
            parts = [g[:sz].cpu() for g, sz in zip(gathered, sizes)]
            return torch.cat(parts, dim=0)

        all_logits = _pad_and_gather(local_logits, max_size)
        all_labels = _pad_and_gather(local_labels, max_size)

        return all_logits, all_labels

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return (logits, labels)."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        with self._autocast_ctx():
            logits, _ = self.raw_model.predict(model_input)
        logits = logits.squeeze(-1)

        return logits.float(), label
