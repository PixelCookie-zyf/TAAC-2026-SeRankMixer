import os
import random
import copy
import logging
import time
from datetime import timedelta
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_sdp_kernels(
    enable_flash: bool = True,
    enable_mem_efficient: bool = True,
    enable_math: bool = True,
) -> None:
    """Configure CUDA SDPA kernel preferences when the backend exposes them.

    ``F.scaled_dot_product_attention`` chooses among Flash, memory-efficient,
    and math kernels at runtime based on dtype, mask shape, device, and tensor
    layout. These switches only declare which backends are allowed.
    """
    if not torch.cuda.is_available() or not hasattr(torch.backends, "cuda"):
        logging.info("CUDA SDPA kernel config skipped: CUDA is not available")
        return

    if not (enable_flash or enable_mem_efficient or enable_math):
        raise ValueError("At least one SDPA backend must be enabled")

    cuda_backend = torch.backends.cuda
    changed = False
    for name, enabled in (
        ("enable_flash_sdp", enable_flash),
        ("enable_mem_efficient_sdp", enable_mem_efficient),
        ("enable_math_sdp", enable_math),
    ):
        fn = getattr(cuda_backend, name, None)
        if callable(fn):
            fn(enabled)
            changed = True

    if changed:
        logging.info(
            "CUDA SDPA kernels configured: flash=%s, mem_efficient=%s, math=%s",
            enable_flash,
            enable_mem_efficient,
            enable_math,
        )
    else:
        logging.info("CUDA SDPA kernel switches are unavailable in this PyTorch build")


class LogFormatter:
    """Custom ``logging.Formatter`` that prefixes every record with the
    wall-clock timestamp and the elapsed wall-clock time since this
    formatter instance was constructed.

    The prefix format is ``"<locale-date> <locale-time> - H:MM:SS"``, which
    is convenient for tracking long-running training runs where both the
    absolute time and the time-since-start are useful.

    Multi-line messages are re-indented so that continuation lines align
    with the beginning of the message (not the prefix).
    """

    def __init__(self) -> None:
        # Anchor used to compute the elapsed-time part of the log prefix.
        # Can be reset at runtime via ``create_logger(...).reset_time()``.
        self.start_time: float = time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s" % (
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        # Indent continuation lines so they line up with the message body,
        # not with the timestamp prefix.
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(filepath: str) -> logging.Logger:
    """Create and configure the root logger for a training/inference run.

    The returned logger has two handlers attached:

    * A ``FileHandler`` bound to ``filepath`` (opened in write mode,
      truncating any previous content) that records ``DEBUG``-level and
      above messages for post-mortem inspection.
    * A ``StreamHandler`` to stderr that only echoes ``INFO``-level and
      above messages, keeping the console output concise.

    Both handlers share a ``LogFormatter`` so the console and the log file
    stay in sync. Any pre-existing handlers on the root logger are removed
    to avoid duplicate lines when this function is called multiple times.

    Args:
        filepath: Destination path of the log file. Opened in ``"w"`` mode,
            so previous contents are overwritten.

    Returns:
        The root ``logging.Logger`` instance. The returned object is
        augmented with a ``reset_time()`` attribute that resets the
        elapsed-time clock used by the log prefix. This is useful when the
        "interesting" phase of a run starts well after process launch
        (e.g. after schema building and data loading).
    """
    log_formatter = LogFormatter()

    file_handler = logging.FileHandler(filepath, "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Allow callers to reset the elapsed-time clock shown in the log prefix.
    def reset_time() -> None:
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time  # type: ignore[attr-defined]

    return logger


class EarlyStopping:
    """Early-stop training when the validation metric plateaus.

    The tracker assumes a *higher-is-better* metric (typical for AUC or
    accuracy). A candidate ``score`` is considered an improvement iff
    ``score > best_score + delta``; otherwise the internal ``counter`` is
    incremented and training is requested to stop once
    ``counter >= patience``.

    On every improvement the current ``model.state_dict()`` is both
    deep-copied in memory (``self.best_model``) and persisted to disk at
    ``checkpoint_path``. The most recent *improving* score is cached in
    ``self.best_saved_score`` so callers can skip redundant IO.

    Attributes:
        checkpoint_path: Destination path for the best ``state_dict``.
        patience: Number of non-improving calls tolerated before
            ``early_stop`` is flipped to ``True``.
        verbose: If ``True``, emit an ``INFO`` line whenever a checkpoint
            is written.
        counter: Number of consecutive non-improving calls seen so far.
        best_score: Best score observed; ``None`` until the first call.
        early_stop: Set to ``True`` once ``counter >= patience``.
        delta: Minimum absolute improvement required to reset ``counter``.
        best_model: In-memory deep copy of the best ``state_dict``.
        best_saved_score: Score associated with the last checkpoint
            actually written to disk.
        best_extra_metrics: Optional auxiliary metrics captured at the
            best-score step (e.g. logloss, other AUCs).
        label: Short prefix (e.g. ``"val"``) prepended to log lines to
            disambiguate multiple trackers running in parallel.
    """

    def __init__(
        self,
        checkpoint_path: str,
        label: str = "",
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0,
    ) -> None:
        self.checkpoint_path: str = checkpoint_path
        self.patience: int = patience
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.delta: float = delta
        self.best_model: Optional[Dict[str, torch.Tensor]] = None
        self.best_saved_score: float = 0.0
        self.best_extra_metrics: Optional[Dict[str, Any]] = None
        self.label: str = label
        if self.label != "":
            self.label += " "

    def _is_not_improved(self, score: float) -> bool:
        """Return ``True`` iff ``score`` fails to beat ``best_score + delta``.

        Used as the gating condition for incrementing the patience counter.
        ``best_score`` must have been seeded by a prior ``__call__``.
        """
        assert self.best_score is not None, "call __call__ first to seed best_score"
        if score > self.best_score + self.delta:
            return False
        return True

    def __call__(
        self,
        score: float,
        model: nn.Module,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Feed a new validation score into the tracker.

        Three branches, in order:

        1. First call (``best_score is None``): seed the tracker, persist a
           checkpoint, and cache the model weights.
        2. Not improved: increment ``counter`` and log the progress; flip
           ``early_stop`` once ``counter >= patience``.
        3. Improved: reset ``counter`` to ``0``, update ``best_score`` and
           ``best_extra_metrics``, refresh the in-memory ``best_model``,
           and write a new checkpoint to disk.

        Args:
            score: Scalar validation metric (higher is better, e.g. AUC).
            model: Model whose ``state_dict`` is snapshotted on
                improvement. Only the parameters are saved, not the
                optimizer state.
            extra_metrics: Optional dict of auxiliary metrics recorded at
                the same step, e.g.
                ``{"best_val_AUC": ..., "best_val_logloss": ...}``. Stored
                verbatim as ``self.best_extra_metrics``; not interpreted
                by ``EarlyStopping`` itself.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_extra_metrics = extra_metrics
            self.best_saved_score = 0.0
            self.save_checkpoint(score, model)
            self.best_model = copy.deepcopy(model.state_dict())
        elif self._is_not_improved(score):
            self.counter += 1
            logging.info(f'{self.label}earlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            logging.info(f'{self.label}earlyStopping counter reset!')
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_extra_metrics = extra_metrics
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score: float, model: nn.Module) -> None:
        """Persist ``model.state_dict()`` to ``self.checkpoint_path``.

        Creates any missing parent directories, writes atomically via
        ``torch.save``, and records ``score`` as ``self.best_saved_score``
        so subsequent callers can detect "no new improvement since last
        save" without re-reading the checkpoint file.

        Args:
            score: Validation score associated with the weights being
                saved. Exposed to callers via ``best_saved_score`` after
                the write completes.
            model: Model whose parameters are being snapshotted. Only
                ``state_dict()`` is written; optimizer and scheduler state
                are explicitly *not* included.
        """
        if self.verbose:
            logging.info('Validation score increased. Saving model ...')
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        self.best_saved_score = score


def set_seed(seed: int) -> None:
    """Seed every RNG that can influence training reproducibility.

    Seeds ``random``, the ``PYTHONHASHSEED`` env var, NumPy, the CPU
    PyTorch generator and all CUDA generators, then forces cuDNN into
    deterministic mode.

    Note that full bitwise determinism on GPU also requires disabling
    cuDNN auto-tuning (``torch.backends.cudnn.benchmark = False``) and may
    come with a non-trivial throughput cost; this helper intentionally
    only toggles ``deterministic`` to preserve speed for common use cases.

    Args:
        seed: Non-negative integer seed shared by all RNGs listed above.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: (N,) raw logits (before sigmoid).
        targets: (N,) binary labels {0, 1}.
        alpha: positive-class weight in (0, 1). When positives dominate,
            use alpha < 0.5 to downweight the positive class.
        gamma: focusing parameter. gamma=0 degenerates to standard BCE;
            gamma=2 is the standard value.
        reduction: 'mean' | 'sum' | 'none'.
    """
    p = torch.sigmoid(logits)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


# ───────────────────────── Train-log formatting ──────────────────────────────


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_rate(value: float, unit: str) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M {unit}/s"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k {unit}/s"
    return f"{value:.2f} {unit}/s"


def _format_train_progress_line(
    *,
    epoch: int,
    step: int,
    total_step: int,
    loss: float,
    avg_window_loss: float,
    avg_epoch_loss: float,
    window_steps: int,
    step_per_sec: float,
    row_per_sec: float,
    elapsed_sec: float,
    lr: float,
    mem_gb: Optional[float],
) -> str:
    mem = "NA" if mem_gb is None else f"{mem_gb:.1f}G"
    return (
        f"[train] ep={epoch} step={step} gstep={total_step} | "
        f"loss={loss:.4f} avg{window_steps}={avg_window_loss:.4f} "
        f"avg_ep={avg_epoch_loss:.4f} | "
        f"{_format_rate(step_per_sec, 'step')} "
        f"{_format_rate(row_per_sec, 'row')} | "
        f"elapsed={_format_duration(elapsed_sec)} | "
        f"lr={lr:.1e} mem={mem}"
    )
