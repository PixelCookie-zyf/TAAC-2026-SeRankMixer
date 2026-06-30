"""RankMixer model for PCVR experiments (DIN + ChannelSENET + RankMixer cross).

Self-contained module that bundles:
    - ``ModelInput`` (matches the baseline contract).
    - ``DINCrossAttention`` (DIN-style query-to-sequence scoring).
    - ``SafeBatchNorm1d``, ``ChannelSENETLayer`` (flat-vector preprocessing).
    - ``RankMixer*`` (semantic tokenizer + token-mixing + per-token pSwiGLU FFN).
    - ``FeatureFieldTokenizer`` (one embedding field per sparse fid, with
      optional dense-weighted pooling for paired int/dense fids).
    - ``PCVRDINConcatSENETDCN`` (the model: direct-concat DIN -> input BN ->
      ChannelSENET -> RankMixer cross -> MLP head). ``PCVRRankMixer`` is an alias.
"""

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    # post-signed_log1p, pre-zscore copy of user_dense — used by the
    # (int, dense) pair-weighted NS-tokenizer path so weights stay
    # non-negative.
    user_dense_feats_raw: Optional[torch.Tensor] = None
    timestamp: Optional[torch.Tensor] = None
    seq_timestamps: Optional[dict] = None  # {domain: tensor [B, L]}


class DINCrossAttention(nn.Module):
    """DIN-style query-to-sequence attention.

    Keeps a query-token contract `(B, Nq, D) -> (B, Nq, D)` but
    replaces dot-product multi-head attention with a DIN scoring MLP over
    `[q, k, q-k, q*k]`.
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        ln_mode: str = 'pre',
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode
        self.d_model = d_model

        hidden_dim = d_model * hidden_mult
        self.score_mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        score_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode query tokens from sequence tokens using DIN scoring.

        Args:
            query: (B, Nq, D), output residual query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: Ignored; accepted for interface parity with CrossAttention.
            rope_sin: Ignored; accepted for interface parity with CrossAttention.
            score_query: Optional (B, Nq, D) tokens used only for DIN scoring.
                Defaults to ``query``. This allows a target-item DIN ablation
                while preserving the query tokens as the output residual.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        del rope_cos, rope_sin
        residual = query
        score_query = query if score_query is None else score_query

        if self.ln_mode == 'pre':
            score_query = self.norm_q(score_query)
            key_value = self.norm_kv(key_value)

        B, Nq, D = query.shape
        L = key_value.shape[1]

        q = score_query.unsqueeze(2).expand(B, Nq, L, D)
        k = key_value.unsqueeze(1).expand(B, Nq, L, D)
        din_features = torch.cat([q, k, q - k, q * k], dim=-1)
        scores = self.score_mlp(din_features).squeeze(-1)  # (B, Nq, L)

        all_padding = None
        if key_padding_mask is not None:
            all_padding = key_padding_mask.all(dim=1, keepdim=True)  # (B, 1)
            mask = key_padding_mask.unsqueeze(1).expand(B, Nq, L)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        if all_padding is not None:
            weights = torch.where(
                all_padding.unsqueeze(-1),
                torch.zeros_like(weights),
                weights,
            )
        weights = self.dropout(weights)

        values = self.value_proj(key_value)
        context = torch.einsum('bnl,bld->bnd', weights, values)
        out = residual + self.out_proj(context)

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerTokenizer(nn.Module):
    """Contiguous tokenization (RankMixer paper Eq. 2, no domain grouping).

    Splits the flat feature vector ``e_input`` into ``num_tokens`` contiguous
    slices and projects each slice to dimension ``token_dim`` with its OWN
    Linear:  ``x_i = Proj_i(e_input[off_i : off_{i+1}])``.  Slices are sized as
    evenly as possible (the first ``input_dim % num_tokens`` slices get one extra
    element), so any ``input_dim`` works (incl. the platform's odd 6565).
    """

    def __init__(self, input_dim: int, num_tokens: int, token_dim: int) -> None:
        super().__init__()
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        if num_tokens > input_dim:
            raise ValueError(
                f"num_tokens ({num_tokens}) must be <= input_dim ({input_dim})")
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        base, rem = divmod(input_dim, num_tokens)
        sizes = [base + (1 if i < rem else 0) for i in range(num_tokens)]
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + s)
        self.sizes = sizes
        self.offsets = offsets
        self.projs = nn.ModuleList([nn.Linear(s, token_dim) for s in sizes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = [
            proj(x[:, self.offsets[i]:self.offsets[i + 1]])
            for i, proj in enumerate(self.projs)
        ]
        return torch.stack(tokens, dim=1)  # (B, T, D)


class RankMixerSemanticTokenizer(nn.Module):
    """Semantic tokenization (RankMixer paper Sec. 3.1, 'domain knowledge').

    Splits the flat post-SENET vector at its natural feature-GROUP boundaries
    (here: user_emb / item_emb / seq / user_dense — the exact concat layout of
    PCVRDINConcatSENETDCN), and projects each group with its OWN Linear to
    ``n_g`` tokens of dim ``token_dim``. Groups are concatenated -> (B, T, D),
    T = sum(n_g). A zero-width group is skipped (gets no token / no Linear), but
    the running offset still advances past it so absolute slice positions stay
    correct.

    ``group_dims`` is an ordered list of (name, width) matching the flat vector
    layout; ``group_tokens`` the per-group token counts (same order, same length).
    """

    def __init__(
        self,
        group_dims: List[Tuple[str, int]],
        group_tokens: List[int],
        token_dim: int,
        per_token_groups: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if len(group_dims) != len(group_tokens):
            raise ValueError("group_dims and group_tokens must align")
        self.token_dim = token_dim
        per_token = set(per_token_groups or [])
        self.group_names: List[str] = []
        self._specs: List[Tuple[int, int, int]] = []  # (offset, width, n_tok)
        self._per_token: List[bool] = []
        self.projs = nn.ModuleList()
        offset = 0
        for (name, width), n_tok in zip(group_dims, group_tokens):
            if width > 0 and n_tok > 0:
                if name in per_token:
                    # Each token owns its OWN Linear over an equal slice of the
                    # group (e.g. one Linear(d_model -> D) per behavior
                    # domain), instead of one shared Linear(width -> n_tok*D).
                    if width % n_tok != 0:
                        raise ValueError(
                            f"per-token group {name!r}: width {width} not "
                            f"divisible by n_tok {n_tok}")
                    slice_w = width // n_tok
                    self.projs.append(nn.ModuleList(
                        [nn.Linear(slice_w, token_dim) for _ in range(n_tok)]))
                    self._per_token.append(True)
                else:
                    self.projs.append(nn.Linear(width, n_tok * token_dim))
                    self._per_token.append(False)
                self._specs.append((offset, width, n_tok))
                self.group_names.append(name)
            offset += width
        if not self._specs:
            raise ValueError("RankMixerSemanticTokenizer: no non-empty groups")
        self.num_tokens = sum(n for _, _, n in self._specs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        out = []
        for (offset, width, n_tok), proj, per_tok in zip(
                self._specs, self.projs, self._per_token):
            seg = x[:, offset:offset + width]
            if per_tok:
                slice_w = width // n_tok
                toks = [
                    proj[i](seg[:, i * slice_w:(i + 1) * slice_w]).view(
                        B, 1, self.token_dim)
                    for i in range(n_tok)
                ]
                out.append(torch.cat(toks, dim=1))
            else:
                out.append(proj(seg).view(B, n_tok, self.token_dim))
        return torch.cat(out, dim=1)  # (B, T, D)


def _expand_dense_groups(
    layout: List[Tuple[int, int]],
    spec: Optional[str],
    prefix: str,
    expected_total: int,
) -> List[Tuple[str, int]]:
    """Expand a dense block into per-fid (or per-fid-cluster) tokenizer groups.

    ``layout`` is the ordered ``[(fid, dim), ...]`` of the dense fields as they
    appear in the concat (user_dense = the kept tail with the pair fids already
    removed; item_dense = its full schema). ``spec`` groups fids into
    sub-groups: ``'|'`` separates sub-groups and ``','`` joins fids merged into
    ONE token, e.g. ``'61|87|89,90,91,118|120|...'``. Empty / None => one
    sub-group per fid. Each sub-group becomes one ``('<prefix>_<firstfid>',
    summed_width)`` entry that the semantic tokenizer projects to a single
    token. Validates the grouping covers the kept fids contiguously and that the
    expanded width equals the dense block width (so absolute slice offsets stay
    correct).
    """
    fid_to_dim = {fid: dim for fid, dim in layout}
    order = [fid for fid, _ in layout]
    if spec and str(spec).strip():
        groups = [
            [int(t) for t in chunk.split(',') if t.strip()]
            for chunk in str(spec).split('|') if chunk.strip()
        ]
    else:
        groups = [[fid] for fid in order]
    flat = [f for g in groups for f in g]
    if flat != order:
        raise ValueError(
            f"dense_per_fid {prefix!r} groups must cover the kept fids {order} "
            f"contiguously in order, got {flat}")
    out = [(f"{prefix}_{g[0]}", sum(fid_to_dim[f] for f in g)) for g in groups]
    tot = sum(w for _, w in out)
    if tot != expected_total:
        raise ValueError(
            f"dense_per_fid {prefix!r} expanded width {tot} != expected "
            f"{expected_total}")
    return out


class RankMixerTokenMixing(nn.Module):
    """Parameter-free multi-head token mixing (RankMixer paper Eq. 3-4).

    Split each of the T tokens into H heads of dim D/H, then form output token h
    by concatenating the h-th head of every token: ``s^h = Concat_t(x_t^h)``.

    REQUIRES ``num_heads == num_tokens`` (the paper's H = T). That is not a
    convenience default but a hard constraint of this whole module: only H = T
    makes the mixing shape-preserving (B, T, D) -> (B, T, D), which is what the
    residual ``TokenMixing(X) + X`` and the per-token PFFN (both sized to T
    tokens of dim D) need. For a general H, paper Eq.4 yields H tokens of dim
    T*D/H -- a different shape the rest of the block can't consume -- so we
    reject it loudly instead of silently running a non-paper permutation that
    happens to reshape back to (B, T, D).
    """

    def __init__(self, num_tokens: int, token_dim: int, num_heads: int) -> None:
        super().__init__()
        if num_heads != num_tokens:
            raise ValueError(
                "RankMixer token mixing requires num_heads == num_tokens "
                f"(paper H = T); got num_heads={num_heads}, "
                f"num_tokens={num_tokens}. Use --rankmixer_heads 0 (auto H=T) "
                "or set it exactly equal to the total token count.")
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by num_heads "
                f"({num_heads})")
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        head_dim = self.token_dim // self.num_heads
        # (B, T, H, Dh) -> (B, H, T, Dh) -> regroup back to (B, T, D).
        x = x.view(B, self.num_tokens, self.num_heads, head_dim)
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_tokens, self.token_dim)
        return x


# Down-Matrix Small Initialization gain (TokenMixer-Large arXiv 2602.06563
# Sec. 3.4.4): xavier_uniform gain for W_down only, lowered from default 1.0.
_DOWN_MATRIX_INIT_GAIN = 0.01


class RankMixerPSwiGLUFFN(nn.Module):
    """Per-token SwiGLU FFN (TokenMixer-Large paper arXiv 2602.06563, Eq. 17-18).

    Each token t owns its OWN gated FFN
    ``v_t = W_down_t @ ( Swish(W_gate_t @ s_t) ⊙ (W_up_t @ s_t) )``  with
    ``W_up_t, W_gate_t in R^{D x nD}`` and ``W_down_t in R^{nD x D}`` (n = expansion):
    3 matrices/token (3*n*D^2) plus a multiplicative Swish gate. Vectorized over the
    token axis with a batched einsum (NOT a per-token python loop), so all T
    per-token SwiGLUs run in one shot. Swish == SiLU (x*sigmoid(x)).
    """

    def __init__(self, num_tokens: int, token_dim: int, expansion: int,
                 down_init_gain: float = _DOWN_MATRIX_INIT_GAIN) -> None:
        super().__init__()
        hidden = token_dim * expansion
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.hidden = hidden
        self.w_up = nn.Parameter(torch.empty(num_tokens, token_dim, hidden))
        self.b_up = nn.Parameter(torch.zeros(num_tokens, hidden))
        self.w_gate = nn.Parameter(torch.empty(num_tokens, token_dim, hidden))
        self.b_gate = nn.Parameter(torch.zeros(num_tokens, hidden))
        self.w_down = nn.Parameter(torch.empty(num_tokens, hidden, token_dim))
        self.b_down = nn.Parameter(torch.zeros(num_tokens, token_dim))
        with torch.no_grad():
            for t in range(num_tokens):
                nn.init.xavier_uniform_(self.w_up[t])
                nn.init.xavier_uniform_(self.w_gate[t])
                # Down-Matrix Small Initialization (TokenMixer-Large Sec. 3.4.4):
                # init W_down with a small xavier gain (paper 0.01 vs default 1.0) so
                # the SwiGLU output -- F(x) in the residual F(x)+x -- starts ~0, an
                # approximate identity early on (W_up/W_gate keep the default gain).
                # Trade-off: a small gain also SHRINKS the W_up/W_gate gradients at
                # init (they flow through W_down). gain is --rankmixer_down_init_gain.
                nn.init.xavier_uniform_(self.w_down[t], gain=down_init_gain)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        up = torch.einsum('btd,tde->bte', s, self.w_up) + self.b_up
        gate = torch.einsum('btd,tde->bte', s, self.w_gate) + self.b_gate
        h = F.silu(gate) * up
        v = torch.einsum('bte,ted->btd', h, self.w_down) + self.b_down
        return v


class RankMixerReMoEFFN(nn.Module):
    """Per-token ReMoE FFN with a single ReLU router (ICLR'25 arXiv 2412.14711).

    Each RankMixer token owns its own expert bank:
    ``w_up/w_gate: [T, E, D, H]`` and ``w_down: [T, E, H, D]``. A SINGLE per-token
    router produces ``route_weights = relu(W_r @ s + b_r)`` over the experts; the
    output is ``sum_e route_weights_e * Expert_e(s)`` -- unnormalised ReLU gating,
    the same function at train and inference (ReMoE's "fully differentiable"
    property). Sparsity is driven by an L1 penalty on ``route_weights`` with an
    adaptively-tuned coefficient targeting ``target_active`` experts/token.
    ``router_b`` is initialised positive so ~all experts start active (warm DENSE
    start, no dead experts) and the L1 then sparsifies.

    NB on the history: an earlier revision used TWO independent routers (a softmax
    "train" router shaping the experts via a dense gradient, and a separate ReLU
    "infer" router producing the value). That decoupled the combination the
    experts were optimised for from the one actually used, so the per-token FFN
    barely engaged -- training collapsed in ~1 epoch to a sub-pSwiGLU-baseline
    plateau. This single-router form is the fix.
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        expansion: int,
        num_experts: int = 4,
        target_active: int = 2,
        l1_coeff_init: float = 1e-8,
        l1_coeff_multiplier: float = 1.2,
        down_init_gain: float = _DOWN_MATRIX_INIT_GAIN,
        router_bias_init: float = 1.0,
    ) -> None:
        super().__init__()
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if target_active < 1 or target_active > num_experts:
            raise ValueError(
                f"target_active must be in [1, num_experts], got "
                f"target_active={target_active}, num_experts={num_experts}")
        if l1_coeff_init < 0:
            raise ValueError(f"l1_coeff_init must be >= 0, got {l1_coeff_init}")
        if l1_coeff_multiplier <= 1.0:
            raise ValueError(
                f"l1_coeff_multiplier must be > 1, got {l1_coeff_multiplier}")

        hidden = token_dim * expansion
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.hidden = hidden
        self.num_experts = num_experts
        self.target_active = target_active
        self.target_sparsity = 1.0 - float(target_active) / float(num_experts)
        self.l1_coeff_multiplier = float(l1_coeff_multiplier)
        # ONE ReLU router, identical at train and inference (ReMoE). router_b
        # starts positive so relu(logits) > 0 for ~all experts at init -> warm
        # DENSE start with no dead experts; the adaptive L1 then sparsifies
        # toward target_active.
        self.router_w = nn.Parameter(
            torch.empty(num_tokens, token_dim, num_experts))
        self.router_b = nn.Parameter(
            torch.full((num_tokens, num_experts), float(router_bias_init)))
        self.w_up = nn.Parameter(
            torch.empty(num_tokens, num_experts, token_dim, hidden))
        self.b_up = nn.Parameter(torch.zeros(num_tokens, num_experts, hidden))
        self.w_gate = nn.Parameter(
            torch.empty(num_tokens, num_experts, token_dim, hidden))
        self.b_gate = nn.Parameter(torch.zeros(num_tokens, num_experts, hidden))
        self.w_down = nn.Parameter(
            torch.empty(num_tokens, num_experts, hidden, token_dim))
        self.b_down = nn.Parameter(torch.zeros(num_tokens, num_experts, token_dim))
        self.register_buffer(
            'l1_coeff', torch.tensor(float(l1_coeff_init), dtype=torch.float32))
        self._last_aux_loss: Optional[torch.Tensor] = None
        self._last_sparsity: Optional[torch.Tensor] = None
        self._last_router_stats: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for t in range(num_tokens):
                nn.init.xavier_uniform_(self.router_w[t])
                for e in range(num_experts):
                    nn.init.xavier_uniform_(self.w_up[t, e])
                    nn.init.xavier_uniform_(self.w_gate[t, e])
                    nn.init.xavier_uniform_(self.w_down[t, e],
                                            gain=down_init_gain)

    def _routing_l1_loss(
        self,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        return route_weights.mean()

    def _expert_outputs(self, s: torch.Tensor) -> torch.Tensor:
        up = torch.einsum('btd,tedh->bteh', s, self.w_up) + self.b_up
        gate = torch.einsum('btd,tedh->bteh', s, self.w_gate) + self.b_gate
        hidden = F.silu(gate) * up
        return torch.einsum('bteh,tehd->bted', hidden, self.w_down) + self.b_down

    def _sparse_outputs(
        self,
        s: torch.Tensor,
        route_weights: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = s.shape
        # Accumulate in fp32: under AMP (esp. rmsnorm, which autocast does NOT
        # force to fp32) s can be bf16 while expert_out is promoted to fp32 by the
        # fp32 bias add -> index_add_ requires matching dtypes. fp32 also matches
        # the dense train path (_expert_outputs is fp32 after the bias add), so
        # eval stays value-consistent with train. No-op under --no_amp (all fp32).
        flat_out = s.new_zeros(B, T, D, dtype=torch.float32)
        for token_id in range(self.num_tokens):
            token_in = s[:, token_id, :]
            for expert_id in range(self.num_experts):
                row_idx = routing_map[:, token_id, expert_id].nonzero(
                    as_tuple=False).flatten()
                if row_idx.numel() == 0:
                    continue
                expert_in = token_in.index_select(0, row_idx)
                up = (
                    expert_in @ self.w_up[token_id, expert_id]
                    + self.b_up[token_id, expert_id]
                )
                gate = (
                    expert_in @ self.w_gate[token_id, expert_id]
                    + self.b_gate[token_id, expert_id]
                )
                hidden = F.silu(gate) * up
                expert_out = (
                    hidden @ self.w_down[token_id, expert_id]
                    + self.b_down[token_id, expert_id]
                )
                expert_out = expert_out * route_weights[
                    row_idx, token_id, expert_id].unsqueeze(-1)
                flat_out[:, token_id, :].index_add_(
                    0, row_idx, expert_out.to(flat_out.dtype))
        return flat_out

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        B, T, D = s.shape
        if T != self.num_tokens:
            raise ValueError(
                f"RankMixerReMoEFFN expected {self.num_tokens} tokens, got {T}")
        # Single ReLU router, identical at train and inference (ReMoE).
        logits = torch.einsum('btd,tde->bte', s, self.router_w) + self.router_b
        route_weights = torch.relu(logits)
        routing_map = route_weights > 0

        if self.training:
            # Train computes every expert densely and weights by the ReLU gate:
            # gives a gradient to every (token, expert) whose gate is open AND
            # keeps the forward VALUE identical to the sparse eval path below.
            expert_out = self._expert_outputs(s)
            out = (expert_out * route_weights.unsqueeze(-1)).sum(dim=2)
        else:
            out = self._sparse_outputs(s, route_weights, routing_map)

        with torch.no_grad():
            active_mask = routing_map.to(route_weights.dtype)
            active = active_mask.sum()
            denom = float(route_weights.numel())
            sparsity = 1.0 - active / denom
            # Per-expert load = fraction of (sample, token) positions whose gate
            # for that expert slot is open; per-expert weight = mean ReLU gate.
            # Reduced over batch+tokens -> one number per expert slot per layer,
            # the standard MoE load-balance / expert-utilisation signal.
            per_expert_load = active_mask.mean(dim=(0, 1))
            per_expert_weight = route_weights.mean(dim=(0, 1))
        self._last_sparsity = sparsity.detach()
        self._last_router_stats = {
            'avg_active': (active / max(B * T, 1)).detach(),
            'sparsity': sparsity.detach(),
            'l1_coeff': self.l1_coeff.detach(),
            'target_active': route_weights.new_tensor(float(self.target_active)),
            'target_sparsity': route_weights.new_tensor(
                float(self.target_sparsity)),
            'per_expert_load': per_expert_load.detach(),
            'per_expert_weight': per_expert_weight.detach(),
        }
        if self.training and torch.is_grad_enabled():
            l1 = self._routing_l1_loss(route_weights)
            self._last_aux_loss = l1 * self.l1_coeff.to(
                device=route_weights.device, dtype=route_weights.dtype)
        else:
            self._last_aux_loss = None
        # No unused_expert_anchor needed: the dense train path computes all
        # experts + the single router, so every param is in the autograd graph
        # (zero grad, never None) -> safe under DDP find_unused_parameters=False.
        return out

    def consume_aux_loss(self) -> Optional[torch.Tensor]:
        aux_loss = self._last_aux_loss
        self._last_aux_loss = None
        return aux_loss

    def router_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for key, value in self._last_router_stats.items():
            v = value.detach().cpu()
            if v.dim() == 0:
                stats[key] = float(v)
            else:
                stats[key] = [float(x) for x in v.tolist()]
        return stats

    @torch.no_grad()
    def update_l1_coeff(self, current_sparsity: Optional[torch.Tensor] = None) -> None:
        sparsity = self._last_sparsity if current_sparsity is None else current_sparsity
        if sparsity is None:
            return
        if not isinstance(sparsity, torch.Tensor):
            sparsity = torch.tensor(
                float(sparsity), device=self.l1_coeff.device, dtype=torch.float32)
        else:
            sparsity = sparsity.to(device=self.l1_coeff.device, dtype=torch.float32)
        up = torch.tensor(
            self.l1_coeff_multiplier, device=self.l1_coeff.device,
            dtype=self.l1_coeff.dtype)
        down = torch.tensor(
            1.0 / self.l1_coeff_multiplier, device=self.l1_coeff.device,
            dtype=self.l1_coeff.dtype)
        factor = torch.where(
            sparsity < self.target_sparsity,
            up,
            down,
        )
        self.l1_coeff.mul_(factor)


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    """Norm factory for the RankMixer block.

    'ln' = ``nn.LayerNorm`` (paper default = the 0.822706 value); 'rmsnorm' =
    ``nn.RMSNorm`` (no mean-centering / no bias, just a learned RMS gain). RMSNorm
    needs torch>=2.4 -- local 2.6 and the competition platform 2.7.1 both have it,
    so we use the built-in instead of a hand-rolled module.
    """
    if norm_type == 'ln':
        return nn.LayerNorm(dim)
    if norm_type == 'rmsnorm':
        return nn.RMSNorm(dim)
    raise ValueError(f"norm_type must be 'ln' or 'rmsnorm', got {norm_type!r}")


class RankMixerBlock(nn.Module):
    """One RankMixer block (paper Eq. 1, post-norm).

    ``S = N1(TokenMixing(X) + X)`` then ``X' = N2(FFN(S) + S)``, where N is the
    norm (``norm_type='ln'`` = LayerNorm, paper default; ``'rmsnorm'`` = RMSNorm)
    and the FFN is either the per-token SwiGLU (``ffn_type='pswiglu'``) or the
    per-token ReMoE with a single ReLU router (``ffn_type='remoe'``).
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        num_heads: int,
        expansion: int,
        dropout: float = 0.0,
        ffn_type: str = 'pswiglu',
        down_init_gain: float = _DOWN_MATRIX_INIT_GAIN,
        norm_type: str = 'ln',
        moe_experts: int = 4,
        moe_top_k: int = 2,
        remoe_l1_coeff: float = 1e-8,
        remoe_l1_multiplier: float = 1.2,
    ) -> None:
        super().__init__()
        self.ffn_type = ffn_type
        self.token_mixing = RankMixerTokenMixing(num_tokens, token_dim, num_heads)
        self.pffn = _make_per_token_ffn(ffn_type, num_tokens, token_dim, expansion,
                                        down_init_gain=down_init_gain,
                                        moe_experts=moe_experts,
                                        moe_top_k=moe_top_k,
                                        remoe_l1_coeff=remoe_l1_coeff,
                                        remoe_l1_multiplier=remoe_l1_multiplier)
        self.norm1 = _make_norm(norm_type, token_dim)
        self.norm2 = _make_norm(norm_type, token_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.norm1(self.dropout(self.token_mixing(x)) + x)
        x = self.norm2(self.dropout(self.pffn(s)) + s)
        return x


def _make_per_token_ffn(ffn_type: str, num_tokens: int, token_dim: int,
                        expansion: int,
                        down_init_gain: float = _DOWN_MATRIX_INIT_GAIN,
                        moe_experts: int = 4,
                        moe_top_k: int = 2,
                        remoe_l1_coeff: float = 1e-8,
                        remoe_l1_multiplier: float = 1.2) -> nn.Module:
    if ffn_type == 'pswiglu':
        return RankMixerPSwiGLUFFN(num_tokens, token_dim, expansion,
                                   down_init_gain=down_init_gain)
    if ffn_type == 'remoe':
        return RankMixerReMoEFFN(
            num_tokens, token_dim, expansion,
            num_experts=moe_experts,
            target_active=moe_top_k,
            l1_coeff_init=remoe_l1_coeff,
            l1_coeff_multiplier=remoe_l1_multiplier,
            down_init_gain=down_init_gain,
        )
    raise ValueError(
        f"ffn_type must be 'pswiglu' or 'remoe', got {ffn_type!r}")


class RankMixerNetwork(nn.Module):
    """RankMixer cross: tokenize -> L blocks -> aggregate (paper Sec. 3).

    Replaces the DCNv2 cross. Takes a pre-built ``tokenizer`` (contiguous or
    semantic) that maps the flat (B, input_dim) vector to (B, T, D); ``T`` and
    the head count are derived from it. ``pool='mean'`` follows the paper
    (mean-pool the final tokens -> (B, D)); ``pool='flatten'`` keeps the full
    grid -> (B, T*D). ``output_dim`` is exposed so the MLP head can be sized to it.
    """

    def __init__(
        self,
        tokenizer: nn.Module,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        expansion: int,
        dropout: float = 0.0,
        pool: str = 'mean',
        ffn_type: str = 'pswiglu',
        down_init_gain: float = _DOWN_MATRIX_INIT_GAIN,
        norm_type: str = 'ln',
        moe_experts: int = 4,
        moe_top_k: int = 2,
        remoe_l1_coeff: float = 1e-8,
        remoe_l1_multiplier: float = 1.2,
    ) -> None:
        super().__init__()
        if pool not in ('mean', 'flatten'):
            raise ValueError(f"pool must be 'mean' or 'flatten', got {pool!r}")
        self.pool = pool
        self.tokenizer = tokenizer
        num_tokens = tokenizer.num_tokens
        self.num_tokens = num_tokens
        self.blocks = nn.ModuleList([
            RankMixerBlock(num_tokens, token_dim, num_heads, expansion, dropout,
                           ffn_type=ffn_type, down_init_gain=down_init_gain,
                           norm_type=norm_type, moe_experts=moe_experts,
                           moe_top_k=moe_top_k,
                           remoe_l1_coeff=remoe_l1_coeff,
                           remoe_l1_multiplier=remoe_l1_multiplier)
            for _ in range(num_layers)
        ])
        self.output_dim = token_dim if pool == 'mean' else num_tokens * token_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        if self.pool == 'mean':
            return tokens.mean(dim=1)
        return tokens.reshape(tokens.shape[0], -1)


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that tolerates singleton training batches."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and self.track_running_stats and input.shape[0] <= 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(input)


class ChannelSENETLayer(nn.Module):
    """Squeeze-and-excitation over a flattened feature vector."""

    def __init__(self, input_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [input_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.net(x))


class FeatureFieldTokenizer(nn.Module):
    """One embedding field per sparse fid, with optional dense-weighted pooling."""

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        emb_dim: int,
        emb_skip_threshold: int = 0,
        pair_weighted_dense_offsets: Optional[dict] = None,
        pair_pool: str = 'mean',
    ) -> None:
        super().__init__()
        if pair_pool not in ('sum', 'mean'):
            raise ValueError(f"pair_pool must be 'sum' or 'mean', got {pair_pool!r}")
        self.feature_specs = feature_specs
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold
        self.pair_weighted_dense_offsets = (
            dict(pair_weighted_dense_offsets) if pair_weighted_dense_offsets else {}
        )
        self.pair_pool = pair_pool

        embs = []
        for vs, offset, length in feature_specs:
            del offset, length
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

    def forward(
        self,
        int_feats: torch.Tensor,
        dense_feats_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fields = []
        for fid_idx, (vs, offset, length) in enumerate(self.feature_specs):
            del vs
            real_idx = self._emb_index[fid_idx]
            if real_idx == -1:
                fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim, dtype=torch.float)
            else:
                emb_layer = self.embs[real_idx]
                if length == 1:
                    fid_emb = emb_layer(int_feats[:, offset].long())
                else:
                    vals = int_feats[:, offset:offset + length].long()
                    emb_all = emb_layer(vals)
                    if fid_idx in self.pair_weighted_dense_offsets and dense_feats_raw is not None:
                        dense_off = self.pair_weighted_dense_offsets[fid_idx]
                        weights = dense_feats_raw[:, dense_off:dense_off + length]
                        mask = (vals != 0).float()
                        weights = weights * mask
                        if self.pair_pool == 'sum':
                            fid_emb = (emb_all * weights.unsqueeze(-1)).sum(dim=1)
                        else:
                            w_sum = weights.sum(dim=1).clamp(min=1e-6)
                            fid_emb = ((emb_all * weights.unsqueeze(-1)).sum(dim=1)
                                       / w_sum.unsqueeze(-1))
                    else:
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
            fields.append(fid_emb)
        return torch.cat(fields, dim=-1)


def _build_mlp(
    dims: List[int],
    dropout: float,
    zero_init_last: bool = False,
) -> nn.Sequential:
    """Flat MLP: per hidden layer ``Linear -> BatchNorm -> SiLU -> Dropout``, with
    a bare final ``Linear``. ``zero_init_last`` zeros the final Linear's weight+bias
    (used by the two-stream side head so its logit contribution starts at exactly 0
    = identity graft)."""
    layers: List[nn.Module] = []
    last_linear = None
    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i + 1])
        layers.append(linear)
        last_linear = linear
        if i < len(dims) - 2:
            layers.extend([
                SafeBatchNorm1d(dims[i + 1]),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
    if zero_init_last and last_linear is not None:
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
    return nn.Sequential(*layers)


class GroupBilinearFusion(nn.Module):
    """FinalMLP-style group-wise bilinear interaction between the two stream
    hidden vectors (arXiv 2304.00902, the module its ablation calls the single
    biggest contributor — "bilinear fusion plays a more important role than the
    feature selection", and which beats sum/concat fusion).

    Computes ONLY the pure cross term ``sum_g o1_g^T W_g o2_g``: the model already
    carries each stream's own logit (deep_logit, wide_logit), so FinalMLP's
    linear/bias fusion terms ``w1^T o1 + w2^T o2 + b`` are already represented and
    are omitted here. We add this term as a RESIDUAL: ``final = deep_logit +
    wide_logit + bilinear(deep_h, wide_h)``.

    ``o1[B,d1]`` / ``o2[B,d2]`` are split into ``groups`` subspaces with one small
    weight per group, cutting params from O(d1*d2*A) to O(d1*d2*A/groups) (a
    low-rank regularizer, helpful in our overfit regime). The weight is
    ZERO-INITIALIZED, so the fusion contributes exactly 0 at init -> stacking it on
    the proven sum-fusion two-stream model is byte-identical at start (confound-free
    A/B; the gradient w.r.t. the weight is non-zero so it still learns)."""

    def __init__(self, d1: int, d2: int, action_num: int, groups: int):
        super().__init__()
        if groups <= 0:
            raise ValueError(f"bilinear_groups must be > 0, got {groups}")
        if d1 % groups != 0 or d2 % groups != 0:
            raise ValueError(
                f"bilinear_groups={groups} must divide BOTH stream hidden dims "
                f"(deep_h={d1}, wide_h={d2}); pick a common divisor.")
        self.groups = groups
        self.p = d1 // groups
        self.q = d2 // groups
        self.action_num = action_num
        # [groups, action_num, p, q]; zero-init -> 0 contribution at init.
        # Keep the matrix dims last so the Muon optimizer routes this bank of
        # fusion matrices to Muon instead of the low-lr AdamW fallback.
        self.weight = nn.Parameter(torch.zeros(groups, action_num, self.p, self.q))
        # The size-1 action dim makes einsum's weight-grad come back with stride
        # (.,16,16,1) on that dim while DDP's bucket view expects the contiguous
        # (.,256,16,1) -> the harmless but noisy "grad strides do not match bucket
        # view" perf warning under DDP. NOTE: plain .contiguous() is a NO-OP here
        # (torch ignores a size-1 dim's stride, so it thinks the grad is already
        # contiguous); we must FORCE the standard layout with clone(contiguous_format)
        # so the LITERAL strides match the bucket. Pure layout (grad VALUES unchanged)
        # -> training byte-identical, no retrain / no ckpt-shape change.
        self.weight.register_hook(
            lambda grad: grad.clone(memory_format=torch.contiguous_format))

    def forward(self, o1: torch.Tensor, o2: torch.Tensor) -> torch.Tensor:
        b = o1.shape[0]
        o1 = o1.reshape(b, self.groups, self.p)
        o2 = o2.reshape(b, self.groups, self.q)
        # out[b,a] = sum_{g,p,q} o1[b,g,p] * weight[g,a,p,q] * o2[b,g,q]
        return torch.einsum('bgp,gapq,bgq->ba', o1, self.weight, o2)


class PCVRDINConcatSENETDCN(nn.Module):
    """Direct-concat DIN architecture with dense-embedding fids only."""

    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",
        seq_feature_ids: Optional[Dict[str, List[int]]] = None,
        d_model: int = 64,
        emb_dim: int = 64,
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        action_num: int = 1,
        num_time_buckets: int = 65,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        pair_weighted_dense_offsets: Optional[dict] = None,
        pair_pool: str = 'mean',
        seq_hash_buckets: int = 0,
        user_dense_keep_offsets: Optional[List[Tuple[int, int]]] = None,
        concat_senet_dims: Optional[List[int]] = None,
        concat_mlp_dims: Optional[List[int]] = None,
        two_stream_mlp: bool = False,
        two_stream_dims: Optional[List[int]] = None,
        two_stream_input: str = 'senet',
        two_stream_fusion: str = 'sum',
        bilinear_groups: int = 8,
        rankmixer_tokens: int = 16,
        rankmixer_dim: int = 128,
        rankmixer_heads: int = 0,
        rankmixer_layers: int = 2,
        rankmixer_expansion: int = 4,
        rankmixer_pool: str = 'mean',
        rankmixer_ffn: str = 'pswiglu',
        rankmixer_down_init_gain: float = _DOWN_MATRIX_INIT_GAIN,
        rankmixer_moe_experts: int = 4,
        rankmixer_moe_top_k: int = 2,
        rankmixer_remoe_l1_coeff: float = 1e-8,
        rankmixer_remoe_l1_multiplier: float = 1.2,
        rankmixer_norm: str = 'ln',
        rankmixer_tokenize: str = 'semantic',
        rankmixer_group_tokens: Optional[List[int]] = None,
        rankmixer_seq_per_token: bool = False,
        use_senet: bool = True,
        use_input_bn: bool = True,
        seq_dual_din_domains: Optional[List[str]] = None,
        dense_per_fid: bool = False,
        user_dense_fid_layout: Optional[List[Tuple[int, int]]] = None,
        item_dense_fid_layout: Optional[List[Tuple[int, int]]] = None,
        user_dense_groups: Optional[str] = None,
        item_dense_groups: Optional[str] = None,
    ) -> None:
        super().__init__()
        del user_dense_dim  # kept in the build API (train/infer pass it); unused here
        # item_dense is an rd2 feature. Append it
        # RAW to the concat (symmetric with the kept user_dense tail); the dataset
        # has already applied its dense transform. dim 0 => no-op (e.g. demo data).
        self.item_dense_dim = int(item_dense_dim)
        self.has_item_dense = self.item_dense_dim > 0
        self.d_model = d_model
        self.rankmixer_seq_per_token = bool(rankmixer_seq_per_token)
        self.emb_dim = emb_dim
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.num_sequences = len(self.seq_domains)
        seq_feature_ids = seq_feature_ids or {}
        self.seq_feature_ids = {
            domain: list(seq_feature_ids.get(domain, range(len(seq_vocab_sizes[domain]))))
            for domain in self.seq_domains
        }
        self.num_time_buckets = num_time_buckets
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.seq_hash_buckets = seq_hash_buckets
        self.user_dense_keep_offsets = list(user_dense_keep_offsets or [])
        # dense per-fid re-tokenization: feed each dense FIELD as its own
        # RankMixer token instead of merging user_dense / item_dense into 2 blob
        # tokens before the token-mixing runs. OFF => byte-identical to base.
        self.dense_per_fid = bool(dense_per_fid)
        self.user_dense_fid_layout = list(user_dense_fid_layout or [])
        self.item_dense_fid_layout = list(item_dense_fid_layout or [])
        self.user_dense_groups = user_dense_groups
        self.item_dense_groups = item_dense_groups
        self.cross_arch = 'rankmixer'  # only cross architecture (DCNv2/FCN removed)

        self.user_field_tokenizer = FeatureFieldTokenizer(
            feature_specs=user_int_feature_specs,
            emb_dim=d_model,
            emb_skip_threshold=emb_skip_threshold,
            pair_weighted_dense_offsets=pair_weighted_dense_offsets,
            pair_pool=pair_pool,
        )
        self.item_field_tokenizer = FeatureFieldTokenizer(
            feature_specs=item_int_feature_specs,
            emb_dim=d_model,
            emb_skip_threshold=emb_skip_threshold,
        )
        self.num_ns = len(user_int_feature_specs) + len(item_int_feature_specs)

        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}
        self._seq_is_id = {}
        self._seq_is_hash = {}
        self._seq_vocab_sizes = {}
        self._seq_proj = nn.ModuleDict()
        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id, is_hash = self._make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_is_hash[domain] = is_hash
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        self.seq_din_extractors = nn.ModuleList([
            DINCrossAttention(
                d_model=d_model,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                ln_mode='pre',
            )
            for _ in range(self.num_sequences)
        ])
        # Optional dual-DIN: for the listed domains the newest-first head[:L]
        # slice is split into a recent half [0:L/2] and an older half [L/2:L];
        # each half gets its OWN DIN (so the older half is not attention-
        # dominated by the newest, unlike a single DIN over the full L). The two
        # pooled d_model vectors are CONCAT'd to a 2*d_model repr with NO
        # projection: when dual-DIN is on, the seq groups are split per-domain
        # (see the tokenizer block below), so the dual domain becomes its own
        # group of width 2*d_model and the tokenizer's Linear(2*d_model ->
        # token_dim=rankmixer_dim) projects it directly — no intermediate
        # d_model bottleneck. NORMAL init. Set the domain's L via --seq_max_lens
        # (e.g. seq_d:1024 -> half=512).
        self.seq_dual_din_domains = set(seq_dual_din_domains or [])
        unknown = self.seq_dual_din_domains - set(self.seq_domains)
        if unknown:
            raise ValueError(
                f"seq_dual_din_domains {sorted(unknown)} not in seq_domains "
                f"{self.seq_domains}")
        # For each dual domain the two halves' DINs are concat'd to a 2*d_model
        # repr (NO projection): the seq groups are split per-domain (below), so the
        # dual domain becomes its own group of width 2*d_model that the tokenizer's
        # Linear(2*d_model -> token_dim=rankmixer_dim) projects directly.
        self.seq_dual_din = nn.ModuleDict()
        for _domain in sorted(self.seq_dual_din_domains):
            self.seq_dual_din[_domain] = DINCrossAttention(
                d_model=d_model,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                ln_mode='pre',
            )
        dense_keep_dim = sum(length for _, length in self.user_dense_keep_offsets)
        # Each seq domain contributes d_model, except a dual-DIN domain, which
        # contributes 2*d_model (recent + older halves concat'd, no projection).
        seq_repr_dim = sum(
            2 * d_model if dom in self.seq_dual_din_domains else d_model
            for dom in self.seq_domains
        )
        self.input_dim = (
            len(user_int_feature_specs) * d_model
            + len(item_int_feature_specs) * d_model
            + seq_repr_dim
            + dense_keep_dim
            + self.item_dense_dim
        )
        concat_senet_dims = list(concat_senet_dims or [1024, 256])
        concat_mlp_dims = list(concat_mlp_dims or [1024, 512, 256, 128])
        # BN / SENET are SeFCN-heritage flat-vector preprocessing. Kept ON: they
        # normalize the raw heterogeneous concat scale BEFORE the RankMixer
        # tokenizer (RankMixer's own LayerNorms act on tokens, AFTER the
        # tokenizer). Dropping them was tested and made RankMixer much worse.
        self.use_input_bn = bool(use_input_bn)
        self.use_senet = bool(use_senet)
        self.input_bn = (
            SafeBatchNorm1d(self.input_dim) if self.use_input_bn else None)
        self.senet = (
            ChannelSENETLayer(self.input_dim, concat_senet_dims)
            if self.use_senet else None)
        # RankMixer cross (paper 2507.15551): tokenize the post-SENET vector
        # -> L blocks of (parameter-free token mixing + per-token FFN) -> pool.
        # heads default to num_tokens (paper's H = T, shape-preserving mixing).
        # RankMixer is now the ONLY cross architecture (DCNv2 / FCN removed).
        self.rankmixer = None
        if self.cross_arch == 'rankmixer':
            if rankmixer_tokenize == 'semantic':
                # Split at the natural concat boundaries of _make_concat_features:
                # [user_emb | item_emb | seq | user_dense | item_dense]. When
                # dual-DIN is on, the single 'seq' group is SPLIT per behavior
                # domain (seq_<name>), so a dual domain forms its own group of
                # width 2*d_model that the tokenizer projects straight to
                # token_dim (no d_model bottleneck). item_dense is dim 0 when the
                # dataset has none (then its requested tokens are forced to 0).
                split_seq = bool(self.seq_dual_din_domains)
                if split_seq:
                    seq_group_dims = [
                        (dom, 2 * d_model if dom in self.seq_dual_din_domains else d_model)
                        for dom in self.seq_domains
                    ]
                else:
                    seq_group_dims = [('seq', self.num_sequences * d_model)]
                head_groups = [
                    ('user_emb', len(user_int_feature_specs) * d_model),
                    ('item_emb', len(item_int_feature_specs) * d_model),
                    *seq_group_dims,
                ]
                # Per-group token counts are REQUIRED (no auto-split). WITHOUT
                # dense_per_fid: one count per group in group_dims order
                # [user_emb, item_emb, seq..., user_dense, item_dense]. WITH
                # dense_per_fid: the user_dense / item_dense blocks are expanded
                # to one tokenizer group PER dense field (or per ',' cluster) so
                # the token-mixing sees each dense field as its own token, and
                # rankmixer_group_tokens supplies ONLY the head counts
                # [user_emb, item_emb, seq...]; each dense sub-group gets 1 token.
                if self.dense_per_fid:
                    ud_groups = _expand_dense_groups(
                        self.user_dense_fid_layout, self.user_dense_groups,
                        'ud', dense_keep_dim)
                    id_groups = (
                        _expand_dense_groups(
                            self.item_dense_fid_layout, self.item_dense_groups,
                            'id', self.item_dense_dim)
                        if self.item_dense_dim > 0 else [])
                    dense_groups = ud_groups + id_groups
                    group_dims = head_groups + dense_groups
                    if rankmixer_group_tokens is None:
                        raise ValueError(
                            "dense_per_fid requires rankmixer_group_tokens: "
                            f"{len(head_groups)} head counts for "
                            f"{[n for n, _ in head_groups]}.")
                    if len(rankmixer_group_tokens) != len(head_groups):
                        raise ValueError(
                            "dense_per_fid: rankmixer_group_tokens must have "
                            f"{len(head_groups)} head entries for "
                            f"{[n for n, _ in head_groups]}, got "
                            f"{len(rankmixer_group_tokens)} (each dense "
                            "sub-group is auto-assigned 1 token)")
                    group_tokens = [
                        int(t) if d > 0 else 0
                        for t, (_, d) in zip(rankmixer_group_tokens, head_groups)
                    ] + [1 if w > 0 else 0 for _, w in dense_groups]
                else:
                    group_dims = head_groups + [
                        ('user_dense', dense_keep_dim),
                        ('item_dense', self.item_dense_dim),
                    ]
                    if rankmixer_group_tokens is None:
                        raise ValueError(
                            "cross_arch=rankmixer with rankmixer_tokenize='semantic' "
                            f"requires rankmixer_group_tokens: {len(group_dims)} "
                            f"per-group counts for {[n for n, _ in group_dims]}.")
                    if len(rankmixer_group_tokens) != len(group_dims):
                        raise ValueError(
                            f"rankmixer_group_tokens must have {len(group_dims)} "
                            f"entries for groups {[n for n, _ in group_dims]}, got "
                            f"{len(rankmixer_group_tokens)}")
                    # Empty groups (dim 0, e.g. item_dense when the dataset has
                    # none) are forced to 0 tokens regardless of the count.
                    group_tokens = [
                        int(t) if d > 0 else 0
                        for t, (_, d) in zip(rankmixer_group_tokens, group_dims)
                    ]
                if sum(group_tokens) < 1:
                    raise ValueError(
                        "rankmixer_group_tokens sum to 0 after dropping empty "
                        "groups")
                # When the seq group is NOT split, optionally give each seq token
                # its own Linear (one per behavior domain). With split_seq each
                # seq domain is already its own 1-token group, so per_token is moot.
                per_token_groups = (
                    ['seq'] if (self.rankmixer_seq_per_token and not split_seq)
                    else None)
                tokenizer = RankMixerSemanticTokenizer(
                    group_dims, group_tokens, rankmixer_dim,
                    per_token_groups=per_token_groups)
            elif rankmixer_tokenize == 'contiguous':
                tokenizer = RankMixerTokenizer(
                    self.input_dim, rankmixer_tokens, rankmixer_dim)
            else:
                raise ValueError(
                    "rankmixer_tokenize must be 'semantic' or 'contiguous', "
                    f"got {rankmixer_tokenize!r}")
            num_tokens = tokenizer.num_tokens
            rm_heads = rankmixer_heads if rankmixer_heads and rankmixer_heads > 0 else num_tokens
            self.rankmixer = RankMixerNetwork(
                tokenizer=tokenizer,
                token_dim=rankmixer_dim,
                num_heads=rm_heads,
                num_layers=rankmixer_layers,
                expansion=rankmixer_expansion,
                dropout=dropout_rate,
                pool=rankmixer_pool,
                ffn_type=rankmixer_ffn,
                down_init_gain=rankmixer_down_init_gain,
                moe_experts=rankmixer_moe_experts,
                moe_top_k=rankmixer_moe_top_k,
                remoe_l1_coeff=rankmixer_remoe_l1_coeff,
                remoe_l1_multiplier=rankmixer_remoe_l1_multiplier,
                norm_type=rankmixer_norm,
            )

        mlp_in = self.rankmixer.output_dim
        mlp_dims = [mlp_in] + concat_mlp_dims + [action_num]
        mlp_layers = []
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                mlp_layers.extend([
                    SafeBatchNorm1d(mlp_dims[i + 1]),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                ])
        self.mlp = nn.Sequential(*mlp_layers)
        # Two-stream (Wide&Deep) side head (ported from teammate's 2mlp): a shallow
        # MLP from the FLAT feature vector straight to the logit, ADDED to the
        # RankMixer-deep logit. Side input = raw / bn / post-SENET (default senet).
        # zero_init_last => the side logit starts at 0, so OFF and ON-at-init are
        # byte-identical to the deep-only model (a confound-free A/B). _init_params
        # only re-inits embeddings, NOT these Linears, so the zero-init survives.
        if two_stream_input not in ('raw', 'bn', 'senet'):
            raise ValueError(
                "two_stream_input must be one of raw/bn/senet, got "
                f"{two_stream_input!r}")
        self.two_stream_mlp_enabled = bool(two_stream_mlp)
        self.two_stream_input = two_stream_input
        two_stream_dims = list(two_stream_dims or [512, 128])
        self.two_stream_mlp = None
        if self.two_stream_mlp_enabled:
            self.two_stream_mlp = _build_mlp(
                [self.input_dim] + two_stream_dims + [action_num],
                dropout=dropout_rate,
                zero_init_last=True,
            )
        # Two-stream FUSION: 'sum' = deep_logit + wide_logit (current best,
        # byte-identical path); 'bilinear' = ALSO add FinalMLP's group-wise
        # bilinear interaction of the two stream HIDDEN vectors as a zero-init
        # residual (final = deep_logit + wide_logit + bilinear(deep_h, wide_h)).
        # deep_h width = mlp_dims[-2] (deep head's penultimate layer); wide_h width
        # = two_stream_dims[-1] (side MLP's penultimate). bilinear needs both
        # streams -> requires the two-stream head.
        if two_stream_fusion not in ('sum', 'bilinear'):
            raise ValueError(
                f"two_stream_fusion must be 'sum' or 'bilinear', got "
                f"{two_stream_fusion!r}")
        self.two_stream_fusion = two_stream_fusion
        self.bilinear_fusion = None
        if two_stream_fusion == 'bilinear':
            if not self.two_stream_mlp_enabled:
                raise ValueError(
                    "two_stream_fusion='bilinear' requires the two-stream head "
                    "(two_stream_mlp=True): it fuses the two streams' hidden "
                    "vectors.")
            self.bilinear_fusion = GroupBilinearFusion(
                d1=mlp_dims[-2], d2=two_stream_dims[-1],
                action_num=action_num, groups=bilinear_groups)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self._init_params()

    def _make_seq_embs(self, vocab_sizes):
        embs_raw = []
        is_hash_raw = []
        for vs in vocab_sizes:
            vs_i = int(vs)
            if vs_i <= 0:
                embs_raw.append(None)
                is_hash_raw.append(False)
            elif self.emb_skip_threshold > 0 and vs_i > self.emb_skip_threshold:
                if self.seq_hash_buckets > 0:
                    embs_raw.append(nn.Embedding(self.seq_hash_buckets + 1, self.emb_dim, padding_idx=0))
                    is_hash_raw.append(True)
                else:
                    embs_raw.append(None)
                    is_hash_raw.append(False)
            else:
                embs_raw.append(nn.Embedding(vs_i + 1, self.emb_dim, padding_idx=0))
                is_hash_raw.append(False)
        module_list = nn.ModuleList([e for e in embs_raw if e is not None])
        index_map = []
        is_hash = []
        real_idx = 0
        for e, h in zip(embs_raw, is_hash_raw):
            if e is not None:
                index_map.append(real_idx)
                is_hash.append(h)
                real_idx += 1
            else:
                index_map.append(-1)
                is_hash.append(False)
        is_id = [int(vs) > self.seq_id_threshold for vs in vocab_sizes]
        return module_list, index_map, is_id, is_hash

    def _init_params(self) -> None:
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
        for tokenizer in [self.user_field_tokenizer, self.item_field_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(self, cardinality_threshold: int = 10000) -> "set[int]":
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()
        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1
        for tokenizer in [self.user_field_tokenizer, self.item_field_tokenizer]:
            for i, (vs, offset, length) in enumerate(tokenizer.feature_specs):
                del offset, length
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1
        if self.num_time_buckets > 0:
            skip_count += 1
        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _remoe_modules(self) -> List[RankMixerReMoEFFN]:
        return [m for m in self.modules() if isinstance(m, RankMixerReMoEFFN)]

    def consume_aux_loss(self) -> Optional[torch.Tensor]:
        losses = [
            loss
            for module in self._remoe_modules()
            for loss in [module.consume_aux_loss()]
            if loss is not None
        ]
        if not losses:
            return None
        total = losses[0]
        for loss in losses[1:]:
            total = total + loss
        return total

    @torch.no_grad()
    def update_remoe_l1_coeff(self) -> None:
        modules = self._remoe_modules()
        if not modules:
            return
        sparsities = [
            module._last_sparsity
            for module in modules
            if module._last_sparsity is not None
        ]
        if not sparsities:
            return
        avg_sparsity = torch.stack([
            s.to(device=sparsities[0].device, dtype=torch.float32)
            for s in sparsities
        ]).mean()
        for module in modules:
            module.update_l1_coeff(avg_sparsity)

    def remoe_router_stats(self) -> Dict[str, Any]:
        per_layer = [module.router_stats() for module in self._remoe_modules()]
        per_layer = [s for s in per_layer if s]
        if not per_layer:
            return {}
        # Aggregate the scalar keys across layers (mean) for the summary line;
        # expose the per-layer dicts (incl. per-expert load/weight lists) under
        # 'per_layer' so the trainer can log each expert slot separately.
        scalar_keys = [
            key for key, value in per_layer[0].items()
            if not isinstance(value, list)
        ]
        agg: Dict[str, Any] = {
            key: sum(s[key] for s in per_layer) / len(per_layer)
            for key in scalar_keys
        }
        agg['per_layer'] = per_layer
        return agg

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        is_hash: List[bool],
        skip_feature_indices: Optional[set] = None,
    ) -> torch.Tensor:
        B, S, L = seq.shape
        emb_list = []
        H = self.seq_hash_buckets
        skip_feature_indices = skip_feature_indices or set()
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1 or i in skip_feature_indices:
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                ids = seq[:, i, :]
                if is_hash[i]:
                    non_pad = ids != 0
                    ids = torch.where(non_pad, ((ids - 1) % H) + 1, ids)
                e = emb(ids)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)
        token_emb = F.gelu(proj(cat_emb))
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)
        return token_emb

    def _make_padding_mask(self, seq_len: torch.Tensor, max_len: int) -> torch.Tensor:
        idx = torch.arange(max_len, device=seq_len.device).unsqueeze(0)
        return idx >= seq_len.unsqueeze(1)

    def _slice_kept_dense(self, user_dense_feats: torch.Tensor) -> torch.Tensor:
        if not self.user_dense_keep_offsets:
            return user_dense_feats.new_zeros(user_dense_feats.shape[0], 0)
        chunks = [
            user_dense_feats[:, offset:offset + length]
            for offset, length in self.user_dense_keep_offsets
        ]
        return torch.cat(chunks, dim=-1)

    def _make_concat_features(
        self,
        inputs: ModelInput,
        apply_dropout: bool,
    ) -> torch.Tensor:
        user_flat = self.user_field_tokenizer(
            inputs.user_int_feats,
            dense_feats_raw=inputs.user_dense_feats_raw,
        )
        item_flat = self.item_field_tokenizer(inputs.item_int_feats)
        kept_dense = self._slice_kept_dense(inputs.user_dense_feats)

        B = item_flat.shape[0]
        item_fields = item_flat.view(B, -1, self.d_model)
        target = item_fields.mean(dim=1, keepdim=True)

        seq_reprs = []
        for domain, extractor in zip(self.seq_domains, self.seq_din_extractors):
            seq_tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain],
                self._seq_proj[domain],
                self._seq_is_id[domain],
                self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                self._seq_is_hash[domain],
            )
            seq_mask = self._make_padding_mask(
                inputs.seq_lens[domain],
                inputs.seq_data[domain].shape[2],
            )
            seq_tokens_for_din = self.emb_dropout(seq_tokens) if apply_dropout else seq_tokens
            if domain in self.seq_dual_din_domains:
                # recent half [0:L/2] + older half [L/2:L], each pooled by its
                # own DIN, then concat to a 2*d_model repr (NO projection — the
                # per-domain tokenizer Linear projects it straight to token_dim).
                L = seq_tokens_for_din.shape[1]
                half = L // 2
                rep_recent = extractor(
                    target, seq_tokens_for_din[:, :half], seq_mask[:, :half]).squeeze(1)
                rep_older = self.seq_dual_din[domain](
                    target, seq_tokens_for_din[:, half:], seq_mask[:, half:]).squeeze(1)
                seq_reprs.append(torch.cat([rep_recent, rep_older], dim=-1))
            else:
                seq_reprs.append(
                    extractor(target, seq_tokens_for_din, seq_mask).squeeze(1))

        seq_flat = torch.cat(seq_reprs, dim=-1)
        parts = [user_flat, item_flat, seq_flat, kept_dense]
        if self.has_item_dense:
            # rd2 item_dense (raw; already transformed by the dataset). Must be the
            # LAST segment to match the semantic tokenizer's group order.
            parts.append(inputs.item_dense_feats)
        x = torch.cat(parts, dim=-1)
        if apply_dropout:
            x = self.emb_dropout(x)
        return x

    def _forward_impl(self, inputs: ModelInput, apply_dropout: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._make_concat_features(inputs, apply_dropout=apply_dropout)
        raw_x = x
        if self.input_bn is not None:
            x = self.input_bn(x)
        bn_x = x
        if self.senet is not None:
            x = self.senet(x)
        side_x = x
        if self.two_stream_mlp is not None:
            if self.two_stream_input == 'raw':
                side_x = raw_x
            elif self.two_stream_input == 'bn':
                side_x = bn_x
        x = self.rankmixer(x)
        if self.two_stream_mlp is not None:
            if self.two_stream_fusion == 'bilinear':
                # Split each stream's bare final Linear off to expose its
                # penultimate hidden vector. mlp[:-1] then mlp[-1] is exactly
                # mlp(x) (Sequential runs in order) -> deep_logit/wide_logit are
                # numerically unchanged; we just also feed deep_h/wide_h to the
                # zero-init bilinear residual. mlp[:-1] is a transient view, so
                # state_dict keys (mlp.*, two_stream_mlp.*) are unchanged.
                deep_h = self.mlp[:-1](x)
                deep_logit = self.mlp[-1](deep_h)
                wide_h = self.two_stream_mlp[:-1](side_x)
                wide_logit = self.two_stream_mlp[-1](wide_h)
                logits = deep_logit + wide_logit + self.bilinear_fusion(deep_h, wide_h)
            else:
                deep_logit = self.mlp(x)
                wide_logit = self.two_stream_mlp(side_x)
                logits = deep_logit + wide_logit
        else:
            deep_logit = self.mlp(x)
            logits = deep_logit
        return logits, x

    def forward(self, inputs: ModelInput):
        logits, _ = self._forward_impl(inputs, apply_dropout=self.training)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._forward_impl(inputs, apply_dropout=False)


class PCVRRankMixer(PCVRDINConcatSENETDCN):
    """Backwards-compatible alias for the RankMixer model (arXiv 2507.15551).

    The model is now RankMixer-only, so this is identical to its parent
    ``PCVRDINConcatSENETDCN``; the name is kept for checkpoint / log lineage and
    existing import sites. Pipeline: features -> input BN -> ChannelSENET ->
    semantic tokenizer -> RankMixer blocks -> mean-pool -> MLP head.
    """
