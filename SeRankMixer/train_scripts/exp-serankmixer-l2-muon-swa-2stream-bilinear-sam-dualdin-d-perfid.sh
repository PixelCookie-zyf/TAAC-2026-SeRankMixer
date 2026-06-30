#!/bin/bash
# PER-FID DENSE RE-TOKENIZATION (candidate A) + width 1152.
# 🟢 RESULT 2026-06-24: "24t-1152" (this config) = NEW BEST Test 0.828814
#    (+万1.8 over seq_d-dual 0.828634). per-fid REOPENED the capacity axis (dead
#    for blob: width 1536 / L3 / PKM all overfit, 1024 was the blob knee) -> the
#    structured per-field tokenization lets a larger model NOT overfit. NEXT =
#    WIDTH-SWEEP this: bump --rankmixer_dim 1152 -> 1344 (must stay %24==0 for
#    T=24: 768/1152/1344/1536 ok, 1024/1280 not). Greedy: climb while it pays.
#
# Original single-variable design (vs 0.828634): --dense_per_fid (everything else
# identical: pw2 + bilinear + SWA + SAM + seq_d dual-DIN). NOTE per-fid forces
# T 16->24, and RankMixer's per-token FFN scales with T (+~55M at 768, more at
# 1152) -> per-fid is INSEPARABLE from "raise T" (a capacity axis); the +万1.8
# mixes per-field granularity + extra width/T. User opted NOT to split them.
#
# WHAT IT CHANGES: today user_dense (1765d, 12 kept fields) and item_dense (770d,
# 4 fields) are each merged by ONE Linear into 2 "blob" tokens BEFORE RankMixer's
# token-mixing runs -> the mixing can never form cross-field interactions, it
# only sees 4 averaged blobs. This expands the dense blocks to one token PER
# dense FIELD (big semantic vectors independent, tiny stat fids merged), so the
# token-mixing sees each field as its own token. Seq is already per-domain; this
# brings dense to the same per-field granularity.
#
# Layout (demo/platform schema, pair fids 62-66 already pulled out as pooling
# weights, fid 130 capped 1284->259):
#   user_dense kept: 61(256) 87(320) [89,90,91,118](60) 120(256) 121(100)
#                    123(128) 130(259) 131(258) 132(128)        = 9 tokens
#   item_dense:      124(128) [127,128](512) 129(130)           = 3 tokens
#   head:            user_emb 5 + item_emb 3 + seq_a/b/c/d 1·4  = 12 tokens
#   T = 12 + 9 + 3 = 24.  H=T and 768 % 24 == 0 (head_dim 32). OK.
# rankmixer_group_tokens now supplies ONLY the 6 HEAD counts "5,3,1,1,1,1";
# each dense sub-group is auto-assigned 1 token by --dense_per_fid.
#
# WHY THIS IS NOT A REFUTED AXIS: it is pure re-tokenization, NOT capacity. The
# dense tokenizer params DROP ~50% (1765*1536 + 770*1536 = 3.9M  ->  768*(1765
# +770) = 2.0M), because each field projects to ONE 768-token instead of the
# block projecting to 2*768. Params down, structure up = a pure inductive-bias
# change, orthogonal to width/depth/memory (dead), gates (dead), fusion variants
# (dead). EXPERIMENTS has NO dense-tokenization-granularity run -> genuinely new.
#
# WHY IT MIGHT BE FLAT: input-BN + SENET may already capture some cross-field
# signal, and the optimum is saturated (0.8286). HONEST PRIOR ~14% (shortlist
# top). Confound status: --dense_per_fid OFF => model is byte-identical to the
# seq_d-dual base (group_dims/group_tokens take the original else-branch); ON
# only changes the tokenizer group split, forward/backward unchanged.
#
# JUDGE best `.swa` Test, DEFAULT row-group-tail holdout, vs seq_d-dual 0.828634:
#   > 0.828634 -> dense per-field tokenization carries cross-field signal the
#                 blob tokens destroyed; adopt, then try T=32 + width-1024 stack.
#   <= 0.828634 -> dense-block internal structure is exhausted (~17th axis);
#                 revert (drop --dense_per_fid / delete script).
#
# NOTE width: this stays on rankmixer_dim 768 (768 % 24 == 0). width-1024 is a
# SEPARATE lever (it needs T in {16,32}, incompatible with T=24), validate it on
# its own; a 1024 + per-fid stack would use T=32 (dense = 20 sub-groups).
#
# Platform: upload SeRankMixer/, copy this script to SeRankMixer/run.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${SCRIPT_DIR}/train.py" ]]; then
  CODE_DIR="${SCRIPT_DIR}"
else
  CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# GPU detection
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES// /}" ]]; then
    NGPUS=0
  else
    NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
  fi
elif command -v nvidia-smi &> /dev/null; then
  NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
  NGPUS=0
fi
echo "Using ${NGPUS} GPU(s) (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}')"

TRAIN_ARGS=(
    --concat_senet_dims "1024,256"
    --pair_pool mean
    --concat_mlp_dims "1024,512,256,128"
    --rankmixer_tokenize semantic
    # dense_per_fid: head counts ONLY (user_emb,item_emb,seq_a,b,c,d). Each dense
    # sub-group (9 user_dense + 3 item_dense) auto-gets 1 token -> T = 12+12 = 24.
    --rankmixer_group_tokens "5,3,1,1,1,1"
    --rankmixer_seq_per_token 1
    --rankmixer_dim 1152      # 24t-1152 = NEW BEST 0.828814; width-sweep -> 1344 (%24)
    --rankmixer_heads 0
    --rankmixer_layers 2
    --rankmixer_expansion 2
    --rankmixer_pool mean
    --rankmixer_norm rmsnorm

    # -- the ONE lever under test (stacked on the seq_d-dual 0.828634 best) --
    --dense_per_fid
    --user_dense_groups "61|87|89,90,91,118|120|121|123|130|131|132"
    --item_dense_groups "124|127,128|129"

    # -- pw2 = current-best loss; fixed --
    --loss_type bce_pos_weight
    --pos_weight 2.0

    # -- dual-DIN on seq_d (confirmed positive); fixed --
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:1024"
    --seq_dual_din_domains "seq_d"

    # -- two-stream Wide&Deep side MLP + FinalMLP bilinear fusion --
    --two_stream_mlp 1
    --two_stream_dims "512,128"
    --two_stream_input senet
    --two_stream_fusion bilinear
    --bilinear_groups 8

    # -- base: Muon+L2@5e-4 + SWA + SAM + pw2 --
    --optimizer muon
    --muon_lr 5e-4
    --muon_momentum 0.95
    --muon_weight_decay 0.01
    --muon_ns_steps 5
    --use_swa
    --swa_start_epoch 2
    --swa_collect_every 200
    --swa_bn_batches 50
    --use_sam
    --sam_rho 0.05

    --emb_skip_threshold 1000000
    --num_workers 8
    --d_model 64
    --emb_dim 64
    --hidden_mult 4
    --dropout_rate 0.01
    --batch_size 1024

    --no_amp

    --longtail_transform log1p
    --weighted_pair_pool_fids "62,63,64,65,66"
    --longtail_fids "62,63,64,65,66"

    --save_every_epoch
)

if [[ "${NGPUS}" -gt 1 ]]; then
  echo "Detected ${NGPUS} GPUs, launching DDP with torchrun"
  torchrun --standalone --nproc_per_node="${NGPUS}" \
    "${CODE_DIR}/train.py" "${TRAIN_ARGS[@]}" "$@"
else
  echo "Single GPU / CPU mode"
  python3 -u "${CODE_DIR}/train.py" "${TRAIN_ARGS[@]}" "$@"
fi
