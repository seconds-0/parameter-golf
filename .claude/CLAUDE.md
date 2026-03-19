# Parameter Golf

## Competition
OpenAI's Parameter Golf: train the best language model that fits in a **16MB artifact** and trains in **under 10 minutes on 8xH100 GPUs**. Evaluated by bits-per-byte compression on FineWeb validation. Runs March 18 – April 30, 2026.

- **Metric**: `val_bpb` (bits per byte), lower is better
- **Baseline**: 1.2244 bpb
- **Artifact cap**: 16,000,000 bytes (code + int8 zlib-compressed model)
- **Records must beat SOTA by ≥0.005 nats with p < 0.01**
- **Compute credits**: https://openai.com/index/parameter-golf/#credit-form

## Baseline Architecture
- 9 transformer layers, 512 dim, 8 heads, 4 KV heads (GQA), 2x MLP expansion
- 1024 vocab (SentencePiece BPE), 1024 seq len, tied input/output embeddings
- U-Net skip connections: encoder half stores skips, decoder half consumes them reversed
- ReLU² activation in MLP, RMSNorm (no learned weight), RoPE, logit softcapping
- Learned per-block residual mix, attention/MLP scales, per-head Q gain
- ~15.8MB compressed at int8+zlib

## Optimizer
- **Muon**: matrix params — SGD momentum + Newton-Schulz orthogonalization
- **Adam**: embeddings (lr=0.05) + scalar/control params (lr=0.04)
- Warmdown: cosine decay based on remaining wallclock time

## Key Files
| File | Purpose |
|------|---------|
| `train_gpt.py` | Main CUDA training script (8xH100 via torchrun) |
| `train_gpt_mlx.py` | Apple Silicon variant — **DO NOT RUN LOCALLY, crashes machine** |
| `data/cached_challenge_fineweb.py` | Download FineWeb dataset shards from HuggingFace |
| `data/tokenizers/fineweb_1024_bpe.model` | SentencePiece tokenizer (1024 vocab) |
| `records/track_10min_16mb/` | Official record submissions |
| `records/track_non_record_16mb/` | Experimental/unlimited-compute submissions |

## Running on Cloud GPU (8xH100)
```bash
# Download data
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train (defaults: 10min cap, 524K batch tokens, 20K iterations)
RUN_ID=my_run \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are configurable via environment variables. Key ones:
`NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `MLP_MULT`, `VOCAB_SIZE`,
`TRAIN_BATCH_TOKENS`, `TRAIN_SEQ_LEN`, `ITERATIONS`, `MAX_WALLCLOCK_SECONDS`,
`MATRIX_LR`, `SCALAR_LR`, `TIED_EMBED_LR`, `MUON_MOMENTUM`, `GRAD_ACCUM_STEPS`

## Quantization Pipeline
- Per-row int8 for 2D weight matrices (clip at 99.99984th percentile)
- Per-tensor int8 for vectors/scalars
- fp16 passthrough for small tensors (<65K elements)
- fp32 passthrough for control tensors (scales, gains, mix weights)
- zlib level 9 compression on the pickled result

## Submission Format
PR adding a folder to `records/track_10min_16mb/` or `records/track_non_record_16mb/` with:
1. `README.md` — approach explanation
2. `submission.json` — metadata (author, github_id, val_bpb, bytes_total, etc.)
3. `train.log` — full training log
4. `train_gpt.py` — complete training script (must compile and run standalone)

## Local Dev
**MLX training works locally after fixing a lazy-eval bug** (replaced `mx.synchronize()` with `mx.eval(...)` in training loop).
Safe settings for M4 Pro 48GB: `TRAIN_BATCH_TOKENS=8192-65536`, `VAL_BATCH_SIZE=65536`.
For quick smoke tests: `VAL_LOSS_EVERY=0` skips periodic validation (final val is slow at tiny batch sizes).
All official training runs still go to cloud GPUs (Vast.ai or Prime Intellect).

## Current State — READ ON RESUME

**Start here:** `docs/tracker.md` — master progress tracker with all work items and status.

- **Strategy**: `docs/experiment_plan_prd.md` — full experiment plan (proxy design, sequential halving, 23 experiments)
- **Track details**: `docs/tracks/track_{a-f}_*.md` — per-track thesis, experiments, decision rules, learnings
- **Key metrics**: `Δpq` (post-roundtrip delta vs baseline, negative=better), `qgap` (post-quant minus pre-quant, smaller=better)
- **Rule**: Never promote on pre-quant improvement alone. Export retention is the battlefield.
- **Budget**: Conservative plan ~12-15 H100-hours. No grids. Sequential halving only.

## Git Setup
- `origin`: `seconds-0/parameter-golf` (our fork)
- `upstream`: `openai/parameter-golf` (official repo)
- Python 3.11.9 via pyenv, venv at `.venv/`
