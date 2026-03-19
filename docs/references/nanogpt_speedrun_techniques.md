# NanoGPT Speedrun: Complete Technique Classification for Parameter Golf

- **Source:** [github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- **Accessed:** 2026-03-19
- **Records:** 77 world records, from 45 min → 1.435 min (31.3x speedup) for val_loss ≤ 3.28 on FineWeb, 8xH100
- **Our competition:** Parameter Golf — minimize val_bpb within 16MB artifact + 10 min on 8xH100

## How to Read This

The speedrun optimizes **wall-clock time to a fixed quality target**. Parameter Golf optimizes **quality within fixed time and artifact size**. Techniques that improve quality per compute are relevant; techniques that only reduce wall-clock (kernel fusions, communication overlaps) are not — we already fit in 10 minutes with headroom.

Each technique is classified into one of four tiers:
- **Tier 1:** High relevance — quality improvement, fits constraints, not yet in our codebase
- **Tier 2:** Medium — worth investigating but uncertain fit
- **Tier 3:** Already implemented in our baseline
- **Tier 4:** Not relevant (speed-only or wrong scale)

---

## Tier 1: HIGH — Not Yet Implemented, Quality Improvement, Fits Constraints

### 1.1 Multi-Token Prediction (MTP)

**What:** Predict 2+ future tokens simultaneously using additional lightweight prediction heads. The speedrun also uses a trick where embeddings are tied for the first 2/3 of training (small artifact) then untied to give the model a dedicated output head for the final 1/3.

**Speedrun impact:** Record 34 (2.203 → 1.988 min). Part of the `ForwardScheduleConfig` system with scheduled `mtp_weights`.

**Why it matters for us:** More learning signal per training step within our fixed 10-minute budget. Each step teaches the model about multiple positions simultaneously. The extra prediction heads are small linear layers. The "untie at 2/3" trick means the final model has a separate `lm_head` which adds parameters but may be worth the quality gain.

**Artifact concern:** An untied `lm_head` adds `vocab_size × model_dim` = 1024 × 512 = 524,288 params (~512KB at int8). Currently we have only 138KB slack. This technique would need to be combined with layer sharing (E18) to fit, OR the MTP heads could be discarded at export time (only used for training signal, not inference). If MTP heads are training-only, the artifact cost is zero.

**Implementation complexity:** Medium. Need additional linear heads in the model, modified loss computation (sum of per-position losses with scheduled weights), and careful handling at export time.

### 1.2 Cautious Weight Decay

**What:** Gated weight decay that only applies when the weight update direction aligns with the decay direction. Implemented as `_cautious_wd_and_update_inplace()` with per-parameter `wd_mul` scaling and tied to the learning rate schedule.

**Speedrun impact:** Record 44 (2.345 → 2.284 min). Extended to Adam parameters as well.

**Why it matters for us:** More principled than the fixed heavy weight decay planned in E24. Standard weight decay always pushes weights toward zero, even when the gradient says "grow this weight." Cautious decay only applies when weight and gradient agree on the direction, avoiding counterproductive updates. This could give better quality AND better quantization friendliness than a blanket L2 penalty.

**Artifact concern:** Zero — pure optimizer change.

**Implementation complexity:** Low. ~15 lines in the optimizer step function. The `wd_mul` per-parameter config lets us apply different decay strengths to different parameter groups (e.g., heavier on MLP weights, lighter on embeddings).

### 1.3 Batch Size Schedule

**What:** Progressive batch size increase over training. Start with smaller batches (more gradient updates, faster early learning), increase later (smoother convergence with lower gradient noise).

**Speedrun impact:** Record 37 (2.358 → 2.203 min). Implemented as `TrainingStage` dataclass with per-stage `batch_size`.

**Why it matters for us:** We use a fixed `TRAIN_BATCH_TOKENS=524288`. With our 8xH100 setup, smaller early batches mean more steps in the same wall-clock time. More steps = more gradient updates = faster early learning when the model is improving rapidly. Late in training when the model is near convergence, larger batches reduce gradient noise for final refinement.

**Artifact concern:** Zero — pure training schedule change.

**Implementation complexity:** Low-Medium. Need to modify the training loop to change `batch_size` (or equivalently `grad_accum_steps`) at specified iteration thresholds. The gradient accumulation logic already exists.

### 1.4 Sequence Length Schedule

**What:** Start training with shorter sequences, progressively increase to full context length. Short sequences mean faster attention (O(n²) → much cheaper), so more steps per second early in training.

**Speedrun impact:** Part of the `TrainingStage` system. `train_max_seq_len` increases across stages.

**Why it matters for us:** With 1024 seq len, attention is already cheap. But shorter sequences (256 or 512) early in training would still mean faster steps and more data seen in the first minutes. The model learns local patterns first (which short sequences capture) and then benefits from longer context later.

**Artifact concern:** Zero — pure training schedule change.

**Implementation complexity:** Medium. Need to modify the data loader to produce variable-length sequences and adjust the training loop. RoPE naturally handles variable lengths.

### 1.5 Value Embeddings / Extra Embeddings

**What:** Mix input (token) embeddings directly into attention value computations with learned gating. Instead of computing V only from the hidden state, also add a gated contribution from the original token embeddings. Implemented with `ve_gate_w` (value embedding gate weight) parameters.

**Speedrun impact:** Multiple records spanning significant improvements:
- Record 13 (5.03 → 4.41 min): Extra value embeddings introduced
- Record 16 (4.41 → 3.80 min): Split value embeddings
- Record 22 (2.812 → 2.547 min): Smear module + value embedding refinement

**Why it matters for us:** Cheap capacity increase. The embeddings are already computed; mixing them into values gives each attention layer direct access to token identity, which is especially useful in shallow networks (we have only 9 layers). The gating weights are tiny (one scalar or small vector per block) and stored as fp32 control tensors.

**Artifact concern:** Minimal. Gating weights are control tensors kept in fp32 (a few hundred bytes total). The embedding table is already stored. No new weight matrices needed.

**Implementation complexity:** Low-Medium. Add a gating parameter per block, modify the attention forward to mix in embeddings before or alongside the V projection.

### 1.6 Document-Aligned Batching

**What:** Align batch boundaries with end-of-document tokens instead of cutting at arbitrary positions. Prevents the model from attending across document boundaries within a sequence.

**Speedrun impact:** Record 28 (2.966 → 2.863 min).

**Why it matters for us:** Our `TokenStream` (train_gpt.py lines 451-479) reads shards sequentially and wraps around, with no awareness of document boundaries. This means some training sequences contain the end of one document and the start of another, teaching the model to attend across unrelated content. Document alignment is a pure data quality improvement.

**Artifact concern:** Zero — data loading change only.

**Implementation complexity:** Low. Need to detect EOS/document boundary tokens in the data stream and align sequences to start at document boundaries. SentencePiece should have a document separator token or we can use the existing tokenizer's EOS.

### 1.7 Bigram Hash Embedding

**What:** A second embedding table indexed by consecutive token pairs (bigrams), added alongside the standard unigram embeddings. Uses a hash function to map token pairs to a fixed-size table without storing all `vocab² `entries.

**Speedrun impact:** Record 38 (1.820 → 1.655 min). Implemented as `Shard` class with `get_bigram_hash()`.

**Why it matters for us:** Our 1024 vocab is very small — many common character/subword patterns span 2 tokens. A bigram embedding gives the model immediate access to pair-level patterns that otherwise require at least one attention layer to compose. The hash table size is configurable; a small table (e.g., 8K-16K entries × dim) could add meaningful capacity without blowing the artifact budget.

**Artifact concern:** Moderate. A bigram table of 8192 × 512 = 4,194,304 params would be ~4MB at int8. That's way too much. Would need a very small table (e.g., 2048 × 128 projected up = ~262K params, ~256KB). Need careful budget analysis. The speedrun uses sparse gradient communication for this, which may not be needed at our scale.

**Implementation complexity:** Medium. Need the hash function, the extra embedding table, an additive or gated combination with the unigram embeddings, and careful artifact sizing.

### 1.8 NorMuon + Polar Express (Muon Optimizer Upgrades)

**What:** Two complementary improvements to the Muon optimizer:
- **NorMuon:** Adds low-rank variance estimation (Adafactor-style) to Muon. Per-matrix learning rate multipliers based on shape ratio. Reference: arXiv:2510.05491.
- **Polar Express:** Replaces the Newton-Schulz iteration (our 5-step `zeropower_via_newtonschulz5`) with a tuned polynomial approximation. Faster convergence to the same orthogonalization quality.

**Speedrun impact:**
- NorMuon: Record 45 (2.358 → 2.345 min)
- Polar Express: Record 32 (2.547 → 2.476 min)

**Why it matters for us:** Our Muon optimizer (lines 101-173) uses the original Newton-Schulz iteration with 5 backend steps. NorMuon's variance estimation could give better per-parameter learning rates, improving convergence quality within our fixed time budget. Polar Express is primarily a speed optimization (faster orthogonalization), but if it allows us to use more backend steps in the same time, it could also improve quality.

**Artifact concern:** Zero — pure optimizer changes.

**Implementation complexity:** Medium. NorMuon requires additional state (variance estimates) per parameter. Polar Express requires replacing `zeropower_via_newtonschulz5()` with the polynomial approximation and associated Triton kernels (XXT, XTX, ba_plus_cAA). Could use PyTorch-native implementations instead of Triton.

### 1.9 Asymmetric Logit Rescale

**What:** Apply different scaling factors to logits in positive vs negative directions, beyond the symmetric tanh softcap we currently use (softcap=30.0).

**Speedrun impact:** Record 41 (2.075 → 1.940 min).

**Why it matters for us:** Our logit softcap applies symmetric compression: `30 * tanh(logits / 30)`. Asymmetric rescaling recognizes that the cost of over-confident correct predictions (pushing logits too high) differs from the cost of over-confident wrong predictions (pushing logits too negative). Asymmetric rescaling can improve loss computation efficiency, leading to better gradient signals.

**Artifact concern:** Zero — just a small constant or two in the forward pass.

**Implementation complexity:** Low. Modify the softcap computation to use different scales for positive vs negative logits. One or two extra scalar hyperparameters.

### 1.10 Partitioned Hyperconnections

**What:** Replace U-Net skip connections with multiple parallel residual streams. Instead of a single residual stream with skips from encoder to decoder, maintain 2-3 parallel streams through all layers, with learned mixing between streams at each block.

**Speedrun impact:** Record 77 (1.485 → 1.435 min) — the final record. Evolution path: U-Net skips → value embedding skips → partitioned hyperconnections.

**Why it matters for us:** Our U-Net skip architecture (lines 677-679, 714-720) connects encoder layers to decoder layers with learned weights. Partitioned hyperconnections generalize this to a richer information flow pattern. The final speedrun record uses this, suggesting it's the best-known skip pattern for this model scale.

**Artifact concern:** Low-Medium. Replaces `skip_weights` (shape: `num_skip_weights × dim`) with mixing matrices between streams. If 2 streams with per-block mixing vectors, similar artifact cost. If 3+ streams with full mixing matrices, could add significant params.

**Implementation complexity:** Medium-High. Requires restructuring the forward pass to maintain parallel streams. The mixing matrices replace both the U-Net skips and the existing `resid_mix` parameters. Architecturally invasive.

---

## Tier 2: MEDIUM — Worth Investigating, Uncertain Fit

### 2.1 Smear Module

**What:** 1-token lookback embedding propagation — each position gets access to the previous token's embedding via a small transformation matrix. Like a minimal convolution or shift operation.

**Speedrun impact:** Record 22 (part of value embedding refinement bundle).

**Assessment:** Adds a small matrix per block (~dim² per block, substantial in aggregate). Quality gain unclear for our model size. Lower priority than value embeddings (1.5) which achieve similar goals with less parameter cost.

### 2.2 Dropped First MLP/Attention Layer

**What:** Remove the first MLP layer or attention layer from the model. The first layer often contributes less than later layers; removing it saves compute that can be reinvested in more training steps.

**Speedrun impact:** Record 24 (dropped MLP, 2.863 → 2.717 min), Record 25 (dropped attention, 2.717 → 2.527 min).

**Assessment:** We only have 9 layers. Removing one is ~11% of the model. In the speedrun, they had 12 layers and could afford to lose one. For us, the quality hit may be too large unless we reinvest the savings (wider model, more training steps). Risky but could be tested at P0/P1 cheaply.

### 2.3 Sparse Attention Gate

**What:** Per-head gating mechanism that dynamically controls attention head activation. Some heads can be effectively turned off for certain inputs.

**Speedrun impact:** Record 27 (2.966 → 2.812 min). Implemented with `attn_gate_w` parameter.

**Assessment:** Dynamic sparsity is powerful but adds control parameters and forward-pass complexity. With only 8 attention heads, the gating has limited expressiveness. Better suited for larger models.

### 2.4 Paired Head Attention

**What:** Process attention heads in pairs, sharing some computation between them. Reduces per-head overhead.

**Speedrun impact:** Record 39 (1.940 → 1.820 min).

**Assessment:** Primarily a speed optimization. With 8 heads and GQA (4 KV heads), pairing is already partially achieved. Marginal expected quality benefit.

### 2.5 Mimetic V/O Initialization

**What:** Use non-zero initialization for value and output projection weights, tuned to mimic specific attention patterns at initialization time.

**Speedrun impact:** Record 37 (part of batch size schedule bundle, 2.358 → 2.203 min).

**Assessment:** We use zero initialization for projections (lines 586, 620, 697). Mimetic init could help early training convergence. Low risk, low expected gain. Worth trying if other changes don't pan out.

### 2.6 Partial Key Offset

**What:** Reduce key computation for single-layer induction heads by adding a small offset to keys.

**Speedrun impact:** Record 43 (part of a bundle).

**Assessment:** Q Labs also mentioned this. Marginal expected gain. Low priority.

### 2.7 Backout / Model Steering

**What:** Architecture modification allowing layers to suppress or override predictions from earlier layers. Enables the model to "back out" of bad early predictions.

**Speedrun impact:** Record 40 (2.358 min entry).

**Assessment:** Interesting concept but more beneficial for deep models where early layers can mislead. With only 9 effective layers, the benefit is limited.

### 2.8 Exponential Decay of Residual Stream

**What:** Apply scheduled exponential decay to the residual stream, reducing the contribution of earlier layers relative to later ones.

**Assessment:** We already have `resid_mix` (learned per-block mixing of current and initial representations). This is a manually scheduled variant. Our learned approach is likely better.

### 2.9 YaRN RoPE Scaling

**What:** Dynamic RoPE frequency adjustment for extended context lengths. Scales position embeddings to handle sequences longer than training length.

**Speedrun impact:** Record 30 (2.717 → 2.656 min). Used with variable sequence length scheduling.

**Assessment:** Primarily for long-context extension. At our fixed 1024 seq len, standard RoPE is sufficient. Only relevant if we implement sequence length scheduling (1.4) with very short starting lengths, where YaRN could help with the transition.

---

## Tier 3: ALREADY IMPLEMENTED in train_gpt.py

| # | Technique | Location | Notes |
|---|-----------|----------|-------|
| 1 | Muon optimizer (Newton-Schulz orthogonalization, Nesterov momentum) | Lines 101-173 | 5 backend steps, momentum warmup 0.85→0.95 |
| 2 | RoPE (Rotary Position Embeddings) | Lines 529-559 | Base frequency 10000.0, cached |
| 3 | GQA (Grouped Query Attention) | Lines 562-610 | 8 heads, 4 KV heads |
| 4 | QK-Norm | Lines 595-596 | RMSNorm on Q and K before attention |
| 5 | Per-head Q gain | Lines 587, 600 | Learnable, initialized to 1.5 |
| 6 | ReLU² activation | Lines 622-624 | `relu(fc(x)).square()` |
| 7 | RMSNorm (no learned weight) | Lines 505-511 | Stateless, uses F.rms_norm |
| 8 | U-Net skip connections | Lines 677-679, 714-720 | Learned per-skip weights, encoder→decoder |
| 9 | Learned residual mix | Lines 644, 647-648 | Per-block (2, dim) mixing of current + initial |
| 10 | Per-block attention/MLP scales | Lines 642-643, 650-651 | Learnable, initialized to 1.0 |
| 11 | Logit softcap | Line 730 | Symmetric tanh, cap=30.0 |
| 12 | Tied embeddings | Lines 724-729 | Configurable via TIE_EMBEDDINGS |
| 13 | Zero init for output projections | Lines 586, 620, 697, 704-705 | `_zero_init = True` flag |
| 14 | CastedLinear | Lines 514-518 | fp32 weights, bf16 compute |
| 15 | torch.compile | Lines 835, 942 | fullgraph=True, dynamic=False |
| 16 | Flash Attention (SDPA) | Lines 865-868 | Explicit Flash backend selection |
| 17 | bfloat16 autocast | Lines 1051, 1121 | Training forward pass |
| 18 | TF32 matmul | Lines 861-862 | CUDA TF32 enabled |
| 19 | Gradient accumulation | Lines 849-850, 1117-1125 | 8 // world_size micro-steps |
| 20 | Selective backward sync | Lines 1048-1049, 1118-1119 | Sync only on final micro-step |
| 21 | Cosine warmdown | Lines 1028-1037 | Wallclock-based or iteration-based |
| 22 | Separate optimizer groups | Lines 945-992 | Muon (matrices), Adam (embed, scalars, head) |
| 23 | Fused Adam | Lines 968, 982, 990 | fused=True for speed |
| 24 | Compiled warmup | Lines 1041-1065 | 20 steps, then restore initial state |
| 25 | Muon momentum warmup | Lines 1127-1130 | 0.85→0.95 over 500 steps |
| 26 | fp32 control parameters | Lines 521-526 | restore_low_dim_params_to_fp32() |
| 27 | DDP broadcast_buffers=False | Line 943 | Reduces sync overhead |
| 28 | Embedding init std control | Lines 701-702 | tied_embed_init_std=0.005 |

---

## Tier 4: NOT RELEVANT (Speed-Only or Wrong Scale)

| Technique | Why not relevant |
|-----------|-----------------|
| FP8 Matmul Head (`CastedLinearT`, `mm_t_op`) | Speed optimization for larger models. Our model fits comfortably in bf16 compute. |
| Fused Linear-ReLU² Triton kernel | Speed only — `torch.compile` already fuses these ops for us. |
| Fused Softcapped Cross-Entropy Triton kernel | Speed only. Our CE computation is not a bottleneck. |
| Custom transpose kernels | Pure speed optimization. |
| Reduce-scatter replacing all-reduce | Communication optimization. Our 8xH100 communication is not a bottleneck at 43ms/step. |
| Overlap compute/communication | Speed only. Already have headroom in 10-min budget. |
| Backward hooks on Adam | Speed optimization for gradient synchronization. |
| Adam all-reduce | Communication pattern change. Speed only. |
| FlexAttention 64K context | We use 1024 seq len. 64K context is irrelevant. |
| Long-short sliding window attention | For long contexts. Not applicable at 1024 seq len. |
| bf16 attention/MLP weight storage | We already use CastedLinear (fp32 weights, bf16 compute). |
| Unified optimizers Triton kernel | Single kernel for Adam+Muon. Speed only. |
| Flattened GPT forward pass | Removes post-attention lambdas. Speed only. |
| Sparse bigram gradient communication | Communication optimization for bigram embeddings at scale. |
| Asynchronous data loading coroutines | Speed only. Our data loading is not a bottleneck. |
| `Shard` class sparse reduce-scatter | Distributed communication optimization. |

---

## Already Planned (from Q Labs Blog Analysis)

These techniques were already identified and added to the experiment plan from the Q Labs "10x Data Efficiency" blog (see `qlabs_10x_data_efficiency.md`):

| Experiment | Technique | Phase |
|------------|-----------|-------|
| E23 | EMA weight averaging | Phase 1 |
| E24 | Weight decay / L2 penalty | Phase 3 |
| E25 | SwiGLU activation | Phase 4 |
| E26 | Layer sharing + SwiGLU combo | Phase 4 |
| E18 (enhanced) | Layer sharing / looping | Phase 4 |

---

## Speedrun Records Progression (Key Milestones)

| Record | Time | Key Technique | Contributor |
|--------|------|---------------|-------------|
| 1 | 45.0 min | llm.c baseline | — |
| 3 | 24.9 min | Muon optimizer | @kellerjordan0, @jxbz |
| 5 | 15.2 min | QK-Norm, ReLU², zero-init, embedding skips | @Grad62304977 |
| 8 | 8.2 min | Logit softcap, value embedding skips | @Grad62304977 |
| 10 | 7.2 min | U-Net skip connections | @brendanh0gan |
| 12 | 5.03 min | FlexAttention 64K context | @KoszarskyB |
| 13 | 4.41 min | Extra value embeddings | @KoszarskyB |
| 20 | 2.99 min | Sub-3-minute barrier (communication overlaps) | @ryanyang0 |
| 24 | 2.72 min | Dropped first MLP layer | @EmelyanenkoK |
| 28 | 2.86 min | Document-aligned batching | @classiclarryd |
| 32 | 2.48 min | Polar Express (Muon upgrade) | @varunneal |
| 34 | 1.99 min | Multi-token prediction | @varunneal |
| 38 | 1.66 min | Bigram hash embeddings | @classiclarryd |
| 40 | 2.36 min | Backout / model steering | @classiclarryd |
| 44 | 2.28 min | Cautious weight decay | @varunneal |
| 45 | 2.35 min | NorMuon | @li_zichong |
| 60 | 1.77 min | Fused softcap cross-entropy kernel | Locus AI |
| 77 | 1.44 min | Partitioned hyperconnections (simplified) | @sisovicm |

---

## Summary Statistics

- **Total techniques identified:** 50+
- **Already in our baseline:** 28
- **Tier 1 (high relevance, not implemented):** 10
- **Tier 2 (uncertain fit):** 9
- **Tier 4 (not relevant):** 16
- **Already planned from Q Labs:** 5 (E18, E23-E26)
