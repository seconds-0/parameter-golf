# Q Labs: 10x Data Efficiency with Infinite Compute

- **Source:** [https://qlabs.sh/10x](https://qlabs.sh/10x)
- **Accessed:** 2026-03-19
- **Competition:** NanoGPT Slowrun (fixed 100M tokens, optimize data efficiency)
- **Experiments inspired:** E23 (EMA), E24 (weight decay), E25 (SwiGLU), E26 (combo), E18 enhanced (layer sharing)

## Summary

Trained an ensemble of 1.8B parameter models (18B total across 8 models) on 100M tokens, achieving 10x data efficiency — matching performance that typically requires 1B tokens. Key insight: "data efficiency matters because compute grows much faster than data."

Their competition differs from Parameter Golf: they have unlimited model size and compute, we have 16MB artifact cap + 10min on 8xH100. But several individual techniques transfer.

## Techniques

### 1. Ensemble + Chain Distillation

**NOT APPLICABLE to Parameter Golf** — we submit a single model in 16MB.

Train K models sequentially, each distilling from predecessor:
- Train M_1 with cross-entropy loss
- For k=2 to K: Loss = (1-alpha) * CE(M_k(x), y) + alpha * T^2 * KL(M_k(x)/T || M_{k-1}(x)/T)
- alpha=0.5, T=1.0
- 8 chain-distilled models: ensemble loss 3.0530 (single model ~3.20)
- 10 model ensemble: 3.0453

### 2. Heavy Regularization

**APPLICABLE** — directly targets our export retention / qgap problem.

- Weight decay: **1.6** (16x standard 0.1)
- Dropout: 0.1
- Rationale: models are heavily overparameterized (1.8B params on 100M tokens vs Chinchilla's ~5M recommendation)
- Heavy L2 acts as "proxy for simplicity/compression"

**Parameter Golf relevance:** Smaller weight magnitudes from high weight decay → smaller per-row int8 scales → less quantization noise → smaller qgap. Acts as implicit quantization-aware training.

**Experiment:** E24 — sweep weight_decay in {0.1, 0.5, 1.0, 1.6}

### 3. Looped Transformers (Layer Sharing)

**HIGHLY APPLICABLE** — directly attacks our 16MB artifact constraint.

- Looped layers 15-24 four times during inference (in their 30-layer model)
- Same parameters, more effective depth
- "Important not to loop the last few layers"
- Contributed 8.88x → 10.05x data efficiency improvement (with EMA)

**Parameter Golf relevance:** Fewer unique parameters stored (smaller artifact) but deeper effective computation. Our 9-layer model uses ~15.86MB with only 138KB slack. Sharing layers could free MB-scale budget to reinvest in width.

**Experiment:** E18 enhanced (variants A/B/C: 6/5/4 unique layers looped to 9-12 effective)

### 4. SwiGLU Activation

**APPLICABLE** — well-proven replacement for ReLU^2.

- Replaced squared ReLU with SwiGLU
- Part of a bundle that gave 5.25x → 5.79x improvement
- Used in Llama, Mistral, and most modern transformers

**Parameter Golf relevance:** SwiGLU needs 3 weight matrices vs 2 for standard MLP. To match param budget: hidden_dim drops from 1024 to ~704. Trade-off: better activation vs narrower MLP. Best combined with layer sharing to absorb the overhead.

**Experiment:** E25 (standalone), E26 (combined with layer sharing)

### 5. EMA (Exponential Moving Average)

**APPLICABLE** — nearly free, low risk.

- Used EMA of model weights
- Combined with increased looping for final 10.05x result

**Parameter Golf relevance:** Smooths late-training weight oscillations → fewer outlier weights → smaller qgap after int8 quantization. ~20 lines of code, essentially zero compute cost.

**Experiment:** E23 — sweep decay in {0.999, 0.9999}

### 6. Other Techniques (lower priority)

**XSA (Exclusive Self Attention):** Removes value projection. Saves ~1.18MB at int8 across 9 layers but risky architectural departure. Not planned as standalone experiment.

**Half-truncated RoPE:** RoPE on only half the head dimensions. Marginal expected gain. Not prioritized.

**Value embeddings:** Learned projection from input embeddings for values. Interesting but adds complexity. Not prioritized.

**Partial key offset for induction heads:** Small tweak. Not prioritized.

## Results Timeline

| Date | Technique | Data Efficiency |
|------|-----------|-----------------|
| 02/26 | Muon, heavy regularization, multi-epoch | 3.81x |
| 02/27 | Ensemble of 8 models | 5.25x |
| 03/02 | Value projections + SwiGLU | 5.79x |
| 03/04 | U-Net + Attention Gating | 6.39x |
| 03/05 | Longer ensemble training | 6.88x |
| 03/07 | Chain distillation | 7.92x |
| 03/13 | Add looping | 8.88x |
| 03/19 | Increase looping, use weight EMA | 10.05x |

## Key Insights

- "Post-hoc transforms like ensembling reverse usual overfitting dynamics"
- Models trained past individual optimum learn different things → ensemble benefits
- Neural architecture search was critical for data efficiency
- Massive overparameterization (3600x Chinchilla) works with aggressive regularization
- Weight decay=1.6 is critical — far higher than standard practice

## Contributors

@ChinmayK0607, @not-nonymous, @shmublu, @zhiweixx, @em-see-squared, @ms337, @kvegesna, @akshayvegesna
