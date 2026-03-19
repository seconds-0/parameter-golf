# Small Model Optimization Landscape (Beyond NanoGPT)

- **Scope:** Chinese small model research, QAT techniques, training recipe innovations, architecture papers
- **Accessed:** 2026-03-19
- **Complements:** `nanogpt_speedrun_techniques.md` (NanoGPT-specific) and `qlabs_10x_data_efficiency.md` (Q Labs)
- **Experiments inspired:** E32 (WSD schedule), E33 (R² regularization), E34 (Turbo-Muon), E35 (higher β₂ cooldown)

---

## HIGH Applicability — New Experiments Created

### 1. WSD Learning Rate Schedule (MiniCPM / Tsinghua)

**Source:** MiniCPM technical report (arXiv:2404.06395), ICLR 2025 validation

**What:** Warmup-Stable-Decay (WSD) divides training into three phases:
- **Warmup** (~1% of steps): Linear ramp from 0 to peak LR
- **Stable** (~75% of steps): Constant peak LR (the "exploration" phase)
- **Decay** (~24% of steps): Cosine or linear decay to near-zero

This replaces our current cosine warmdown (train_gpt.py lines 1028-1037) which uses a continuous cosine curve tied to wallclock or iteration count.

**Why it's better:**
- ICLR 2025 showed WSD consistently outperforms cosine across model sizes
- The long stable phase gives the optimizer maximum time to explore the loss landscape at full learning rate
- No predetermined total length required — can checkpoint mid-stable and resume training later
- Adopted by MiniCPM, OLMo 2, Phi-4, LongCat-Flash

**For Parameter Golf:** With our 10-minute / ~13,780-step budget:
- Warmup: ~140 steps (1%)
- Stable: ~10,335 steps (75%)
- Decay: ~3,305 steps (24%)
- Compare to current: 20 compiled-warmup steps, then cosine warmdown over remaining iterations

**Artifact cost:** Zero — pure schedule change.

**Experiment:** E32

### 2. Range Regularization (R²) for Quantization

**Source:** OpenReview: "R²: Range Regularization for Quantization"

**What:** Adds a regularization term during training that penalizes the range (max - min) of weight values within each tensor or row. This tightens the weight distribution, making int8 quantization more efficient.

**How it's different from existing experiments:**
- **E24 (weight decay):** Penalizes weight magnitude (L2 norm). Pushes all weights toward zero.
- **E13 (clamp-aware regularizer):** Penalizes per-row outliers that get clipped at the int8 percentile.
- **R² (this):** Penalizes the range of the distribution itself. A tensor with weights in [-0.5, 0.5] quantizes better than [-0.1, 0.1] with one outlier at 2.0, even if the L2 norm is smaller for the second.

**Implementation:** Add `λ_R² * (max(W) - min(W))` per weight matrix to the loss, or equivalently apply per-row range penalties matching our per-row int8 scheme.

**Artifact cost:** Zero — training regularization only.

**Related techniques (bundle candidates):**
- **Kurtosis Regularization (KURE)** (arXiv:2602.03614): Penalizes high kurtosis (spiky distributions). Complements R² by targeting distribution shape rather than range.
- **Bin Regularization** (ICCV 2021): Sharpens quantization bin boundaries. More complex implementation.

**Experiment:** E33 (R² standalone, potentially bundle with KURE)

### 3. Turbo-Muon (AOL Spectral Preconditioning)

**Source:** arXiv:2502.16982 "Muon is Scalable for LLM Training"

**What:** Replaces or augments Newton-Schulz orthogonalization with AOL (Approximate Orthogonal Learning) spectral preconditioning. Claims 2.8x speedup on the orthogonalization step with ~20% overhead reduction.

**How it differs from other Muon variants in our docs:**
- **NorMuon** (speedrun doc): Adds Adafactor-style low-rank variance estimation for per-matrix LR
- **Polar Express** (speedrun doc): Replaces Newton-Schulz with tuned polynomial approximation
- **Turbo-Muon** (this): Uses spectral preconditioning to reduce the number of orthogonalization iterations needed

**For Parameter Golf:** Our Muon uses 5 Newton-Schulz iterations (MUON_BACKEND_STEPS=5). If Turbo-Muon achieves equivalent quality in 3-4 steps, we get faster per-step time → more training steps in 10 minutes → potentially better final quality.

**Artifact cost:** Zero — optimizer change only.

**Experiment:** E34

### 4. Higher β₂ During Cooldown

**Source:** arXiv:2508.01483 "WSD Cooldown Dynamics"

**What:** During the learning rate decay phase, increase Adam's β₂ from 0.95 to 0.97-0.99. Higher β₂ gives longer memory in the second moment estimator, which smooths out gradient noise during cooldown and improves the final model quality.

**Implementation:** ~3 lines of code. In the LR scaling loop (train_gpt.py ~lines 1132-1134), also adjust β₂ when the warmdown factor drops below a threshold.

**Artifact cost:** Zero.

**Experiment:** E35 (can combine with E32 WSD schedule or standalone with existing cosine warmdown)

---

## MEDIUM Applicability — Reference Only

### CompleteP Parameterization
**Source:** arXiv:2505.01618

12-34% compute efficiency improvement by enabling "non-lazy learning in all layers" through proper depth-wise hyperparameter transfer. The key insight: standard parameterizations waste compute because deeper layers learn lazily. CompleteP ensures all layers contribute equally.

**For us:** Informational. Could guide architecture decisions if we explore depth/width trade-offs (E16-E17). Not a standalone experiment.

### NoPE Every 4th Layer (SmolLM3)
**Source:** HuggingFace SmolLM3 technical report

SmolLM3 removes RoPE from every 4th transformer layer. These "NoPE" layers attend without positional bias, acting as pure content-matching layers. Shown not to hurt short-context performance while improving long-context.

**For us:** At 1024 seq len, the long-context benefit is moot. But the compute savings of skipping RoPE application on 2-3 of our 9 layers could free cycles for more training steps. Low priority.

### Three-Stage Pretraining (SmolLM3, OLMo 2)
**Source:** HuggingFace, AI2

Stage 1: Massive broad web data → Stage 2: High-quality filtered data → Stage 3: Task-specific tuning. Proven at 1-3B scale.

**For us:** FineWeb is a single dataset with no obvious quality tiers to stage over. Could potentially be combined with batch size schedule (E30) for a curriculum effect, but not a standalone experiment.

### Sequence Length Warmup
**Source:** arXiv:2108.06084, validated by MiniCPM

Start at 256 seq len, ramp to 1024 over first 5% of steps. Faster early steps, more data seen.

**For us:** Already noted in NanoGPT speedrun doc (Tier 1 #4, not promoted). MiniCPM research adds validation. If E30 (batch size schedule) works, consider bundling seq len warmup into it.

---

## NOT Applicable — Filtered Out

| Technique | Source | Why not |
|-----------|--------|---------|
| **Mixture of Experts (MoE)** | OLMoE, DeepSeek | Too complex for ~15M params. Routing overhead + expert params eat the 16MB budget. |
| **Logit Distillation** | Qwen 3.5 | Requires a teacher model. We're training from scratch in 10 min. |
| **RL-Guided Data Selection** | arXiv:2509.25850 | Requires offline pre-computation on FineWeb. May violate competition spirit rules. |
| **Gated Linear Attention (GLA)** | arXiv:2312.06635 | Unproven on BPB compression metric. Risky architectural swap. |
| **Multi-Head Latent Attention (MLA)** | DeepSeek | Designed for much larger scale (67B+). Overhead not justified at 15M. |
| **LoRA** | Original paper | For fine-tuning, not pre-training from scratch. |
| **GRPO (DeepSeek RL)** | DeepSeek-R1 | Requires RL infrastructure and reward model. Doesn't fit 10-min budget. |
| **Token Pruning** | LazyLLM, COMPACT, SDTP | Inference-time optimization, not training. |
| **Gradient Checkpointing** | Standard technique | Memory not constrained on 8xH100 (we have headroom). |
| **FP8 Mixed Precision Training** | DeepSeek | Our model is too small for FP8 to help; bf16 is sufficient. |
| **SOAP Optimizer** | arXiv:2409.11321 | Shampoo variant. Memory/compute overhead impractical for <50M params. Adam+Muon hybrid is likely superior. |

---

## Key Chinese Research Landscape

### MiniCPM (Tsinghua / ModelBest)
- 1.2B and 2.4B params achieving 7B-13B capabilities
- **WSD scheduler** (adopted as E32) is their headline contribution
- **Model Wind Tunnel Experiments (MWTE):** Hyperparameter optimization across model scales via Tensor Program framework. Informational for our architecture decisions.
- arXiv:2404.06395

### Qwen 3.5 (Alibaba)
- 0.8B to 9B variants with hybrid Gated Delta Networks + sparse MoE
- **Logit distillation at 1/10 GPU hours** — powerful but requires teacher model
- Qwen3.5-9B reached 81.7 on GPQA Diamond (surpassing models 10x larger)
- Not directly applicable due to distillation dependency

### DeepSeek
- **GRPO** (Group Relative Policy Optimization) — RL without value function, more efficient than PPO
- **FP8 training** at scale — pioneering but not applicable to our model size
- Key insight: "two-stage: pre-training (self-supervised) → optimization training (RL-based)"
- Distilled versions as small as 1.5B but the technique requires RL infrastructure

### SmolLM3 (HuggingFace)
- GQA with 4 groups (matches our setup)
- **NoPE every 4th layer** — interesting but low priority for us
- **Three-stage pretraining** — limited applicability to single-dataset FineWeb
- Key data insight: "smart mixing >> volume"

### CT-LLM (Chinese-centric)
- 2B model, 1200B tokens (800B Chinese, 300B English, 100B code)
- Language-specific training curriculum — not applicable to English-only FineWeb

---

## Cross-Reference with Existing Experiments

| New finding | Closest existing experiment | Relationship |
|-------------|---------------------------|-------------|
| WSD schedule (E32) | Cosine warmdown (baseline) | **Replacement** — WSD replaces cosine |
| R² regularization (E33) | E13 (clamp-aware), E24 (weight decay) | **Complement** — different target (distribution shape vs magnitude vs row outliers) |
| Turbo-Muon (E34) | NorMuon/Polar Express (ref only) | **Alternative** — different Muon improvement axis |
| Higher β₂ cooldown (E35) | E32 (WSD) | **Combinable** — works with WSD or cosine |
| KURE | E33 (R²) | **Bundle candidate** — both target distribution shape |
| CompleteP | E16-E17 (arch rebudget) | **Informational** — guides width/depth decisions |
| NoPE layers | Baseline RoPE | **Low priority** — minor compute savings |
| Data curriculum | E30 (batch schedule) | **Combinable** — similar spirit |

---

## Sources

- MiniCPM: arXiv:2404.06395
- CompleteP: arXiv:2505.01618
- Turbo-Muon: arXiv:2502.16982
- NorMuon: arXiv:2510.05491
- KURE: arXiv:2602.03614
- R² Range Regularization: OpenReview
- WSD Cooldown: arXiv:2508.01483
- Sequence Length Warmup: arXiv:2108.06084
- SmolLM3: huggingface.co/blog/smollm3
- OLMoE: Mixture of Experts research
- DeepSeek-R1: cdn.deepseek.com
- Qwen 3.5: qwenlm.github.io/blog/qwen3/
- EfficientQAT: arXiv:2407.11062
- LieQ: arXiv:2508.03332
- SOAP: arXiv:2409.11321
- GLA: arXiv:2312.06635
- MobileLLM: arXiv:2402.14905
