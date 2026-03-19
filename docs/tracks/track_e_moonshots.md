# Track E: Architecture Experiments (Layer Sharing, Activation, Combos)

## Thesis
Share parameters across layer groups and re-spend saved bytes into width, better activations, or depth. Originally speculative, now backed by external evidence: [Q Labs achieved 10x data efficiency](../references/qlabs_10x_data_efficiency.md) using layer looping, SwiGLU, and EMA on the NanoGPT Slowrun. Their looping alone contributed ~1.2x improvement (8.88x → 10.05x with EMA).

## Experiments

### E18: Layer Sharing (concrete design)
Share parameters across layer groups, looping a subset of layers to create more effective depth with fewer stored parameters.

**Current baseline:** 9 unique layers, ~15.86MB compressed, 138KB artifact slack.

**Variants to test:**
- **A:** 6 unique layers, loop middle 3 ×2 → 9 effective layers. Saves ~1.7MB → reinvest in wider dim (512→576 or 640).
- **B:** 5 unique layers, loop ×2 → 10 effective layers. More savings, deeper effective model.
- **C:** 4 unique layers, loop ×3 → 12 effective layers. Aggressive. Most savings, biggest risk.

**Implementation notes:**
- Q Labs looped layers 15-24 ×4 in their 30-layer model (33% unique, 400% effective depth).
- They found "it is important not to loop the last few layers" — keep first and last layers unique.
- U-Net skip connections interact with looping: skip weights may need adjustment since encoder/decoder split changes.
- Muon optimizer receives gradients through shared weights multiple times per forward pass. May need gradient scaling or adjusted learning rate.

### E25: SwiGLU Activation
Replace `torch.relu(self.fc(x)).square()` with SwiGLU: `silu(gate(x)) * up(x)`.

**Parameter budget trade-off:**
- Current MLP: 2 matrices (fc + proj) = 1,048,576 params/layer at 2x expansion
- SwiGLU MLP: 3 matrices (gate + up + down). At hidden=704: 1,081,344 params/layer (+32,768 each)
- 9 layers × 32,768 extra params = ~288KB added, **exceeding** the baseline's 138KB artifact slack
- **Standalone test:** use hidden=640 (983,040 params/layer, saves ~65K params/layer vs baseline) to fit 16MB
- **With layer sharing (E26):** can use hidden=704 or larger since sharing frees MB-scale budget
- Alternative: accept param increase, reduce layers from 9 to ~7 to fit 16MB

**Kill:** Worse than baseline by ≥0.004 Δpq at P1.

### E26: Layer Sharing + SwiGLU Combo
Best layer sharing variant + SwiGLU. The freed artifact budget from sharing absorbs SwiGLU's extra gate matrix.

**Example config:** 6 unique layers with SwiGLU (hidden=704), looped to 9 effective. Saves layer params, spends on better activation quality.

Depends on E18 and E25 results to pick best variants.

### E29: Value Embeddings with Gating
Mix input (token) embeddings directly into attention value computations with a learned gating weight per block. Multiple speedrun records (13, 16, 22) showed progressive improvements from this technique.

**Implementation:** Add a `ve_gate_w` scalar parameter per block (initialized to 0). In attention forward, compute `V = V_proj + ve_gate_w * embed_proj(tok_emb)` where `embed_proj` is a small learned projection (or the identity if dims match). The gating starts at zero (no change) and the model learns how much embedding signal to mix in.

**Artifact concern:** Minimal. Gating weights are tiny control tensors kept as fp32. If using a projection, it adds `dim × kv_dim` per layer but can be omitted for a simpler variant.

**Kill:** Worse than baseline by ≥0.003 Δpq at P1.

### E31: Multi-Token Prediction (Training-Only)
Predict 2+ future tokens using additional lightweight linear heads during training. Discard MTP heads at export time — they provide extra gradient signal but are not needed for inference.

**Implementation:** Add 1-2 extra `nn.Linear(dim, vocab_size)` heads. During training, compute cross-entropy loss for positions t+1, t+2, etc., weighted by scheduled `mtp_weights`. At export, remove the extra heads so artifact size is unchanged.

**Artifact concern:** Zero if MTP heads are discarded at export. They exist only for training signal.

**Speedrun impact:** Record 34 (2.203 → 1.988 min). Uses scheduled MTP weights that fade in/out.

**Kill:** If adding MTP heads slows step time by >15% without compensating quality gain, or if Δpq ≥ +0.002.

## Key Metrics
- Δpq, qgap, artifact slack (especially freed bytes from sharing)
- Step time (looping adds compute; SwiGLU is roughly same cost as ReLU²)
- Effective depth vs unique parameters ratio

## Decision Rules
- Promote if it beats baseline outright OR frees ≥300kB while staying within 0.003 Δpq
- E18: test variant A first (least aggressive), promote to P2 if promising
- E25: standalone test before combining with E18
- E26: only if both E18 and E25 show promise independently
- Kill fast if neither quality nor artifact-savings condition is met

## External Evidence
- [Q Labs 10x Data Efficiency](../references/qlabs_10x_data_efficiency.md) — layer looping, SwiGLU, and other architectural innovations

## Status
Not started. E18/E25 depend only on E02 (baseline reproduction, complete). Can begin any time.

## Learnings
(Updated as experiments complete)
