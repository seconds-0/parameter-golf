# Experiment Interactions, Conflicts, and Composition

> How the experiments play together. Read before composing winners in Phase 5.

## Experiment Grouping by Pipeline Stage

Each experiment touches a specific part of the training pipeline. Experiments within the same group interact; experiments across groups are usually independent.

### Group A: LR Schedule (PICK ONE base)
| Exp | What | Conflict |
|-----|------|----------|
| Baseline | Cosine warmdown | — |
| **E32** | WSD (Warmup-Stable-Decay) | **Replaces** cosine. Pick one. |
| **E35** | Higher β₂ during cooldown | **Layers on top** of either E32 or cosine. Not exclusive. |

**Verdict:** E32 vs cosine is a binary choice. E35 is a modifier that works with either.

### Group B: Optimizer Internals (INDEPENDENT)
| Exp | What | Conflict |
|-----|------|----------|
| **E34** | Turbo-Muon (faster orthogonalization) | Replaces Newton-Schulz internals. Independent of everything else. |
| **E24** | Weight decay (adds penalty to optimizer step) | Independent of E34. Different concern. |

**Verdict:** E34 and E24 live on different axes — speed vs regularization. Compose freely.

### Group C: Export Retention Regularizers (DIMINISHING RETURNS risk)
| Exp | Target | How |
|-----|--------|-----|
| **E24** | Weight magnitude | L2 penalty pushes weights toward 0 |
| **E33** | Distribution range | Penalizes max-min per row |
| **E13** | Row outliers | Penalizes values exceeding clip percentile |
| **E23** | Export-time smoothing | EMA of weights, applied at export only |

**Conflict:** E24 + E33 + E13 all regularize weights during training. Stacking all three risks **over-regularization** — quality regression where the model is too constrained to learn well. Each one narrows the weight distribution from a different angle, but the net effect compounds.

**Verdict:** Test individually at P1. Compose at most the top 1-2 winners. E23 (EMA) is safe to stack with any of them since it's export-time only (doesn't affect training dynamics).

### Group D: Exporter-Only Knobs (RE-SWEEP after regularizers)
| Exp | What |
|-----|------|
| **E03** | Clip percentile sweep |
| **E04** | Keep-float threshold sweep |

**Interaction:** The optimal clip percentile and threshold depend on the weight distribution. If any Group C regularizer wins and changes the distribution, E03/E04 need re-sweeping on the new distribution.

**Verdict:** Run E03/E04 on the baseline first (they're already queued). Re-sweep at composition time after regularizer winners are chosen.

### Group E: Architecture Changes (BUDGET-COUPLED)
| Exp | Params change | Artifact impact |
|-----|---------------|-----------------|
| **E18** | Fewer unique layers (saves MB) | Frees artifact budget |
| **E25** | SwiGLU hidden=640 (saves 65K/layer) | Fits in current 138KB slack |
| **E26** | E18 + SwiGLU hidden=704 | Uses freed E18 budget for better hidden |
| **E29** | Adds gating params (tiny) | Minimal, fp32 control tensors |
| **E31** | MTP heads (training-only, discarded) | Zero at export |
| **E16/E17** | Fewer KV heads (saves params) | Frees artifact budget |

**Mutual exclusivity:**
- **E25 standalone vs E26 combo:** If E18 (layer sharing) works, E26 supersedes E25. If E18 fails, E25 standalone is the only SwiGLU path. **Don't run both to conclusion** — E25 is a stepping stone.
- **E18 variants A/B/C:** Pick one sharing strategy. Not mix-and-match.
- **E16/E17 vs E18:** Both free artifact budget but through different means (fewer KV heads vs fewer layers). Could theoretically combine, but that's a LOT of architectural change at once. Better to pick the axis that wins at P1.

**E29 and E31 are additive:** Value embeddings (E29) add tiny params. MTP (E31) adds nothing at export. Both can layer on top of any architecture.

### Group F: Data & Forward Pass (INDEPENDENT)
| Exp | What | Conflict |
|-----|------|----------|
| **E27** | Document-aligned batching | None. Pure data quality. |
| **E28** | Asymmetric logit rescale | Minor: if E31 (MTP) also used, rescale must apply to all heads. |
| **E30** | Batch size schedule | None. Orthogonal to LR schedule. |

**Verdict:** All three are independent of each other and nearly everything else. E27 is the safest experiment in the entire suite.

---

## Key Mutual Exclusivities (Must Choose)

| Choice | Option A | Option B | When to decide |
|--------|----------|----------|----------------|
| LR Schedule | Cosine warmdown (baseline) | E32 (WSD) | After E32 P1 result |
| SwiGLU path | E25 standalone (hidden=640) | E26 combo (with E18 sharing) | After E18 P1 result |
| Layer sharing variant | E18-A (6 unique) | E18-B (5 unique) or E18-C (4 unique) | After E18-A P1 result |
| Artifact budget axis | E16/E17 (KV rebudget) | E18 (layer sharing) | After both P1 results |

---

## Composition Risk Matrix

**Safe to compose (independent axes):**
- E27 (doc-aligned) + anything
- E30 (batch schedule) + E32 (WSD) + E35 (β₂)
- E34 (Turbo-Muon) + any schedule change
- E23 (EMA) + any regularizer
- E31 (MTP) + any architecture (zero export cost)
- E28 (logit rescale) + most things (watch E31 interaction)

**Compose carefully (same axis, diminishing returns):**
- E24 + E33 + E13 — max 2 of 3 regularizers
- E03/E04 re-sweep needed after any regularizer change
- E32 + E35 — synergistic but test E32 alone first

**Compose with budget check (artifact-coupled):**
- E18 + E25 -> E26 — explicitly designed as combo, but verify artifact fits
- E29 (value embeds) on top of E26 — check remaining slack
- E16/E17 + E18 — aggressive, pick one budget-freeing axis

---

## Recommended Composition Layers (for the final model)

Build the final candidate in layers, testing each before adding the next:

**Layer 1 — Foundation (independent wins, test in parallel at P1):**
- E27 (doc-aligned batching) — always on if it helps
- E32 (WSD) or cosine — pick winner
- E34 (Turbo-Muon) or standard — pick winner

**Layer 2 — Recipe (layer on top of L1 winner):**
- E35 (beta2 cooldown) — on top of L1 schedule
- E30 (batch size schedule) — on top of L1
- E28 (asymmetric logit rescale)

**Layer 3 — Export retention (pick best 1-2 from P1 results):**
- E23 (EMA) — almost always include
- Best of {E24, E33, E13} — pick the one with best qgap improvement
- Re-sweep E03/E04 on the L1+L2+L3 composed model

**Layer 4 — Architecture (if budget axis was explored):**
- Winner of {E18 -> E26, or E25 standalone}
- E29 (value embeddings) on top
- E31 (MTP training-only heads) on top

**Layer 5 — Final composition (E19/E20):**
- Everything from L1-L4 composed into a single P2 run
- Two seeds for statistical confidence

---

## Execution Order

The suite is NOT a simple list to run sequentially. It's a **decision tree:**

1. Run Phase 1 experiments in parallel where possible (E27, E32, E35, E23, E28, E03/E04)
2. Results from Phase 1 determine which branches to pursue
3. Architecture experiments (E18, E25, E16) are independent explorations — run the most promising first
4. Regularizers (E24, E33, E13) should be tested individually before composing
5. Composition happens in Phase 5 (E19/E20), building on winners from all tracks

No experiments are truly "wasted" — even if E25 is superseded by E26, the E25 P1 result tells us whether SwiGLU activation quality is worth pursuing at all.
