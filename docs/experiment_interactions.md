# Experiment Interactions, Conflicts, and Composition

> How the experiments play together. Read before composing winners in Phase 5. Detailed closeout reviews for completed branches live in [docs/postmortems/](./postmortems/README.md).

## Experiment Grouping by Pipeline Stage

Each experiment touches a specific part of the training pipeline. Experiments within the same group interact; experiments across groups are usually independent.

### Group A: LR Schedule (PICK ONE base)
| Exp | What | Conflict |
|-----|------|----------|
| Baseline | Current baseline warmdown | — |
| **E32** | WSD (Warmup-Stable-Decay) | **Replaces** the current baseline warmdown. Pick one. |
| **E35** | Higher β₂ during cooldown | **Layers on top** of either E32 or cosine. Not exclusive, but now tested and killed on top of WSD. |

**Verdict:** E32 vs the current baseline schedule is a binary choice, and the live repo result now favors WSD. `E35` was the obvious modifier to test next, but it regressed cleanly on top of WSD, so the active schedule base remains plain WSD.

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
| **E23** | Export-time smoothing | EMA of weights, applied at export only; now a killed branch on the current short-run proxy as implemented |

**Conflict:** E24 + E33 + E13 all regularize weights during training. Stacking all three risks **over-regularization** — quality regression where the model is too constrained to learn well. Each one narrows the weight distribution from a different angle, but the net effect compounds.

**Verdict:** Test individually at P1. Compose at most the top 1-2 winners. `E23` is still conceptually orthogonal, but the live repo result is now strongly negative: both tested EMA decays made the exported checkpoint dramatically worse while leaving live prequant quality near baseline, so it should not stay in the active composition set.

### Group D: Exporter-Only Knobs (RE-SWEEP after regularizers)
| Exp | What |
|-----|------|
| **E03** | Clip percentile sweep |
| **E04** | Keep-float threshold sweep |

**Interaction:** The optimal clip percentile and threshold depend on the weight distribution. If any Group C regularizer wins and changes the distribution, E03/E04 need re-sweeping on the new distribution.

**Verdict:** E03/E04 are already complete on the trusted baseline checkpoint and came back flat. Re-sweep only after a winner from E24, E33, or E13 changes the weight distribution, or later during final composition.

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
| **E27** | Document-aligned batching | None mechanically, but on the current published BOS-delimited shards it is already a killed branch unless the packing strategy changes. |
| **E28** | Asymmetric logit rescale | Minor: if E31 (MTP) also used, rescale must apply to all heads. Live result now favors the specific `(cap_pos=20, cap_neg=30)` variant. |
| **E30** | Batch size schedule | None. Orthogonal to LR schedule. |

**Verdict:** These are structurally independent, but the live repo state matters: `E27` was the safest thing to test and it already failed on the current published BOS-delimited shards because only about `69%` of target positions remained supervised. `E28` is now complete and promoted, but only for the negative-favored `(20,30)` setting; the positive-heavy asymmetric variants regressed. Treat `E27` as killed, `E28(20,30)` as the active winner, and `E30` as the next open branch.

---

## Key Mutual Exclusivities (Must Choose)

| Choice | Option A | Option B | When to decide |
|--------|----------|----------|----------------|
| LR Schedule | Current baseline warmdown | E32 (WSD) | Decided at E32 P1 |
| SwiGLU path | E25 standalone (hidden=640) | E26 combo (with E18 sharing) | After E18 P1 result |
| Layer sharing variant | E18-A (6 unique) | E18-B (5 unique) or E18-C (4 unique) | After E18-A P1 result |
| Artifact budget axis | E16/E17 (KV rebudget) | E18 (layer sharing) | After both P1 results |

---

## Composition Risk Matrix

**Safe to compose (independent axes):**
- E30 (batch schedule) + E32 (WSD)
- E34 (Turbo-Muon) + any schedule change
- E31 (MTP) + any architecture (zero export cost)
- E28 `(20,30)` (logit rescale) + most things (watch E31 interaction)

**Compose carefully (same axis, diminishing returns):**
- E24 + E33 + E13 — max 2 of 3 regularizers
- E03/E04 re-sweep needed after any regularizer change
- E32 + E35 — now tested on the WSD base and killed in the current repo state
- Do not sweep more E28 softcap points until another result changes the training/logit distribution materially

**Compose with budget check (artifact-coupled):**
- E18 + E25 -> E26 — explicitly designed as combo, but verify artifact fits
- E29 (value embeds) on top of E26 — check remaining slack
- E16/E17 + E18 — aggressive, pick one budget-freeing axis

---

## Recommended Composition Layers (for the final model)

Build the final candidate in layers, testing each before adding the next:

**Layer 1 — Foundation (independent wins, test in parallel at P1):**
- E32 (WSD) or the current baseline schedule — pick winner
- E34 (Turbo-Muon) or standard — pick winner

**Layer 2 — Recipe (layer on top of L1 winner):**
- E35 (beta2 cooldown) — already tested on the WSD base and killed; do not include in the active composition stack
- E28 `(20,30)` (asymmetric logit rescale) — now promoted on top of WSD
- E30 (batch size schedule) — next on top of the WSD + E28 base

**Layer 3 — Export retention (pick best 1-2 from P1 results):**
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

1. Start with the live cheap `E02`-unblocked independent tranche from the current state: `E30` on top of the promoted `E32` WSD + `E28(20,30)` base
2. `E27` is already complete and killed on the current BOS-delimited shard format, so it is no longer part of the active queue unless the packing/data path changes materially
3. `E23` is also complete and killed on the current short-run proxy, so the live tranche now centers on schedule/logit experiments unless we intentionally switch to Track B and run `E24`
4. The active base recipe now uses WSD plus asymmetric `(20,30)` because `E35` regressed and `E28` promoted on top of WSD; after that, the next Track C side branches are `E30` and `E34`
5. If Track B is the next lane to push after that tranche, test regularizers individually in this order of discipline: `E24a`, then `E24b`, then `E33` or `E13`; only re-sweep `E03/E04` after one of those changes the distribution
6. Tokenizer-dependent recipe work (`E10`-`E12`) waits for `X-06`, `E05`, and `E09`; architecture experiments (`E16`, `E18`, `E25`) remain independent side branches, not default next steps
7. Composition happens in Phase 5 (`E19`/`E20`), building on the winners from the independent tranche plus the best surviving Track B / Track A / architecture branches

No experiments are truly "wasted" — even if E25 is superseded by E26, the E25 P1 result tells us whether SwiGLU activation quality is worth pursuing at all.

---

## Sub-Experiment Gating Dependencies

These decomposed ideas have internal sequential gates:

### E14 (QAT-lite): alternatives, cheapest first
```
E14b (noise, 2% overhead) → if promising → E14a (STE, 10%) → if promising → E14c (LSQ-lite)
```
All are alternatives targeting the same goal (quantization robustness during training). Run cheapest first.

### E31 (MTP): sequential, each gates the next
```
E31a (1 head, no curriculum) → if passes → E31b (2 heads, curriculum) → if passes → E31c (decaying weights)
```
If E31a fails (MTP doesn't help at 15M scale), the entire branch is killed immediately.

### E34 (Muon upgrades): mostly alternatives, E34c is independent
```
E34a (Polar Express, drop-in) → if insufficient speedup → E34b (Turbo-Muon AOL)
E34c (NorMuon, adaptive LR) — independent axis, can run in parallel with E34a
```
E34a and E34b are alternatives for faster orthogonalization. E34c is a different improvement axis (per-neuron LR) and is independent.
