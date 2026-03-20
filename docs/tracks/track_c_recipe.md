# Track C: Recipe, Schedule & Optimizer Tuning

## Thesis
In this small-vocab tied-embedding regime, the embedding matrix is unusually load-bearing. The separate TIED_EMBED_LR knob matters more than generic LR sweeps. Small, targeted optimizer stars (not grids) on the tokenizer winner can unlock cheap gains. Additionally, training schedule changes (batch size, logit processing) are zero-artifact-cost recipe improvements drawn from the [NanoGPT speedrun](../references/nanogpt_speedrun_techniques.md).

## Experiments
- **E10**: Tied-embed LR star: test {0.03, 0.05, 0.08} on tokenizer winner (3-point, ~0.3 H100-hrs)
- **E11**: Matrix/scalar LR star: test MATRIX_LR ∈ {0.025, 0.04, 0.06} on tokenizer winner (3-point, ~0.3 H100-hrs)
- **E12**: Embedding norm penalty A/B
- **E23**: EMA weight averaging at export (decay sweep: 0.999, 0.9999) — inspired by [Q Labs 10x blog](../references/qlabs_10x_data_efficiency.md). ~20 lines of code. Maintain shadow EMA weights during training, swap in at export time. Targets qgap reduction via smoother weights. Kill if qgap doesn't improve by ≥10% or val_bpb regresses by >0.002. **Complete, killed** — both tested decays catastrophically worsened the exported checkpoint on the P1 proxy.
- **E28**: Asymmetric softcap — replace symmetric `30 * tanh(logits/30)` with separate caps for positive vs negative logits: `cap_pos * tanh(logits/cap_pos)` for logits > 0, `cap_neg * tanh(logits/cap_neg)` for logits ≤ 0. Test (cap_pos, cap_neg) ∈ {(30,20), (30,15), (20,30)}. Kill if Δpq ≥ +0.002. Ref: [NanoGPT speedrun 1.9](../references/nanogpt_speedrun_techniques.md#19-asymmetric-logit-rescale). **Complete, promoted** — `(20,30)` won cleanly on the WSD base while the other two asymmetric settings regressed.
- **E30**: Batch size schedule — two-stage: `TRAIN_BATCH_TOKENS=131072` (¼ baseline) for the first 30% of steps, then `524288` (baseline) for the remaining 70%. Implemented by changing `grad_accum_steps` at the iteration threshold. More gradient updates early = faster initial learning. Kill if Δpq ≥ +0.003. Ref: [NanoGPT speedrun 1.3](../references/nanogpt_speedrun_techniques.md#13-batch-size-schedule). **Unblocked** — depends only on E02.
- **E32**: WSD LR schedule — Warmup-Stable-Decay replacing the current baseline warmdown path. MiniCPM (arXiv:2404.06395) + ICLR 2025 showed WSD outperforms cosine-like schedules. Three phases: 1% warmup, 75% stable, 24% cosine decay. Zero artifact cost. Kill if val_bpb regresses >0.003 vs the matched baseline schedule. Ref: [small model landscape](../references/small_model_optimization_landscape.md#1-wsd-learning-rate-schedule-minicpm--tsinghua). **Complete, promoted** — strong same-host P1 win.
- **E34a**: Polar Express — drop-in replacement for `zeropower_via_newtonschulz5()` using minimax-optimal spectral polynomial. Proven in NanoGPT speedrun record #38. ~50-100 lines, pure PyTorch. Code: github.com/NoahAmsel/PolarExpress. Test both reducing iterations (5→4) and keeping 5 for better quality. Kill if Δpq ≥ +0.003 AND step time doesn't improve. **Unblocked** — depends only on E02.
- **E34b**: Turbo-Muon AOL — add diagonal spectral preconditioner before NS iterations. `AOL = fast_inv_sqrt(sum(abs(W@W^T)))`. ~200-300 lines, PyTorch version available at github.com/thib-s/flash-newton-schulz/. Kill if Δpq ≥ +0.003 AND no speedup. Run only if E34a insufficient.
- **E34c**: NorMuon — row-wise normalization after orthogonalization using per-neuron second-order momentum. ~150-200 lines. Code: github.com/zichongli5/NorMuon. Independent axis from E34a/b (adaptive LR, not faster ortho). Kill if Δpq ≥ +0.003. Can run in parallel with E34a.
- **E35**: Higher β₂ during cooldown — increase Adam β₂ from 0.95 to 0.97-0.99 during the LR decay phase. arXiv:2508.01483 showed this improves final val loss. ~3 lines of code. Can combine with E32 (WSD) or existing cosine warmdown. Kill if val_bpb regresses >0.001. **Complete, killed** — the matched same-host P1 follow-up on top of WSD regressed cleanly.

## Key Metrics
- Δpq (post-roundtrip delta vs baseline)
- qgap (export retention)
- Step time (must not regress significantly)

## Decision Rules
- Promote if Δpq ≤ -0.003 at P1
- Kill if center point remains best (no improvement found)
- Do NOT run the 15-config LR grid — it's explicitly the wrong approach per the PRD

## Status
Partially unblocked. Tokenizer-dependent recipe work `E10`-`E12` still depends on `E09` (tokenizer winner selected), while the cheap side experiments `E30` and `E34` remain unblocked by `E02`. `E23` and `E35` are complete and killed, and `E32` plus `E28` are now complete and promoted. The active Track C base is WSD plus asymmetric softcap `(20,30)`. Detailed closeout reviews: [E23](../postmortems/e23_ema_export.md), [E28](../postmortems/e28_asymmetric_logit_rescale.md), [E32](../postmortems/e32_wsd_schedule.md), [E35](../postmortems/e35_cooldown_beta2.md).

The recommended next Track C order is:
- `E30` batch schedule on top of the promoted WSD + asymmetric `(20,30)` base
- `E34a` Polar Express as the next independent side branch after `E30`

`E30` and `E34` remain valid `E02`-unblocked side branches, but `E30` is now the highest-priority cheap Track C follow-up because `E28` has already produced a strong same-host P1 win that should become the default base before we test the next independent recipe lever.

## Learnings
- `E23` produced a clean negative result in the matched same-host P1 bundle: control `phase1_e23_control_p1-20260320-163308-38d0b485` landed at postquant `1.38639890`, while `EMA_DECAY=0.999` landed at `2.03365982` and `EMA_DECAY=0.9999` landed at `5.45891420`
- The failure mode is informative: both EMA runs kept the live prequant path near baseline before the EMA swap (`1.38512695` and `1.38464865`), so the branch failed specifically because the export-time EMA snapshot was bad, not because EMA destabilized training itself
- `EMA_DECAY=0.999` already fails the Track C kill rule badly (`Δpq=+0.64726092`, `qgap=0.00562494` vs control `0.00272267`) and `0.9999` is much worse (`Δpq=+4.07251530`, `qgap=0.03394086`)
- Runtime is not the reason for the kill: step time stayed near the control (`341.05 ms` and `339.15 ms` vs `337.97 ms`), so the branch should be considered conceptually wrong for this proxy rather than merely too expensive
- `E32` produced a strong same-host P1 win: control `phase1_e32_control_p1-20260320-172313-5aaa1c50` landed at postquant `1.46474630`, while WSD `phase1_e32_wsd_p1-20260320-173257-a73c389c` improved to `1.44033240`
- The win is broad, not a qgap-only fluke: WSD improved prequant by `-0.01689080`, post-roundtrip by `-0.02441390`, and qgap by `-0.00752309`
- Step time moved only modestly (`557.65 ms` → `568.16 ms`, `+1.88%`), which is small enough that WSD should become the active base schedule for the next Track C follow-up rather than being treated as a fragile special case
- `E35` cleanly failed as the first WSD follow-up: control `phase1_e35_wsd_control_p1-20260320-183746-f3f18702` landed at postquant `1.44615320`, while the cooldown-β₂ candidate `phase1_e35_wsd_beta2_p1-20260320-184735-878125dc` regressed to `1.45758252`
- The regression was not a qgap-only trade: prequant also worsened (`1.44512983` → `1.45651059`), qgap ticked up slightly (`0.00102337` → `0.00107193`), and artifact bytes increased (`12,756,384` → `12,797,094`) even though step time improved a little (`639.8 ms` → `630.1 ms`)
- `E28` then provided the next clean answer on top of WSD: control `phase1_e28_control_p1-20260320-192913-2c4b106b` landed at postquant `1.42336352`, while `(30,20)` regressed to `1.43924815` and `(30,15)` regressed further to `1.44641556`
- The family is not uniformly good. The only promote-worthy point was the negative-favored variant `(20,30)`, which improved to postquant `1.40769697`, prequant `1.40659045`, and `qgap=0.00110652`, for `Δpq=-0.01566655` at effectively unchanged step time (`511.9 ms` → `512.6 ms`)
- That means the correct lesson is specific: keep WSD, keep asymmetric `(20,30)`, and do not generalize the win to positive-heavy asymmetric caps
