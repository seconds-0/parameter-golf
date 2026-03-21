# Track C: Recipe, Schedule & Optimizer Tuning

## Thesis
In this small-vocab tied-embedding regime, the embedding matrix is unusually load-bearing. The separate TIED_EMBED_LR knob matters more than generic LR sweeps. Small, targeted optimizer stars (not grids) on the tokenizer winner can unlock cheap gains. Additionally, training schedule changes (batch size, logit processing) are zero-artifact-cost recipe improvements drawn from the [NanoGPT speedrun](../references/nanogpt_speedrun_techniques.md).

## Experiments
- **E10**: Tied-embed LR star: test {0.03, 0.05, 0.08} on tokenizer winner (3-point, ~0.3 H100-hrs)
- **E11**: Matrix/scalar LR star: test MATRIX_LR ∈ {0.025, 0.04, 0.06} on tokenizer winner (3-point, ~0.3 H100-hrs)
- **E12**: Embedding norm penalty A/B
- **E23**: EMA weight averaging at export (decay sweep: 0.999, 0.9999) — inspired by [Q Labs 10x blog](../references/qlabs_10x_data_efficiency.md). ~20 lines of code. Maintain shadow EMA weights during training, swap in at export time. Targets qgap reduction via smoother weights. Kill if qgap doesn't improve by ≥10% or val_bpb regresses by >0.002. **Complete, killed** — both tested decays catastrophically worsened the exported checkpoint on the P1 proxy.
- **E28**: Asymmetric softcap — replace symmetric `30 * tanh(logits/30)` with separate caps for positive vs negative logits: `cap_pos * tanh(logits/cap_pos)` for logits > 0, `cap_neg * tanh(logits/cap_neg)` for logits ≤ 0. Test (cap_pos, cap_neg) ∈ {(30,20), (30,15), (20,30)}. Kill if Δpq ≥ +0.002. Ref: [NanoGPT speedrun 1.9](../references/nanogpt_speedrun_techniques.md#19-asymmetric-logit-rescale). **Complete, promoted** — `(20,30)` won cleanly on the WSD base while the other two asymmetric settings regressed.
- **E30**: Batch size schedule — two-stage: `TRAIN_BATCH_TOKENS=131072` (¼ baseline) for the first 30% of steps, then `524288` (baseline) for the remaining 70%. Implemented via `BATCH_SCHEDULE`, not a second config family. More gradient updates early = faster initial learning. Kill if Δpq ≥ +0.003. Ref: [NanoGPT speedrun 1.3](../references/nanogpt_speedrun_techniques.md#13-batch-size-schedule). **Complete, promoted** — strong same-host P1 win under the matched eager fallback.
- **E32**: WSD LR schedule — Warmup-Stable-Decay replacing the current baseline warmdown path. MiniCPM (arXiv:2404.06395) + ICLR 2025 showed WSD outperforms cosine-like schedules. Three phases: 1% warmup, 75% stable, 24% cosine decay. Zero artifact cost. Kill if val_bpb regresses >0.003 vs the matched baseline schedule. Ref: [small model landscape](../references/small_model_optimization_landscape.md#1-wsd-learning-rate-schedule-minicpm--tsinghua). **Complete, promoted** — strong same-host P1 win.
- **E34a**: Polar Express — drop-in replacement for `zeropower_via_newtonschulz5()` using minimax-optimal spectral polynomial. Proven in NanoGPT speedrun record #38. ~50-100 lines, pure PyTorch. Code: github.com/NoahAmsel/PolarExpress. Test both reducing iterations (5→4) and keeping 5 for better quality. Kill if Δpq ≥ +0.003 AND step time doesn't improve. **Complete, neutral / no promote** — `PolarExpress-5` was essentially tied on quality and `PolarExpress-4` traded a tiny quality loss for a tiny speed gain.
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
Partially unblocked. Tokenizer-dependent recipe work `E10`-`E12` still depends on `E09` (tokenizer winner selected), while the cheap side experiments `E30` and `E34` remain unblocked by `E02`. `E23`, `E24a`, and `E35` are complete and killed, `E34a` is complete and neutral / no promote, `E34c` is complete and mildly positive but below the promote bar, and `E32`, `E28`, and `E30` are now complete and promoted. The active Track C base is still WSD plus asymmetric softcap `(20,30)` plus the `E30` early-small-batch schedule. Detailed closeout reviews: [E23](../postmortems/e23_ema_export.md), [E24a](../postmortems/e24a_fixed_weight_decay.md), [E28](../postmortems/e28_asymmetric_logit_rescale.md), [E30](../postmortems/e30_batch_schedule.md), [E32](../postmortems/e32_wsd_schedule.md), [E34a](../postmortems/e34a_polar_express.md), [E34c](../postmortems/e34c_normuon.md), [E35](../postmortems/e35_cooldown_beta2.md).

The recommended next Track C order is:
- first real `8xH100` calibration on Runpod for the promoted WSD + asymmetric `(20,30)` + `E30` base
- `E36a/E36b` if we want the cheapest eval-only follow-up after that calibration
- `E24b`, `E33`, or `E13` if we want to keep pushing Track B training-aware regularization one branch at a time
- `E34b` remains the "stay in Muon internals" fallback only if the optimizer lane looks newly attractive after the full calibration

`E30` was worth promoting immediately because the matched same-host eager-fallback bundle was decisive: control `phase1_e30_control_p1-20260320-211549-b83629b3` landed at postquant `1.46588267`, while `phase1_e30_batch_schedule_p1-20260320-212550-76454fcd` improved to `1.41664772` (`Δpq=-0.04923495`). The important caveat is that the paired run used `TORCHDYNAMO_DISABLE=1` due a fresh-host compile regression on Vast, so future Track C side branches should either use the same matched eager fallback or fix the compile path before switching measurement regimes.

Current prioritization thesis:
- The strongest live recipe signal is "better training steps" rather than "harder exporter tuning." `E32`, `E28`, and `E30` all improved prequant quality materially, while exporter-only knob sweeps stayed flat and current `qgap` is already small.
- That makes optimizer-improvement branches like `E34a` and `E34c` higher-EV than most remaining cheap experiments, as long as they stay low-overhead and easy to kill.
- It also means overhead-heavy ideas such as `E31a` MTP should stay later in the queue until we have stronger evidence that richer-but-heavier steps beat simply fitting better updates into the wallclock.

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
- `E30` then produced the strongest cheap Track C win so far on top of that base: control `phase1_e30_control_p1-20260320-211549-b83629b3` landed at postquant `1.46588267`, while the batch-schedule candidate `phase1_e30_batch_schedule_p1-20260320-212550-76454fcd` improved to `1.41664772`
- The mechanism is straightforward and useful: the early-small-batch schedule processed fewer total tokens (`156.5M` vs `206.0M`) but fit `1194` optimizer steps into the wallclock instead of `393`, and that step-density gain dominated the slight `qgap` regression (`0.00119921` → `0.00139435`)
- The one caveat is measurement regime: the paired bundle ran under `TORCHDYNAMO_DISABLE=1` because fresh-host compiled warmup is currently crashing on Vast. Treat the win as real within that matched regime, and keep later quick Track C follow-ups in the same regime unless the compile blocker gets fixed
- `E34a` came back neutral but informative: control `phase1_e34a_control_p1-20260320-222847-48fb49f2` landed at postquant `1.40742446`, `PolarExpress-5` landed at `1.40728640`, and `PolarExpress-4` landed at `1.40755582`
- That is far too small to promote on quality, but it does keep the optimizer lane alive: the 5-step polynomial stayed essentially tied, and the 4-step variant bought a small speed edge (`235.3 ms` → `232.2 ms`) at a very small quality cost
- `E34c` then answered the next optimizer-quality question on the promoted WSD + asymmetric `(20,30)` + `E30` base: control `phase1_e34c_control_p1-20260321-004949-c4a83875` landed at postquant `1.41100907`, while the NorMuon candidate `phase1_e34c_normuon_p1-20260321-010303-d12f0bd7` improved slightly to `1.40981103`
- The direction is encouraging but not decisive. NorMuon improved prequant by about `-0.0011`, post-roundtrip by about `-0.0012`, and qgap by about `-0.00010`, while step time got about `+1.9%` slower (`242.95 ms` → `247.54 ms`)
- The first implementation attempt did fail immediately due to a dtype bug in the adaptive second-momentum buffer. That is now fixed by keeping the buffer in fp32, and the rerun is the only result that should count. With that fix in place, the branch is a valid near-miss rather than a broken experiment
- The right lesson is "optimizer lane still alive, but cash out next." `E34c` did not clear the `-0.003` promote bar, and `E24a` then failed catastrophically, so the active base stays unchanged and the next branch should be the first real `8xH100` calibration rather than another same-family tweak by default
