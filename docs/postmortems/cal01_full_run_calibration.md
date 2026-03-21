# CAL-01 Post-Mortem: First Real 8xH100 Calibration

## Decision

Directional failure, but not a clean final-score comparison. The promoted proxy stack did **not** transfer cleanly to a real `8xH100` full run, and the result is strong enough to stop treating `E32 + E28(20,30) + E30` as a trusted full candidate. At the same time, the full-run watchdog policy is too aggressive for calibration runs because it killed the job before export/final exact eval, so we did **not** get a post-roundtrip final metric.

The most important refinement after review is that this was not just “some composed stack failed.” The likely failure surface is narrower:

- `E30` was promoted from a short proxy that effectively validated only phase 1 of the schedule
- `E30` was also promoted under a matched eager fallback, not the compiled regime used by the full run
- `E30`’s batch transition happens almost exactly where the full-run curve peaks and then starts regressing
- under WSD, that transition occurs while LR is still in the stable max-LR phase

So the current best diagnosis is: **real negative transfer, with `E30` phase 2 and its interaction with WSD as the leading suspect, plus a real compile/eager measurement confound**.

## Question tested

Does the current promoted proxy stack:

- `E32` WSD
- `E28` asymmetric softcap `(20,30)`
- `E30` batch schedule

transfer to the real `8xH100` challenge regime strongly enough to justify further full-run iteration on top of it?

## Matched baseline / controls used

- Candidate run: `phase3_calibration_wsd_e28_e30-20260321-024914-5924c918`
- Trusted baseline reference: `baseline_repro-20260319-093041-82dade88`

This was **not** a same-provider matched baseline control. The trusted baseline was produced earlier on a different `8xH100` lane. So the result is a real transfer check, but not yet a same-provider apples-to-apples decomposition.

## Runs reviewed

- [phase3_calibration_wsd_e28_e30-20260321-024914-5924c918/manifest.json](/Users/alexanderhuth/Code/oai-param-golf/experiments/results/phase3_calibration_wsd_e28_e30-20260321-024914-5924c918/manifest.json)
- [phase3_calibration_wsd_e28_e30-20260321-024914-5924c918/metrics.json](/Users/alexanderhuth/Code/oai-param-golf/experiments/results/phase3_calibration_wsd_e28_e30-20260321-024914-5924c918/metrics.json)
- [phase3_calibration_wsd_e28_e30-20260321-024914-5924c918/train.log](/Users/alexanderhuth/Code/oai-param-golf/experiments/results/phase3_calibration_wsd_e28_e30-20260321-024914-5924c918/train.log)
- [phase3_calibration_wsd_e28_e30-20260321-024914-5924c918/launcher.stdout](/Users/alexanderhuth/Code/oai-param-golf/experiments/results/phase3_calibration_wsd_e28_e30-20260321-024914-5924c918/launcher.stdout)
- [baseline_repro-20260319-093041-82dade88/train.log](/Users/alexanderhuth/Code/oai-param-golf/experiments/results/baseline_repro-20260319-093041-82dade88/train.log)

## What changed in code/config

- Full calibration config:
  - [phase3_calibration_wsd_e28_e30.yaml](/Users/alexanderhuth/Code/oai-param-golf/experiments/configs/phase3_calibration_wsd_e28_e30.yaml)
- Same model shape as baseline, but with:
  - WSD schedule
  - asymmetric softcap `(20,30)`
  - `E30` batch schedule
- Run provider:
  - Runpod secure-cloud `8xH100`

## Observed result

- Provider/runtime:
  - Runpod pod `fki567995kmwjx`
  - secure-cloud `8xH100 SXM`
  - actual rate `$21.52/hr`
  - actual run cost about `$4.83`
- Run ended as:
  - `status=failed`
  - `failure_reason=regressing_val_bpb`
  - watchdog kill at about `580.8s`
- Best live validation:
  - best `val_bpb = 1.3372` at step `6200`
- End-of-run live validation before kill:
  - `val_bpb = 1.3785` at step `16000`
- Last four validation points:
  - `1.3685`
  - `1.3749`
  - `1.3765`
  - `1.3785`

Those four points triggered the current watchdog regression rule.

Against the trusted baseline, the calibration stack looked worse at matched step counts and wallclock throughout the run:

- around step `200`: candidate `1.8224` vs baseline `1.6907` (`+0.1317`)
- around step `2000`: candidate `1.4182` vs baseline `1.3242` (`+0.0940`)
- around step `4000`: candidate `1.3764` vs baseline `1.2850` (`+0.0914`)
- around step `6200`: candidate best `1.3372` vs baseline about `1.2684` (`+0.0688`)
- around step `8000`: candidate `1.3535` vs baseline `1.2572` (`+0.0963`)
- around step `12000`: candidate `1.3718` vs baseline `1.2442` (`+0.1276`)
- around step `14000`: candidate `1.3737` vs baseline `1.2192` (`+0.1545`)

That step-matched framing is incomplete, though, because `E30` intentionally changes batch size and therefore changes step density. On an equal-tokens basis, the candidate was much healthier early:

- around `200M` tokens: candidate `1.4372` vs baseline `1.5199` (`-0.0827`)
- around `500M` tokens: candidate `1.3793` vs baseline `1.3830` (`-0.0037`)
- around `900M` tokens: candidate `1.3372` vs baseline `1.3327` (`+0.0045`)
- around `1.0B` tokens: candidate `1.3384` vs baseline `1.3242` (`+0.0142`)
- around `2.0B` tokens: candidate `1.3540` vs baseline `1.2881` (`+0.0659`)

That token-matched view is the key process lesson: the stack was not simply bad from the beginning. It was competitive through phase 1, then clearly failed later.

There is also a notable inflection in the candidate curve:

- best live `val_bpb` occurs at step `6200`
- the `E30` batch schedule transitions at step `6000` (`30%` of `20000`)
- WSD does not begin decaying LR until step `15200` (`75%` of `20000`)

So the candidate appears to hit the large-batch transition while still at max LR, then flatten and regress. That makes the most likely root cause a **schedule-throughput-tokens mismatch**, not just “full runs dislike the whole stack.”

## Why we believe it

- The local artifact set is complete enough to trust the training curve:
  - `manifest.json`
  - `metrics.json`
  - `train.log`
  - `launcher.stdout`
  - trainer snapshot
- The remote host itself was healthy:
  - preflight passed
  - `8` GPUs visible to `nvidia-smi`
  - torch CUDA saw `8` GPUs
  - compile toolchain passed
  - shared dataset/tokenizer were present
- The launcher log shows the workers were terminated by our own watchdog rather than by infra failure.

So this is not an SSH/provider crash disguised as a model result.

## Implementation-error check

No direct implementation bug was found in the launch itself. The calibration config matched the intended promoted stack exactly.

The systems issue is instead in policy:

- the full-run watchdog currently kills any run whose last four `val_bpb` points are strictly increasing
- that rule is reasonable for cheap proxy triage
- it is too aggressive for full calibration because it can prevent final exact eval/export and leave us without the post-roundtrip score we actually care about

This did not create the poor training curve, but it did cut off the final measurement we wanted.

There are two additional non-bug confounds that matter:

- **Compile/eager mismatch:** `E30` was promoted in an eager fallback regime on Vast because fresh-host `torch.compile` was crashing there, but `CAL-01` ran in the normal compiled regime on Runpod. That is not a cosmetic difference; it changes the measurement regime for the strongest proxy win in the stack.
- **Same-provider baseline missing:** we still do not have a same-provider compiled Runpod baseline control, so `CAL-01` is a strong negative transfer signal, but not yet a perfect apples-to-apples causal decomposition.

There is also a process gap that is now explicit:

- **Phase coverage gap:** the winning `E30` proxy never exercised the later large-batch stage in a representative way. So the proxy promoted phase 1 of a staged schedule without really validating phase 2.

## What we learned

- The current promoted proxy stack is **not** a trustworthy full candidate as-is.
- The failure is not best summarized as "worse from the start." On an equal-tokens basis, the candidate was competitive through roughly `900M` tokens and only clearly failed later.
- That means our current proxy-to-full mapping is over-optimistic in a specific way: it allowed a staged schedule to promote without validating the later stage.
- `E30` is the leading suspect because:
  - it contributed the biggest proxy gain by far
  - it was promoted under a matched eager fallback rather than the compiled full-run regime
  - the original short proxy effectively validated only phase 1
  - its batch transition lines up almost exactly with the candidate curve’s inflection point
  - under WSD, that transition happens while LR is still at the stable maximum rather than in decay
- The likely mechanism is not merely “bigger batch is bad.” It is more specifically that `E30` phase 2 may interact badly with WSD at full scale because throughput and effective update scale jump sharply at the transition before LR has started to decay.
- The compile/eager mismatch still matters, but it is now better framed as a real confound layered on top of a likely phase-coverage failure.
- `E32` and `E28` may still be useful, but the combined stack no longer deserves unconditional trust at full scale.
- Full-run monitoring needs a separate policy from proxy monitoring.
- Staged schedules now require phase-aware proxies before promotion. The process lesson is documented separately in [proxy_calibration_meta.md](./proxy_calibration_meta.md).

## Reopen conditions

Reopen this exact stack as a serious full candidate only if:

- a same-provider compiled baseline control on Runpod behaves normally, and
- a rerun with a full-run-safe watchdog still shows the candidate tracking the baseline competitively

Otherwise this calibration should stand as a real negative transfer result.

## Follow-up impact on queue

The next disciplined tranche should be:

1. run a same-provider full baseline control on Runpod under the compiled regime
2. soften or disable the regression-tail watchdog for full calibration runs so we always get final exact/export metrics
3. add a phase-aware `E30` proxy that explicitly crosses the later batch-schedule stage
4. resolve the compile/eager confound at small scale by getting a compiled `1xH100` datapoint for `E30` on Runpod
5. run `E32` alone on full as the highest-confidence individual positive
6. if `E32` is healthy, run `E32 + E28` on full
7. then isolate `E30` specifically:
   - original `E30` transition as the decomposition control
   - a later batch transition
   - an `E30` transition aligned to the WSD decay boundary
   - optionally a shorter WSD stable phase or token-linked schedule if the interaction is confirmed

`E36a/E36b` should not become the default next move. The right next work is to repair our proxy-to-full calibration and isolate which promoted component broke transfer.
