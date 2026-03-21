# CAL-01 Post-Mortem: First Real 8xH100 Calibration

## Decision

Directional failure, but not a clean final-score comparison. The promoted proxy stack did **not** transfer cleanly to a real `8xH100` full run, and the result is strong enough to stop treating `E32 + E28(20,30) + E30` as a trusted full candidate. At the same time, the full-run watchdog policy is too aggressive for calibration runs because it killed the job before export/final exact eval, so we did **not** get a post-roundtrip final metric.

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

Against the trusted baseline, the calibration stack was worse at matched training stages throughout the run:

- around step `2000`: candidate `1.4182` vs baseline `1.3242` (`+0.0940`)
- around step `4000`: candidate `1.3764` vs baseline `1.2850` (`+0.0914`)
- around step `8000`: candidate `1.3535` vs baseline `1.2572` (`+0.0963`)
- around step `12000`: candidate `1.3718` vs baseline `1.2442` (`+0.1276`)
- around step `14000`: candidate `1.3737` vs baseline `1.2192` (`+0.1545`)

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

## What we learned

- The current promoted proxy stack is **not** a trustworthy full candidate as-is.
- The failure is not just late-run wobble. The stack underperformed the trusted baseline from early training onward.
- That means our current proxy-to-full mapping is over-optimistic.
- `E30` is the leading suspect because:
  - it contributed the biggest proxy gain by far
  - it was promoted under a matched eager fallback rather than the compiled full-run regime
- `E32` and `E28` may still be useful, but the combined stack no longer deserves unconditional trust at full scale.
- Full-run monitoring needs a separate policy from proxy monitoring.

## Reopen conditions

Reopen this exact stack as a serious full candidate only if:

- a same-provider compiled baseline control on Runpod behaves normally, and
- a rerun with a full-run-safe watchdog still shows the candidate tracking the baseline competitively

Otherwise this calibration should stand as a real negative transfer result.

## Follow-up impact on queue

The next disciplined tranche should be:

1. run a same-provider full baseline control on Runpod under the compiled regime
2. soften or disable the regression-tail watchdog for full calibration runs so we always get final exact/export metrics
3. do targeted full ablations starting with `E30` before opening new proxy branches

`E36a/E36b` should not become the default next move. The right next work is to repair our proxy-to-full calibration and isolate which promoted component broke transfer.
