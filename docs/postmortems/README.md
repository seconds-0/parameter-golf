# Experiment Post-Mortems

This directory holds the detailed closeout reviews for completed non-baseline experiments.

## Standard

An experiment is not fully closed until it has:
- a tracker decision
- the relevant track/interactions docs updated
- a post-mortem in this directory

Each post-mortem answers the same questions:
- What question was actually tested?
- Was the control/baseline choice correct?
- Did the implementation match the intended hypothesis?
- Do the metrics support the current promote/kill/flat decision?
- What would justify reopening the branch later?

## Decision Table

| Experiment | Current decision | Confidence | Main reason | Reopen trigger |
| --- | --- | --- | --- | --- |
| [E03](./e03_exporter_clip_star.md) | Flat / no win | High | Best clip setting was still slightly worse than the trusted baseline, and the default clip was effectively tied | Re-sweep only after a distribution-shaping winner changes the checkpoint |
| [E04](./e04_keep_float_threshold.md) | Flat / no win | High | All cap-safe thresholds were identical; the only quality improvement broke the byte cap badly | Reopen only if tensor size mix or weight distribution changes materially |
| [E23](./e23_ema_export.md) | Killed | Medium | Export-time EMA catastrophically hurt exported quality while the live training path stayed healthy | Reopen only with a materially different EMA design, longer horizon, or a different export integration |
| [E27](./e27_doc_aligned_batching.md) | Killed on current shards | High | The current BOS-delimited packing path sacrificed too much supervision and slowed training heavily | Reopen only with a different packing/data format that preserves supervision density |
| [E28](./e28_asymmetric_logit_rescale.md) | Promoted | High | Only the negative-favored asymmetric softcap `(20,30)` improved post-roundtrip quality, with essentially no runtime cost | Reopen the sweep only if a later training/logit change suggests a different softcap balance |
| [E30](./e30_batch_schedule.md) | Promoted | Medium | The early-small-batch schedule improved post-roundtrip quality dramatically by fitting far more optimizer steps into the same wallclock | Reconfirm on the standard compiled path once the fresh-host compile regression is fixed |
| [E32](./e32_wsd_schedule.md) | Promoted | High | Same-host P1 run improved both prequant and post-roundtrip quality with only modest runtime cost | Revisit only if later compositions reveal incompatibility or a better schedule replacement |
| [E34a](./e34a_polar_express.md) | Neutral / no promote | Medium | `PolarExpress-5` was essentially tied and `PolarExpress-4` was slightly faster but slightly worse, so the branch did not clear the promote bar | Reopen only if a later optimizer lane result suggests the speed/quality trade should be composed or re-measured on the full stack |
| [E34c](./e34c_normuon.md) | Neutral / no promote | Medium | NorMuon improved both prequant and post-roundtrip quality slightly, but only by about `-0.0012` bpb and with a small runtime cost, so it stayed below the promote bar | Reopen if a later stack amplifies optimizer-quality gains or if another Muon result suggests NorMuon composes better there |
| [E35](./e35_cooldown_beta2.md) | Killed | High | On top of WSD, cooldown `β₂` made both prequant and post-roundtrip quality worse | Reopen only with a meaningfully different cooldown policy or schedule family |

## Process Checklist

For each completed experiment:
1. Recompute the decision from the logged metrics and official kill/promote rules.
2. Verify the matched control or baseline really was the right comparator.
3. Review the config/code delta for accidental extra changes.
4. Check metric integrity and any replay/export assumptions that affected the readout.
5. Record the narrow lesson learned and the exact reopen condition.

## Scope Covered Now

The current archive covers every completed non-baseline experiment in the live tracker:
- `E03`
- `E04`
- `E23`
- `E27`
- `E28`
- `E30`
- `E32`
- `E34a`
- `E34c`
- `E35`
