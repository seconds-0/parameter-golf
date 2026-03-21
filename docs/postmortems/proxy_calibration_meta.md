# Proxy Calibration Meta-Postmortem

## Why this exists

`CAL-01` exposed a process problem, not just a bad composed stack. The proxy lane was directionally useful, but it let a staged schedule idea (`E30`) promote without ever exercising the later stage that mattered in the real run.

This review captures the stronger rule set we want going forward so the proxy system stays honest.

## What went wrong

- `E30` is a two-stage schedule:
  - phase 1: `131072` tokens
  - phase 2: `524288` tokens
- The short proxy that promoted `E30` ended around `156.5M` tokens and never reached phase 2 of the schedule in a representative way.
- The first real `8xH100` run was therefore the first time the repo tested the full two-stage schedule under realistic throughput.
- `CAL-01` then showed the likely failure region clearly:
  - the stack was competitive early on an equal-tokens basis
  - the curve peaked around the batch transition
  - the run regressed badly after entering the larger-batch regime while WSD was still at max LR

So the process bug was not "we used proxies." The bug was "we allowed a staged idea to promote from a proxy that only validated phase 1."

## New proxy rules

### 1. Staged ideas must be tested across all intended stages before promotion

This applies to:
- batch schedules
- curricula
- delayed regularizers
- cooldown policies
- multi-phase LR schedules

If a proxy does not cross the later stage, it can only be used for an *early-phase ranking* decision, not for full promotion.

### 2. Measure the axis the hypothesis actually changes

For staged schedule ideas, we should compare all of:
- wallclock
- steps
- tokens seen
- validation checkpoints at phase boundaries

Equal-step or equal-wallclock comparisons alone can be misleading when the hypothesis intentionally changes batch size or throughput.

### 3. Separate proxy roles explicitly

- `P1 fast ranking`
  - cheap early signal
  - useful for killing obviously bad ideas
- `phase-aware proxy`
  - must enter later stages
  - required before promoting staged ideas
- `full calibration`
  - validates actual transfer
  - resets the real reference point

### 4. Treat regime mismatches as blockers, not caveats

If a result is promoted under eager fallback but will be deployed under compiled full runs, get at least one compiled datapoint before treating the idea as a trusted full candidate.

### 5. Full runs should be used sooner when the proxy contract is weak

If phase-aware proxies still fail to predict full behavior after this decomposition tranche, then the proxy lane should become:
- good for killing losers
- insufficient by itself for promoting winners

In that world, we should simply do more Runpod full confirmations sooner.

### 6. Risky ideas must pass standalone and layered checks

For changes that can alter optimization dynamics rather than just a local computation, do not jump straight from "proxy win" to "composed base ingredient."

This applies especially to:
- staged schedules
- batch-size or throughput changes
- delayed regularizers
- cooldown policies
- ideas promoted under eager fallback but intended for compiled full runs

For those classes, the minimum ladder is:
- individual test: does the idea help on its own in the relevant regime?
- layered test: does it still help on top of the active base?

The point is to separate:
- standalone effect
- interaction effect

`CAL-01` is the example that forces this rule. Even if `E30` phase 1 is genuinely useful, that still does not imply `E30 + E32` is healthy at full scale.

### 7. Full-run data completeness must be verified, not assumed

The later `CAL-03` decomposition attempt exposed a second process bug:
- the Runpod dataset bootstrap was still filling in train shards in the background
- the first `CAL-03` started on only `116` train shards
- the trusted Runpod baseline `CAL-02` had used the full `195`
- the next layered rung (`CAL-04`) would therefore have compared against a different training set unless caught manually

That means "dataset dir exists" is not a strong enough preflight check for paid full runs.

For full runs we now require:
- the dataset manifest to be present
- the on-disk train and val shard counts to match the manifest exactly
- the tokenizer path to match the manifest-selected tokenizer artifact
- the dataset bootstrap lock to be free before launch

And the launcher should record that dataset snapshot in the run manifest so mismatches are visible later without log archaeology.

## Current best hypothesis after CAL-01

- `E30` phase 1 is probably genuinely useful
- `E30` phase 2 under the current WSD timing is the likely failure surface
- the proxy lane over-promoted the full schedule because it never validated the later phase
- compile/eager mismatch still matters, but it is not the only plausible explanation anymore

## What changes now

The next tranche should follow this order:

1. same-provider Runpod baseline control
2. full-run-safe watchdog policy
3. phase-aware `E30` proxy that crosses the transition
4. compiled `1xH100` Runpod E30 check
5. full decomposition with standalone-then-layered structure (`E32`, then `E32 + E28`, then `E30` variants)

If those phase-aware and compiled checks still do not calibrate the proxy lane, we should accept that this repo needs more real `8xH100` runs for promotion decisions than we originally hoped.
