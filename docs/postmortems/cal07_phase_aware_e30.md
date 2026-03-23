# CAL-07 Post-Mortem: Phase-Aware `E30` Proxy

## Question

Did the original short `E30` proxy overstate the batch schedule because it never spent enough time in the later large-batch phase?

## Verdict

No. `E30` stayed strongly positive even after explicitly crossing the phase boundary and collecting multiple post-transition validations.

This means the remaining `CAL-01` failure explanation has to be more specific than "phase 2 exists." The more plausible failure surface is now the interaction between the full-scale stack and the later large-batch phase under WSD, not the mere presence of the later phase in a single-GPU proxy.

## Runs

### Control

- Run: `phase2_cal07_e30_phaseaware_control_p2-20260323-022447-3d42dda8`
- Result dir: [phase2_cal07_e30_phaseaware_control_p2-20260323-022447-3d42dda8](/Users/alexanderhuth/Code/oai-param-golf/experiments/results/phase2_cal07_e30_phaseaware_control_p2-20260323-022447-3d42dda8)
- Final prequant: `1.41549442`
- Final post-roundtrip: `1.41658131`
- Reloaded post-roundtrip: `1.41658423`
- `qgap`: `0.00108689`
- Tokens seen: `287,834,112`

### Candidate (`E30`)

- Run: `phase2_cal07_e30_phaseaware_batch_schedule_p2-20260323-024450-2311dc9a`
- Result dir: [phase2_cal07_e30_phaseaware_batch_schedule_p2-20260323-024450-2311dc9a](/Users/alexanderhuth/Code/oai-param-golf/experiments/results/phase2_cal07_e30_phaseaware_batch_schedule_p2-20260323-024450-2311dc9a)
- Final prequant: `1.37553203`
- Final post-roundtrip: `1.37740077`
- Reloaded post-roundtrip: `1.37739751`
- `qgap`: `0.00186874`
- Tokens seen: `249,036,800`

## Comparison

- Prequant delta: `-0.03996239`
- Post-roundtrip delta: `-0.03918054`
- Reloaded post-roundtrip delta: `-0.03918672`
- `qgap` delta: `+0.00078185`

Even though the candidate saw fewer total tokens, it remained much better on quality while clearly traversing the later schedule stage.

## Transition Read

The point of `CAL-07` was not just the final metric. It was whether the candidate would visibly collapse after the step-`900` batch transition.

What happened instead:

- step `800`: `val_bpb=1.5136`
- step `900`: `val_bpb=1.4912`
- step `1000`: `val_bpb=1.4208`
- step `1100`: `val_bpb=1.3847`
- step `1150`: `val_bpb=1.3755`

So the candidate did not merely survive transition. It improved substantially after transition.

## What This Rules Out

- "The original `E30` win only existed because the short proxy never touched phase 2."
- "Crossing the later `E30` batch stage is itself enough to explain `CAL-01`."

## What It Does Not Rule Out

- `E30` may still fail specifically when layered with `E32` and `E28` at full scale.
- The problematic variable may be the full-scale timing of the transition relative to WSD, not the transition in isolation.
- The problem may still require `8xH100` throughput and token-velocity effects to appear.

## Updated Diagnosis

After `CAL-06` and `CAL-07`, the most credible story is:

- `E30` is genuinely strong at cheap scale
- `E30` remains strong under compilation
- `E30` remains strong in a phase-aware proxy that crosses the later schedule stage
- therefore `CAL-01` is most likely a full-scale interaction problem, not a simple proxy artifact

The remaining decomposition order should now be:

1. `CAL-03`: `E32` alone on full
2. `CAL-04`: `E32 + E28` on full
3. `CAL-05a/b/c`: `E30` full reintroduction and timing variants

## Decision

- Keep `E30` alive as a real recipe component.
- Stop treating "phase 2 exists" as the leading explanation for `CAL-01`.
- Focus the next paid full-run work on standalone-then-layered full decomposition.
