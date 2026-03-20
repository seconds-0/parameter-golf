# Submission Checklist

Use this only when we have a real candidate worth packaging. Until then, the default is still: follow [tracker.md](/Users/alexanderhuth/Code/oai-param-golf/docs/tracker.md), finish the planned experiment funnel, and avoid spending attention on submission mechanics too early.

## Current Rule Of Thumb

- Do not start preparing a leaderboard PR just because a proxy result looks good.
- Treat submission packaging as a final-stage task after the experiment funnel has produced a trustworthy full-run candidate.
- Prefer finishing the planned experiment list and composition work before attempting to optimize for leaderboard process.

## Upstream Submission Bar

These are the repo-level requirements described in the upstream `README.md`:

- New SOTA submissions must beat the current SOTA by at least `0.005` nats.
- They must provide enough run logs to support `p < 0.01`.
- They must reproducibly run in under `10 minutes` on `8xH100` for training.
- Evaluation must also respect the upstream time restriction.
- The submission artifact must stay under `16,000,000` total bytes.
- Tokenizer or dataset changes require especially careful proof that `val_bpb` is computed correctly.

Practical note:

- The leaderboard is displayed in `val_bpb`, but the required improvement margin is written in nats.
- `0.005 / ln(2) ≈ 0.00721 bpb`.
- So a safe record target is roughly “current SOTA minus `0.0072 bpb` or better,” with enough buffer to survive verification noise.

## Where A Submission Goes

- Main leaderboard candidate: `records/track_10min_16mb/<date>_<name>/`
- Non-record candidate: `records/track_non_record_16mb/<date>_<name>/`

The submission PR should only add a new folder under the appropriate `records/` subdirectory.

## Minimum Required Files

Each submission folder should include:

- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`
- any other dependencies required to reproduce the run from inside that folder

The helper script [submit.py](/Users/alexanderhuth/Code/oai-param-golf/experiments/scripts/submit.py) can scaffold this, but the output should be treated as a starting point, not a finished submission.

## Our Quality Bar

Before opening a submission PR, make sure the folder is good enough that an external reviewer can understand and reproduce it without guesswork.

### README expectations

The `README.md` should clearly state:

- what changed relative to the matched baseline
- why the approach should help
- exact run ID or run IDs
- whether the submission is record-track or non-record
- the exact training command or config used
- post-roundtrip `val_bpb`
- prequant `val_bpb`
- `qgap`
- total artifact bytes
- code bytes
- compressed model bytes
- wallclock and step-time evidence
- hardware used
- statistical evidence if making a record claim
- any caveats, known fragility, or replay details

### `submission.json` expectations

At minimum include:

- author name
- GitHub ID
- run name
- short blurb
- date
- final `val_loss`
- final `val_bpb`
- total bytes
- code bytes

If useful, include extra metadata too, but do not omit the upstream-required basics.

### Reproducibility expectations

Before packaging a record candidate, verify:

- full run completed successfully on the intended track hardware
- parsed metrics match the claimed score
- replay / export path is trusted for that run family
- code snapshot in the record folder matches what produced the run
- all required dependencies are present in the record folder if they are not upstream-stable

## Internal Submission Gate

Treat a run as “ready to package” only if all of the following are true:

- it is a real full-run candidate, not just a proxy win
- it clears the upstream SOTA bar with margin
- it has enough repeated evidence for the required significance bar
- it fits the artifact cap
- it satisfies the runtime constraint on the required hardware
- the record folder is polished enough for external review

If any of those are missing, keep iterating through the experiment plan instead of opening a submission PR.

## Recommended Packaging Flow

1. Identify the candidate run or run bundle.
2. Confirm it meets the record-track or non-record-track requirements.
3. Generate the record folder with `experiments/scripts/submit.py`.
4. Manually improve `README.md` until it meets the quality bar above.
5. Verify the included code and logs match the claimed run.
6. Place the folder under the correct `records/` track.
7. Open a PR that only adds that record folder.

## Default Team Policy

For this repo, the default policy is conservative:

- no rush to get onto the leaderboard
- do not let submission mechanics distract from the experiment funnel
- prioritize finishing the planned experiment list and composition work first
- use this checklist only when we intentionally decide a candidate is ready for packaging
