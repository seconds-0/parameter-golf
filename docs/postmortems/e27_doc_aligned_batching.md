# E27 Post-Mortem: Document-Aligned Batching

## Decision
Killed on the current published BOS-delimited shard format.

## Question Tested
Can document-respecting training windows improve quality by avoiding cross-document training targets on the published SentencePiece shards, without regenerating data?

## Matched Baseline / Controls Used
- Control: `phase1_e27_control_p1-20260320-155905-eb77f1a0`
- Candidate: `phase1_e27_doc_aligned_p1-20260320-160818-f96fe403`
- Same proxy level, same seed, same host class

## Runs Reviewed
- Control postquant: `1.41199216`
- Candidate postquant: `1.52791342`
- Candidate `Δpq`: `+0.11592126`
- Candidate `qgap`: `0.01193720` vs control `0.00408529`
- Candidate step time: `640.23 ms` vs control `416.29 ms`

## What Changed in Code / Config
- Config delta was narrow: `DOC_ALIGNED_BATCHING=1`
- Implementation used `doc_batching.py` plus training-side loader changes in `train_gpt.py`
- The loader:
  - split documents on `bos_id`
  - kept windows within document boundaries
  - padded short tails with `bos_id`
  - masked padded targets with `-100`
- Telemetry added:
  - `train_supervised_tokens_seen`
  - `ignored_target_tokens_seen`
  - `supervised_target_fraction`

## Observed Result
- The candidate was much worse on quality and speed.
- The critical telemetry explains why: only about `69.17%` of target positions remained supervised.
- The run ended with `75,972,568` ignored targets, which is too much supervision waste for this proxy budget.

## Why We Believe It
- This is not a small noisy miss; it is a large regression on multiple axes with a mechanistic explanation from the telemetry.
- The new metrics line up with the intended failure mode: the loader respected boundaries, but the shard format plus naive padding wasted too many targets.

## Implementation-Error Check
- The implementation matched the experiment question: this was explicitly a “use the existing BOS-delimited shards, do not regenerate data” test.
- The result does not look like a parser bug or logging error because the quality regression, speed regression, and supervision-fraction telemetry all tell the same story.
- The correct interpretation is narrow: the branch is killed on the current shard/packing path, not forever in principle.

## What We Learned
- Respecting document boundaries is not free when the current data format forces too much padding and masking.
- For this repo, supervision density matters more than the hoped-for boundary-respect benefit at this proxy.
- This should not remain in the active queue unless the packing/data story changes materially.

## Reopen Conditions
- A different packing strategy is introduced that preserves supervision density.
- Data is regenerated or repacked in a way that lets document alignment avoid the current masking overhead.

## Follow-Up Impact on Queue
`E27` should stay dead in the live queue. It should not compete with `E28`, `E30`, or `E34` unless the underlying data/packing assumptions change.
