# Track A: Tokenizer + Embedding Economics

## Thesis
The current SP-1024 tokenizer may not be optimal. A time-capped run is really a bytes-trained-per-second problem. Changing the tokenizer changes bits/token, bytes/token, sequence fragmentation, embedding size, and artifact bytes — all at once. This is the highest-probability structural lever after exporter tuning.

## Experiments
- **E05**: Tokenizer audit bundle (offline, near-zero GPU)
- **E06**: SP-512 P1
- **E07**: SP-768 P1 (highest-probability winner)
- **E08**: SP-1536 P1
- **E09**: Best tokenizer P2 (winner from E06-E08)
- **E10**: Tied-embed LR star on tokenizer winner (overlaps Track C)

## Key Metrics
- bytes/token, tokens/byte for each vocab size
- effective_train_bytes_per_second = tok_s × bytes_per_token
- Sequence fragmentation (median sequence length for fixed raw text)
- Post-roundtrip Δpq (the only ranking metric)

## Decision Rules
- Kill a tokenizer at P1 if Δpq ≥ +0.004 or step time AND fragmentation are both worse
- Promote if Δpq ≤ -0.003, or within 0.001 of baseline with meaningful artifact slack gain
- Only ONE tokenizer winner goes to composition

## Status
Not started. Depends on: X-06 (tokenizer stats utility), P-01 (config-driven paths).

## Learnings
(Updated as experiments complete)
