Below is the plan I would hand to the engineer.

The governing rule is simple: no idea earns expensive compute until it wins on the only metric that matters, **post-roundtrip validation bits per byte**, on a matched cheap proxy.

## 1. Portfolio framing

The objective is not "get a better training curve." The objective is to produce a **self-contained artifact under 16,000,000 bytes, where code bytes and compressed model bytes both count, that reproduces the lowest possible FineWeb validation bits per byte after the exact export roundtrip, and still finishes under 10 minutes on 8×H100 for leaderboard runs**. The public baseline is already close to the cap: it scores 1.2244 post-roundtrip `val_bpb`, uses 15,863,489 total bytes, and leaves only 136,511 bytes of slack. The 4-hour non-record run keeps the same 9×512 SP-1024 layout and nearly the same step time, but most of its extra pre-quant gain leaks away at export: the baseline's pre/post gap is about 0.0072 bpb, while the 4-hour run's gap is about 0.0325 bpb; by arithmetic, only about 40 percent of the longer run's pre-quant gain survives roundtrip. That is the single strongest argument for putting export robustness ahead of architecture heroics. ([GitHub][1])

The dominant uncertainty axes are fivefold. First, **tokenizer economics**: bits per token and bytes per token both matter, and a time-capped run is really a bytes-trained-per-second problem in disguise. Second, **export retention**: improvements that do not survive the trainer's row-wise clipped int8 plus zlib path are fake gains. Third, **throughput and wallclock risk**: the run must fit under 600 seconds on 8 GPUs, and the trainer changes gradient accumulation when world size changes, so 1-GPU and 8-GPU timing are not interchangeable. Fourth, **artifact economics**: the exporter keeps named control tensors and all small float tensors up to 65,536 elements in float or fp16, so "just add a few gates" is not free. Fifth, **challenge-spirit risk**: the README explicitly reserves the right to disqualify out-of-spirit use of external compute or brute-force seed abuse. ([GitHub][2])

I would convert the dossier into six experimental tracks. Track A is **tokenizer plus embedding economics**. Track B is **export-aware quantization and exporter tuning**. Track C is **short-budget recipe and optimizer tuning**, especially embedding-specific knobs the trainer already exposes. Track D is **lightweight architecture rebudgeting using existing knobs**, such as KV-head allocation and width re-spend. Track E is **parameter sharing and recurrence moonshots**. Track F is **systems and harness discipline**, meaning telemetry, cheap proxies, promotion logic, and code-size hygiene. The repo scaffold already has useful orchestration pieces such as config validation, metadata fields, status, watch, collect, budget reporting, and results folders; the mistake would be to treat those as permission to launch undisciplined sweeps. ([GitHub][2])

The likely cheap winners are: tokenizer shortlist work, exporter-only tuning, embedding learning-rate and norm discipline, clamp-aware export training, and small local optimizer stars. The high-risk moonshots are: layer sharing with width rebudget, depth recurrence, and any architecture family swap. The things that are probably not worth testing early are: full byte-level modeling, full-factorial learning-rate grids, large architecture rewrites, test-time compute tricks, and anything distillation-like until the organizers explicitly bless it. Your scaffold currently ships a 15-config matrix/scalar learning-rate grid that assumes 10-minute runs; that is exactly the kind of plan to reject at the start. ([GitHub][3])

## 2. Baseline replication and instrumentation

The first operational step is to use the scaffold's existing `experiments/configs/baseline.yaml` and reproduce the public NaiveBaseline, because that config already mirrors the published 9-layer, 512-dim, 1024-vocab, tied-embedding, 600-second setup. Do not touch model ideas before this run is trustworthy. The acceptance gate for trust is: one 8×H100 reproduction within about 0.003 post-roundtrip `val_bpb` of 1.2244, within about 5 percent of the published 43.54 ms step average, with total artifact bytes within about 50 kB of the record. If that gate is missed, the work is infrastructure, not research. ([GitHub][4])

For every run, save the following artifacts locally in the result folder: the resolved config, manifest, raw `train.log`, parsed `metrics.json`, exact trainer snapshot, exact git SHA, machine name, Python and PyTorch versions, `final_model.int8.ptz`, and also the raw `final_model.pt`. That last one matters because the trainer already saves it, but your current `run.sh` only copies back the int8 artifact; without `final_model.pt`, exporter-only sweeps and exact roundtrip investigations become clumsy or impossible. ([GitHub][2])

The mandatory metrics on every run are not negotiable. Store and display: pre-quant `val_loss` and `val_bpb`; post-roundtrip `val_loss` and `val_bpb`; the quantization delta in both units; compressed model bytes; code bytes; total artifact bytes; artifact slack to 16,000,000 bytes; step average in milliseconds and, once added, tokens per second; total train tokens seen; parameter count; peak allocated and reserved memory; seed; world size; gradient accumulation; and the exact config diff from baseline. Also add two derived fields that should become first-class everywhere: `Δpq`, defined as candidate post-roundtrip `val_bpb` minus matched-baseline post-roundtrip `val_bpb`, where negative is better; and `qgap`, defined as post-roundtrip `val_bpb` minus pre-quant `val_bpb`, where smaller is better.

Before touching model ideas, make six code changes.

First, fix the scaffold's path rigidity. `run.sh` currently forces `DATA_PATH` and `TOKENIZER_PATH` to the SP-1024 defaults and blindly runs `cached_challenge_fineweb.py --variant sp1024`; `preflight.py` also hardcodes those same expected paths. That means tokenizer experiments are only superficially supported right now. Change both files so the dataset path and tokenizer path come from config, with defaults only when unspecified, and have preflight validate the paths for the selected config rather than hardcoded SP-1024 paths. ([GitHub][5])

Second, expose exporter controls as config flags. The trainer currently hardcodes `INT8_CLIP_PERCENTILE=99.99984`, `INT8_KEEP_FLOAT_MAX_NUMEL=65536`, fp16 scale storage, and the float passthrough control-tensor patterns. Make those env-backed so the harness can tune them without source edits. ([GitHub][2])

Third, add better trainer telemetry. The parser already knows how to read an optional `tok_s` field on train logs, but the trainer does not emit it. Add `tok_s`, `train_tokens_seen`, `final_prequant_exact`, and explicit quantization-delta log lines. Also log the quantization payload bytes and payload ratio that the trainer already computes but the parser currently ignores. ([GitHub][6])

Fourth, extend `parse_log.py`. It already parses train steps, validation steps, peak memory, code bytes, compressed model bytes, total artifact bytes, and exact post-roundtrip metrics, and it merges estimated cost and metadata from the manifest. Extend it to compute and persist pre-quant final metrics, `qgap`, artifact slack, total train tokens seen, config diff versus baseline, payload bytes, payload ratio, and proxy-relative deltas. ([GitHub][6])

Fifth, improve the displays. `compare.py` currently sorts almost entirely by final `val_bpb` and prints only a thin summary, even though it already has access to params and peak memory. Add columns for pre-quant and post-quant metrics, `qgap`, artifact slack, step time, tokens per second, total train tokens, and baseline-relative deltas. Update live status in `launch.py` so it surfaces more than latest train loss and latest `val_bpb`. ([GitHub][7])

Sixth, shrink the budget before the experiments shrink it for you. `machines.yaml` is currently configured with only 8-GPU machines and a `budget_total` of 5000. Add at least one 1×H100 target and one local target, then reduce the default budget to the actual pre-8× limit and add phase caps. The present machine file encourages premature escalation. ([GitHub][8])

## 3. Cheap proxy design

Use four proxy levels.

**P0: integrity smoke.** Run locally or on any single GPU for 2–5 minutes. Use the challenge's small-shard smoke pattern: a tiny training subset, no periodic validation, and a single final validation/export pass. This is for catching crashes, data-path mistakes, tokenizer mismatches, parser regressions, and artifact-size bugs. It predicts operational sanity only. It does not predict leaderboard quality. The challenge README explicitly supports small local smoke subsets with `--train-shards 1` and final-only validation. ([GitHub][1])

**P1: fast single-GPU relative proxy.** Run on 1×H100 for about 4–6 minutes with the same sequence length and, ideally, the same global batch tokens as the 8-GPU run so the optimization geometry is at least directionally matched. Because the trainer enforces `grad_accum_steps = 8 // world_size`, a 1-GPU run is not timing-equivalent to an 8-GPU run, but it is still useful for ranking nearby variants within the same family. P1 is intended to predict the sign and rough magnitude of short-run `Δpq`, early `qgap`, and whether a change hurts step time badly. It cannot predict distributed wallclock precisely. ([GitHub][1])

**P2: medium single-GPU export proxy.** Run on 1×H100 for about 15–20 minutes. Keep the exact tokenizer, dataset variant, sequence length, and export path you would use in a serious run. P2 is the main promotion gate for model ideas. It predicts post-roundtrip ranking much better than P1, especially for tokenizer and export-aware ideas, and is the place to run second seeds. It still cannot predict 8-GPU communication, compile overhead, or every late-stage training effect.

**P3: 8×H100 runtime rehearsal.** Run for 90–120 seconds with `VAL_LOSS_EVERY=0` or a single final validation. This is the wallclock and memory proxy, not the quality proxy. It predicts whether the candidate really fits under 600 seconds on the target hardware and whether compilation and warmup overhead are acceptable. It is mandatory for any candidate that materially changes model shape or step time. The public baseline reached 13,780 steps in 600 seconds at 43.54 ms/step; use that as the timing anchor. ([GitHub][9])

There should also be three dedicated diagnostics that cut across those proxy levels.

The first is a **post-quant robustness proxy**. For every nontrivial candidate, measure exact pre-quant final `val_bpb`, exact post-roundtrip `val_bpb`, and `qgap`. Never promote on pre-quant improvement alone. In this challenge, a candidate with a smaller raw gain but a better retained gain is the better candidate. That is not a theoretical warning; the public 4-hour run is the empirical example. ([GitHub][9])

The second is a **tokenizer economics proxy**. Before any training, compute bytes per token, tokens per byte, median sequence length in tokens for a fixed raw-text chunk size, tail fragmentation, and a short audit of wasted vocabulary pieces. During P1 and P2, also compute `effective_train_bytes_per_second = tok_s × bytes_per_token`. This proxy predicts whether a tokenizer is likely to move the real tradeoff rather than just moving token counts around. It does not predict whether the model can actually exploit the new segmentation; that still requires training.

The third is a **throughput-risk proxy**. On P1 use relative step time versus the matched P1 baseline. On P3 use absolute step time and memory against the published 8-GPU baseline. Tokenizer and export changes usually do not need P3 immediately; shape changes do.

The calibration plan is simple and cheap. In the first two weeks, run a calibration set of six variants across P1 and P2: the baseline, one lower-vocab tokenizer, one higher-vocab tokenizer, one exporter-only tweak, one embedding-learning-rate tweak, and one small clamp-aware regularizer. For the top two or three by P2, run P3 and then one full 8-GPU run. Compute rank correlation on `Δpq` between P1 and P2, and then between P2 and full 8-GPU. Use relative deltas against a matched baseline on the same hardware and same day whenever possible, not absolute scores. Expect P2 `Δpq` and `qgap` to correlate best; expect P0 quality metrics and offline tokenizer audits to correlate worst. That is fine. They are there to kill bad ideas cheaply, not to replace the real measurement.

The honest error model is this: P0 is only for breakage. P1 is good for nearby recipe and export ideas, mediocre across tokenizer families, and poor across architecture changes. P2 is the real cheap proxy. P3 is only about fit. Tokenizer offline metrics are suggestive, not decisive. Exporter-only sweeps can reveal free wins, but they cannot tell you how training would adapt if the exporter changed.

## 4. Sequential experiment design

Use **sequential halving** everywhere. For any bundle, run one seed at the cheapest meaningful proxy. Keep only the top third by the following ordering: best post-roundtrip `Δpq`, then smaller `qgap`, then more wallclock headroom, then more artifact slack. Only survivors get a longer run. Only one surviving winner from each track is allowed into composition. No full factorials. No 15-point grids. No multi-track combo testing before singleton evidence.

For runtime intuition, treat P1 as about 0.08–0.10 H100-hours, P2 as about 0.25–0.33 H100-hours, P3 as about 0.27 H100-hours, and a full 10-minute 8×H100 run as about 1.33 H100-hours. The README's rough figure of about $20 per hour for an 8×H100 box implies about $0.67 for a 2-minute P3 and about $3.33 for a full 10-minute 8-GPU run. ([GitHub][1])

### Phase 0: trust the measurements

**E00 — Baseline P0 smoke.** Hypothesis: the harness can complete a run end-to-end and produce parseable metrics. Minimal code change: none beyond telemetry patching. Expected sign: neutral. Proxy: P0. Promote if the run completes, writes artifacts, and parses correctly. Kill criterion: any failure is infrastructure work, not model work. Runtime: local or about 0.03 H100-hours. Singleton.

**E01 — Baseline P1 control.** Hypothesis: P1 is stable enough to use as a matched baseline for nearby deltas. Minimal code change: none. Expected sign: neutral. Proxy: P1. Promote if repeated P1 controls are within about 0.002 `val_bpb` and 5 percent step time. Kill criterion: if P1 is too noisy, lengthen P1 before testing ideas. Runtime: about 0.1 H100-hours. Singleton.

**E02 — Baseline full 8×H100 reproduction.** Hypothesis: the scaffold can match the public baseline closely enough to trust expensive promotions. Minimal code change: none; use the existing `baseline.yaml`. Expected sign: neutral. Proxy: full 8×H100. Promote if within about 0.003 post-roundtrip `val_bpb`, about 5 percent step time, and about 50 kB total bytes of the public baseline. Kill criterion: fix harness until this is true. Runtime: 1.33 H100-hours. Singleton. ([GitHub][4])

### Phase 1: free or nearly free wins

**E03 — Exporter clip-percentile star.** Hypothesis: the hardcoded clip percentile is not optimal for post-roundtrip score or compressed bytes. Minimal code change: expose `INT8_CLIP_PERCENTILE` as an env var and add an export-only eval path that loads `final_model.pt`. Expected sign: positive on post-roundtrip quality and-or bytes, neutral pre-quant. Proxy: export-only plus P1 exact roundtrip eval. Promote if any setting improves post-roundtrip `val_bpb` by at least 0.002 or saves at least 100 kB with no more than 0.001 bpb loss. Kill criterion: all settings are inside noise. Runtime: about 0.05 H100-hours per setting. Bundle. ([GitHub][2])

**E04 — Keep-float threshold and pattern star.** Hypothesis: the split between int8 matrices and float passthrough tensors is leaving bytes or quality on the table. Minimal code change: expose `INT8_KEEP_FLOAT_MAX_NUMEL` and passthrough patterns; parse payload ratio. Expected sign: positive on the size/quality frontier. Proxy: export-only plus P1. Promote if a setting strictly dominates baseline exporter settings on either bytes or `Δpq` without harming the other much. Kill criterion: dominated frontier. Runtime: about 0.05 H100-hours per setting. Bundle. ([GitHub][2])

### Phase 2: tokenizer first, then recipe on the winner

**E05 — Tokenizer audit bundle.** Hypothesis: current SP-1024 likely wastes some vocabulary mass or misses a better bytes/token tradeoff. Minimal code change: add `tokenizer_stats.py` and config-driven tokenizer paths. Expected sign: direction finding only. Proxy: tokenizer economics audit. Promote if any candidate looks plausibly non-dominated on bytes/token and fragmentation. Kill criterion: obviously dominated vocab choices. Runtime: near-zero GPU. Bundle.

**E06 — SP-512 P1.** Hypothesis: smaller vocab increases bytes seen per fixed token budget and reduces embedding burden enough to win after export. Minimal code change: tokenizer path and dataset path only. Expected sign: mildly positive or clearly negative, but worth knowing. Proxy: P1. Promote if `Δpq ≤ -0.003`, or if the candidate is within 0.001 of baseline while gaining meaningful artifact slack or effective train bytes per second. Kill if `Δpq ≥ +0.004` or step time and sequence fragmentation are both worse. Runtime: about 0.1 H100-hours. Singleton.

**E07 — SP-768 P1.** Hypothesis: this is the highest-probability tokenizer winner because it pushes below 1024 without going extreme. Minimal code change: tokenizer and dataset paths only. Expected sign: positive. Proxy: P1. Promote if it is the best tokenizer candidate or clears the same threshold as E06. Kill if dominated by SP-1024 or SP-512. Runtime: about 0.1 H100-hours. Singleton.

**E08 — SP-1536 P1.** Hypothesis: a modestly larger vocab may reduce bits/token enough to offset embedding cost. Minimal code change: tokenizer and dataset paths only. Expected sign: flat to slightly negative, but needed as an upper anchor. Proxy: P1. Promote only if it clearly wins on `Δpq`. Kill otherwise. Runtime: about 0.1 H100-hours. Singleton.

**E09 — Best tokenizer P2.** Hypothesis: the P1 winner survives longer training and exact export. Minimal code change: none beyond selected tokenizer. Expected sign: positive if the tokenizer track is real. Proxy: P2. Promote if it retains at least about a 0.004 post-roundtrip advantage or remains basically tied while opening obvious downstream budget. Kill if P2 ranking reverses. Runtime: about 0.3 H100-hours. Singleton.

**E10 — Tied-embedding learning-rate local star on tokenizer winner.** Hypothesis: in this small-vocab tied-embedding regime, the embedding matrix is unusually load-bearing, and the separate `TIED_EMBED_LR` knob matters more than generic learning-rate sweeps. Minimal code change: config only. Expected sign: positive. Proxy: P1. Promote if any point achieves `Δpq ≤ -0.003`. Kill criterion: no point beats the center. Runtime: about 0.3 H100-hours for a 3-point star. Bundle. ([GitHub][2])

**E11 — Matrix/scalar learning-rate star on tokenizer winner.** Hypothesis: a tiny local star around the baseline optimizer split is worthwhile, but a grid is not. Minimal code change: config only. Expected sign: small positive. Proxy: P1. Promote if the winner clears about 0.003 `Δpq`. Kill if the center remains best or if gains are below noise. Runtime: about 0.3 H100-hours for a tight 3-point star. Bundle.

**E12 — Embedding norm penalty A/B.** Hypothesis: explicit embedding norm discipline improves exporter robustness with minimal compute tax. Minimal code change: small auxiliary penalty or post-step clamp on `tok_emb`. Expected sign: neutral to slightly worse pre-quant, better post-quant. Proxy: P1 then P2 if it wins. Promote if `qgap` shrinks by at least 20 percent with no meaningful post-roundtrip regression, or if `Δpq ≤ -0.003`. Kill if pre-quant improves but post-quant does not. Runtime: about 0.1 H100-hours at P1. Singleton.

### Phase 3: export-aware training

**E13 — Clamp-aware row-outlier regularizer.** Hypothesis: because matrices are quantized row-wise with quantile clipping, explicitly discouraging heavy row tails will improve retained score. Minimal code change: one auxiliary loss on selected 2-D weights, gated by env flags. Expected sign: modest pre-quant cost, better post-quant retention. Proxy: P1, then P2 if it wins. Promote if `qgap` drops by at least 25 percent or `Δpq ≤ -0.003`. Kill if step time grows more than about 8 percent or retained score does not improve. Runtime: about 0.1 H100-hours at P1. Singleton. ([GitHub][2])

**E14 — QAT-lite on selected weights.** Hypothesis: fake-quantizing only the most important matrices during forward will improve export retention more than it costs in speed. Minimal code change: fake-quant wrappers for selected linear weights, disabled by default. Expected sign: positive post-roundtrip, negative throughput. Proxy: P1. Promote if `Δpq ≤ -0.004` and step slowdown is under about 12 percent. Kill if speed tax dominates or post-roundtrip quality does not improve. Runtime: about 0.12 H100-hours. Singleton.

**E15 — Best exporter setting plus best export-aware training trick.** Hypothesis: the exporter and training interventions compose. Minimal code change: none beyond combining E03/E04 with E13 or E14. Expected sign: positive. Proxy: P2. Promote if the combination beats the better parent by at least 0.002 `Δpq`. Kill if gains are not additive or if bytes headroom collapses. Runtime: about 0.3 H100-hours. Singleton.

### Phase 4: byte-efficient capacity trades

**E16 — KV-head rebudget.** Hypothesis: reducing `NUM_KV_HEADS` from 4 to 2 is a cheap stored-byte and possibly compute win that may not cost much quality. Minimal code change: config only. Expected sign: small positive or small negative, but very high EV because no new code. Proxy: P1 and P3 if it looks promising. Promote if `Δpq ≤ -0.003` or if it is nearly tied while obviously opening artifact or runtime budget. Kill if quality degrades materially. Runtime: about 0.1 H100-hours at P1. Singleton. ([GitHub][2])

**E17 — KV-head plus width rebudget.** Hypothesis: savings from fewer KV heads can be re-spent into slightly larger `MODEL_DIM` and beat the baseline frontier. Minimal code change: config only. Expected sign: higher variance positive. Proxy: P1, then P3 immediately if it looks good, then P2. Promote if the candidate clears about 0.005 `Δpq` at P2 and fits runtime. Kill if the speed or memory tax is ugly. Runtime: about 0.1 H100-hours at P1, 0.27 H100-hours at P3. Singleton.

**E18 — Grouped layer sharing moonshot.** Hypothesis: share parameters across small groups of layers and re-spend saved bytes into width or tokenizer slack. Minimal code change: moderate; new env-backed sharing map. Expected sign: high variance. Proxy: P1 only at first. Promote only if it either beats baseline outright or frees at least 300 kB while staying within about 0.003 `Δpq`. Kill fast otherwise. Runtime: about 0.1 H100-hours. Singleton.

### Phase 5: composition and promotion

**E19 — Composed finalist P2, seed A.** Hypothesis: the best singleton from tokenizer, the best recipe tweak, and the best export-aware tweak compose cleanly. Minimal code change: composition only. Expected sign: positive. Proxy: P2. Promote if it beats the best parent by at least 0.002 and clears about 0.006 `Δpq` versus matched baseline. Kill if interactions erase the gains. Runtime: about 0.3 H100-hours. Singleton.

**E20 — Composed finalist P2, seed B.** Hypothesis: the composed winner survives a second seed. Minimal code change: seed only. Expected sign: confirmation. Proxy: P2. Promote if the two-seed mean still clears the threshold and neither seed is worse than baseline. Kill if the result collapses. Runtime: about 0.3 H100-hours. Singleton.

**E21 — 8×H100 runtime rehearsal.** Hypothesis: the finalist genuinely fits under 600 seconds with acceptable memory and warmup behavior. Minimal code change: none. Expected sign: neutral. Proxy: P3. Promote if projected full-run time is at most about 540 seconds with a memory buffer, or at most about 570 seconds if the quality edge is exceptional. Kill if it is obviously too slow. Runtime: about 0.27 H100-hours. Singleton.

**E22 — One full 8×H100 candidate run.** Hypothesis: the finalist beats the public baseline on the actual objective. Minimal code change: none. Expected sign: positive. Proxy: full leaderboard-like run. Promote to a true multi-seed record attempt only if it wins on post-roundtrip `val_bpb` and is strong enough to justify satisfying the README's statistical evidence requirement. Kill if the cheap proxies did not transfer. Runtime: 1.33 H100-hours. Singleton. ([GitHub][1])

## 5. Statistics and decision rules

Use one seed when the question is "does this deserve to live another hour." That means exporter-only sweeps, P0, and most P1 screens are single-seed decisions. Use two seeds when the decision could change the next phase allocation. That means P2 finalists and any composed candidate. Use three or more full 8×H100 seeds only when you are close enough to a serious SOTA claim that the README's evidence requirement matters. New records must beat SOTA by at least 0.005 nats and provide enough logs for `p < 0.01`; tokenizer edits also get extra scrutiny on correct `val_bpb` accounting. ([GitHub][1])

One seed is enough to kill an idea if it is clearly bad. As a practical rule, any candidate that is worse than the matched baseline by 0.004 bpb or more at P1, or by 0.005 bpb or more at P2, dies immediately unless it buys extraordinary artifact slack. One seed is also enough to keep a deterministic exporter-only setting.

At cheap stages, care only about nontrivial effect sizes. The default screen thresholds should be: for P1, a post-roundtrip improvement of about 0.003 bpb, or a `qgap` reduction of at least 20 percent with neutral quality, or a clear throughput/artifact win that is plausibly monetizable later. For P2, the threshold should be about 0.005 bpb, or about 0.003 with a very strong side benefit. Anything smaller is too easy to fool yourself with in this regime.

Promotion from P0 to 1×H100 is easy: the run must complete, the parser must work, and the offline audit must not say the idea is obviously dominated. Promotion from 1×H100 P1 to P2 requires either a clear P1 win or a very plausible monetizable side benefit. Promotion from P2 to 8×H100 requires: two-seed mean `Δpq ≤ -0.006`, no single seed worse than the matched baseline, no worsening of `qgap` by more than about 0.002 bpb versus the matched baseline, a plausible artifact slack story, and no unresolved challenge-spirit ambiguity.

To avoid fooling yourselves with pre-quant gains, rank by post-roundtrip `Δpq` first, always. Track an **export retention fraction**, defined as post-roundtrip gain divided by pre-quant gain relative to the same baseline. Use it diagnostically, not as the primary ranker. A retention fraction below about 0.3 is a red flag; the public 4-hour run is already around 0.4, so anything worse than that needs a compelling excuse. ([GitHub][9])

For a practical posterior update rule, maintain one Gaussian belief per hypothesis family over `Δpq` at the current proxy level. Start every track at mean zero with wide variance. After each batch, update by inverse-variance weighting:
`μ' = (μ/σ² + d/v) / (1/σ² + 1/v)`,
`σ'² = 1 / (1/σ² + 1/v)`,
where `d` is the observed matched-baseline delta and `v` is the estimated observation variance from seed noise plus proxy residual. Then rank tracks by
`Pr(Δpq < -τ) × composability / expected_cost`,
with a multiplicative penalty for code growth and a harsher one for spirit ambiguity. This is rigorous enough to stop wishcasting and light enough to run in a spreadsheet.

## 6. Compute-budget plan

I would deliberately underspend the nominal budget. In this challenge, the main risk is not running out of cloud dollars. It is burning attention on noisy, weakly identified ideas. The README itself recommends iterating on cheaper SKUs before touching 8×H100, and your current scaffold budget default of 5000 is far looser than the operating discipline you asked for. ([GitHub][1])

The **conservative plan** is about 12–15 H100-hours total before any serious 8-GPU campaign. Allocate roughly 2 H100-hours to trust and calibration, 4 to tokenizer and exporter screening, 4 to recipe and export-aware training, 2 to architecture re-budget tests, and reserve about 2 H100-hours for one P3 rehearsal plus one full 8×H100 trial. In ordinary cloud pricing this is comfortably below your stated cash cap. More importantly, it forces the funnel to stay narrow.

The **aggressive plan** is about 25–35 H100-hours. Allocate roughly 3 H100-hours to trust and calibration, 8 to the tokenizer/export frontier, 10 to recipe and export-aware work with extra replications, 8 to one architecture moonshot branch, and the same rehearsal plus full 8×H100 stage at the end. This is still not a huge cloud bill, but it is enough to tempt the team into overexplaining noise.

My recommendation is to use the conservative plan and leave the remaining budget uncommitted. Only unlock the aggressive tranche if one of two things happens: either one track's posterior probability of clearing the P2 promotion threshold rises above about 0.7, or two orthogonal singleton wins appear to compose without obvious interference. Budget should be treated as an option, not a target.

Operationally, also change the scaffold so each phase has a hard cap. Set `budget_total` to the real ceiling, then add per-phase ceilings in config metadata or manifest policy, for example: Phase 0 at 10 percent, Phase 1 at 25 percent, Phase 2 at 35 percent, Phase 3 at 20 percent, and Phase 4 at 10 percent reserved.

## 7. Implementation roadmap

Do the engineering in this order.

First, make the harness support the experiment funnel you actually want. Add `local` and `1xh100` machine entries, because the current machine file only knows about 8-GPU hosts. Then fix `run.sh` and `preflight.py` so dataset and tokenizer paths are config-driven rather than hardcoded to SP-1024. Keep the current defaults for baseline reproduction, but stop overwriting explicit config values. ([GitHub][8])

Second, improve result capture. `run.sh` should copy back `final_model.pt`, `train_gpt.py`, and a tiny environment manifest in addition to the int8 artifact and log. The trainer already writes `final_model.pt`; the harness just is not collecting it. ([GitHub][2])

Third, extend the telemetry and parser. Add `tok_s`, `train_tokens_seen`, `final_prequant_exact`, quantization payload bytes, and payload ratio to the trainer logs. Then extend `parse_log.py` to compute `qgap`, artifact slack, total train tokens seen, config diff versus baseline, payload bytes, payload ratio, and proxy-relative deltas. Because `config_utils.py` discovers allowable env vars by regexing `os.environ.get(...)` in the trainer, every new experiment knob should be introduced as an env-backed trainer flag; that way the harness accepts it automatically. ([GitHub][10])

Fourth, add a cheap exporter-only utility under `experiments/scripts`. Do not put this in the final submission path. This tool should load `final_model.pt`, apply arbitrary exporter settings, produce the compressed artifact, and run exact roundtrip validation. That makes E03 and E04 nearly free and keeps experimental scaffolding out of the code-budgeted submission script.

Fifth, add proxy profiles. Create config snippets or named presets for `p0_smoke`, `p1_fast`, `p2_medium`, and `p3_runtime`. The important thing is consistency, not elegance. Every run should state its proxy level in metadata.

Sixth, add promotion metadata. The current metadata fields are `hypothesis_id`, `group`, and `notes`. Extend them with `phase`, `proxy_level`, `parent_run_id`, `baseline_run_id`, `promotion_rule`, `kill_rule`, and `decision`. The parser already merges manifest metadata; this is an easy extension. ([GitHub][10])

Seventh, improve the views. Keep the existing `launch.py` orchestration, `status`, `watch`, `collect`, `budget`, and results folders. Add one derived leaderboard view and one decision view. The leaderboard view ranks by final post-roundtrip `val_bpb`. The decision view ranks by `Δpq`, `qgap`, step time, artifact slack, and posterior score. Do not bury the score-critical columns. ([GitHub][11])

Every experimental model change should be isolated behind flags. I would use names like `OUTLIER_REG_COEF`, `OUTLIER_REG_TARGETS`, `FAKE_QUANT_TARGETS`, `FAKE_QUANT_MODE`, `INT8_CLIP_PERCENTILE`, `INT8_KEEP_FLOAT_MAX_NUMEL`, `KV_REBUDGET_MODE`, `LAYER_SHARE_MAP`, `EMBED_NORM_TARGET`, and `EMBED_NORM_COEF`. Instrumentation flags are optional; core experiment knobs must be explicit.

The recommended result schema, whether CSV or JSONL, is:

`run_id, parent_run_id, phase, proxy_level, hypothesis_id, group, decision, seed, host, trainer_git_sha, config_hash, baseline_run_id, vocab_size, tokenizer_path, data_path, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult, tie_embeddings, train_batch_tokens, train_seq_len, wallclock_cap_s, model_params, stop_step, train_tokens_seen, step_avg_ms, tok_s, prequant_val_loss, prequant_val_bpb, postquant_val_loss, postquant_val_bpb, qgap_bpb, qgap_loss, model_bytes_raw, model_bytes_int8_zlib, code_bytes, total_submission_bytes, artifact_slack_bytes, payload_bytes, payload_ratio, peak_memory_allocated_mib, peak_memory_reserved_mib, delta_postquant_bpb_vs_baseline, delta_prequant_bpb_vs_baseline, delta_step_avg_vs_baseline, delta_artifact_bytes_vs_baseline, spirit_risk, notes`.

The recommended folder structure is:

`experiments/configs/{phase}_{track}_{name}.yaml`
`experiments/results/{yyyy-mm-dd}_{phase}_{track}_{run_id}/` with `config.yaml`, `manifest.json`, `train.log`, `metrics.json`, `trainer_snapshot.py`, `env.json`, `final_model.pt`, and `final_model.int8.ptz`
`experiments/analysis/results.csv` and `experiments/analysis/decision_view.csv`
`experiments/tokenizers/{variant_name}/` for tokenizer models and stats.

The run notebook template should be very short:

```text
Run ID:
Date:
Owner:
Phase / Proxy:
Hypothesis:
Matched baseline:
Exact config diff:
Expected mechanism:
Primary failure mode:
Observed pre-quant:
Observed post-quant:
qgap:
Artifact bytes / slack:
Step time / tok_s / memory:
Decision: promote / hold / kill
Next action:
```

The weekly review template should also be short:

```text
Week:
Spend to date:
Calibration health: which proxies are actually predicting P2 / 8x?
Top 3 tracks by posterior EV/$:
Top 3 kills and why:
Any spirit-risk items needing organizer clarification:
What gets promoted this week:
What is explicitly frozen this week:
```

## 8. Final recommendation

The top five experiments to run first are these.

First, run the exact baseline reproduction on the scaffold and patch the telemetry until it is trustworthy. Without this, every later delta is theater. ([GitHub][4])

Second, do the exporter-only clip-percentile and keep-float sweeps. They are nearly free and directly target the measured place where the public longer run is bleeding away gains. ([GitHub][2])

Third, do the tokenizer audit and a very small tokenizer shortlist, specifically 512, 768, and 1536, with P1 matched runs. That is the highest-probability structural lever after exporter tuning.

Fourth, on the tokenizer winner, do a tiny tied-embedding learning-rate star, not a generic grid. The trainer already exposes that knob for a reason. ([GitHub][2])

Fifth, test one export-aware training intervention, preferably clamp-aware outlier regularization first and QAT-lite second. The 4-hour run already told you that export retention is the right battlefield. ([GitHub][12])

The top five things to avoid at the beginning are these.

Do not run the scaffold's 15-config matrix/scalar learning-rate grid. It is exactly the wrong shape of search for week one. ([GitHub][3])

Do not start with full byte-level modeling.

Do not start with Mamba, Hyena, RWKV, RetNet, or any other architecture-family rewrite.

Do not touch distillation, offline teacher labels, or any other external-compute-adjacent trick until you have organizer clarity, because the README explicitly leaves room for spirit-based disqualification. ([GitHub][1])

Do not brute-force seeds or do giant random sweeps. The FAQ calls that out directly as the kind of thing that can get disallowed. ([GitHub][1])

The best fast path to a serious submission is: baseline trust, exporter-only tuning, tokenizer shortlist, embedding/recipe tuning on the tokenizer winner, one export-aware training trick, then compose the best singleton from each of those tracks and verify it on P2 with two seeds before touching 8×H100.

The best higher-upside but riskier path is: take that same clean path first, and only after it plateaus try one byte-efficient capacity trade such as KV-head rebudget with slight width re-spend, or a very modest grouped layer-sharing experiment. Those are the only moonshots I would allow before the endgame, because they at least have a first-principles connection to the stored-byte constraint.

Rent the 8×H100 box only when this exact condition is met: **a candidate has two P2 seeds whose mean post-roundtrip improvement is at least 0.006 bpb over the matched baseline, no seed is worse than baseline, the candidate's `qgap` is not worse than the matched baseline by more than about 0.002 bpb, the projected 8-GPU runtime after a P3 rehearsal is at most about 540 seconds with memory headroom, and there is no unresolved spirit-risk question**. Once one full 8-GPU run confirms a serious win, then and only then schedule the extra full runs needed to satisfy the README's statistical submission bar. ([GitHub][1])

That is the funnel I would run. It is narrow on purpose.

[1]: https://github.com/openai/parameter-golf/blob/main/README.md "https://github.com/openai/parameter-golf/blob/main/README.md"
[2]: https://raw.githubusercontent.com/openai/parameter-golf/main/train_gpt.py "https://raw.githubusercontent.com/openai/parameter-golf/main/train_gpt.py"
[3]: https://raw.githubusercontent.com/seconds-0/parameter-golf/main/experiments/configs/sweep_lr.yaml "https://raw.githubusercontent.com/seconds-0/parameter-golf/main/experiments/configs/sweep_lr.yaml"
[4]: https://raw.githubusercontent.com/seconds-0/parameter-golf/main/experiments/configs/baseline.yaml "https://raw.githubusercontent.com/seconds-0/parameter-golf/main/experiments/configs/baseline.yaml"
[5]: https://github.com/seconds-0/parameter-golf/blob/main/experiments/scripts/run.sh "https://github.com/seconds-0/parameter-golf/blob/main/experiments/scripts/run.sh"
[6]: https://github.com/seconds-0/parameter-golf/blob/main/experiments/scripts/parse_log.py "https://github.com/seconds-0/parameter-golf/blob/main/experiments/scripts/parse_log.py"
[7]: https://github.com/seconds-0/parameter-golf/blob/main/experiments/scripts/compare.py "https://github.com/seconds-0/parameter-golf/blob/main/experiments/scripts/compare.py"
[8]: https://raw.githubusercontent.com/seconds-0/parameter-golf/main/experiments/machines.yaml "https://raw.githubusercontent.com/seconds-0/parameter-golf/main/experiments/machines.yaml"
[9]: https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md "https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md"
[10]: https://github.com/seconds-0/parameter-golf/blob/main/experiments/scripts/config_utils.py "https://github.com/seconds-0/parameter-golf/blob/main/experiments/scripts/config_utils.py"
[11]: https://github.com/seconds-0/parameter-golf/blob/main/experiments/scripts/launch.py "https://github.com/seconds-0/parameter-golf/blob/main/experiments/scripts/launch.py"
[12]: https://github.com/openai/parameter-golf/blob/main/records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md "https://github.com/openai/parameter-golf/blob/main/records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md"
