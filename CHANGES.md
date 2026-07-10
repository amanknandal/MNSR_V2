# What changed and why

## Speed

Previously, one full dataset run computed the baseline (BaselineCoT) six
separate times: once in the main experiment, and once again inside every one
of the five ablation configs, because `AblationStudy` built a brand new
`MNSRExperiment` (with its own fresh `Phi3Mini` client) for each config. It
also recomputed the MNSR pipeline's first reasoning pass from scratch every
time, even when that pass is identical across configs (temperature 0, same
prompt).

`models/phi3.py` now caches every temperature-0 call to Ollama, keyed by
`(model, prompt)`. `run.py` creates one `Phi3Mini` client and one cache dict
per dataset, and passes that same client into the main experiment and into
`AblationStudy`, which passes it into every ablation config's
`MNSRExperiment`. The result: each unique question is sent to Ollama once for
its initial reasoning pass, self-evaluation, etc., no matter how many
ablation configs reuse it. Only the parts of the pipeline that actually
differ between configs (correction passes triggered by the real controller)
result in new calls.

`ThreadPoolExecutor` concurrency now defaults to `max_workers=1`
(`MAX_WORKERS` in `run.py`). A single local Ollama instance generally
processes one request at a time anyway, so concurrent threads were mostly
adding request queuing and, more importantly, a source of run-to-run
nondeterminism (see below) without a real speed benefit.

## Correctness bugs fixed

1. **Ablation configs were producing accuracy identical to Full MNSR even
   when they should degrade.** With `MockController` or `MockValidator`
   installed, the pipeline should fall back to an uncorrected first-pass
   answer (matching baseline behavior), not match Full MNSR's corrected
   accuracy. Root cause: repeated independent calls to the model at
   temperature 0 were not guaranteed to return identical text on a local
   Ollama server under concurrent load, so "identical" computations produced
   different answers across runs. Caching the deterministic calls (above)
   removes this source of noise, so ablations that genuinely disable
   correction will now reliably show baseline-level accuracy, not an
   inflated number.

2. **`correction_success_rate` in `mnsr/metrics.py` counted a question as a
   correction "success" whenever MNSR was correct after a correction pass,
   even if the baseline was already correct on that question too** (i.e. no
   correction was actually needed to get the right answer). It now only
   counts a success when MNSR is correct **and** baseline was wrong,
   matching what "correction success" should mean.

## Not changed here

The `SymbolicValidator` correctness-vs-formatting issue, and whether
`SELF_VERIFY` should be a mandatory first pass, are unchanged in this drop —
those are modeling decisions, not the speed/consistency bugs asked about
here. Happy to make those changes next if you want them in this same
codebase.

## Comparison baselines added (Self-Consistency, Reflexion)

`evaluation/comparison_baselines.py` adds two new reasoners that share the
same `Phi3Mini` client/cache as everything else:

- `SelfConsistencyReasoner`: samples `num_paths` reasoning trajectories at
  temperature 0.7 and takes a majority vote over the extracted answers, per
  Wang et al.'s Self-Consistency method. These calls are intentionally not
  cached (temperature > 0, diversity is the point).
- `ReflexionReasoner`: generates an initial answer, self-evaluates its own
  confidence, and if confidence is below threshold, critiques and revises
  itself for up to `max_iterations` rounds, stopping early once its
  self-critique reports no issues or confidence clears the threshold.

`evaluation/experiment.py` now accepts an `extra_methods` dict and runs each
one per question alongside baseline CoT and MNSR, logging
`{method}_answer`, `{method}_correct`, and `{method}_latency_sec` into each
result entry.

`evaluation/comparison_report.py` reads `results.json`, builds an accuracy +
latency table across every method present, and runs McNemar's test of MNSR
against each individual method (not just against baseline), so you get a
real significance number for "MNSR beats Self-Consistency" and "MNSR beats
Reflexion", not just "MNSR beats plain CoT."

`run.py` wires both baselines in by default (`RUN_COMPARISON_BASELINES =
True`) and produces `comparison_report.json` plus a
`figure_method_comparison.png` bar chart alongside your existing figures.
Set `RUN_COMPARISON_BASELINES = False` if you want a faster run without
them (Self-Consistency in particular costs ~5x the LLM calls per question
since it isn't cacheable at temperature 0.7).

This still doesn't include a symbolic/tool-augmented neurosymbolic baseline
(e.g. program-of-thought) — worth adding next if you want to defend the
"neurosymbolic" label specifically, since right now the comparison set
covers self-correction methods but not execution-based symbolic reasoning.

## Gold-answer comma bug fixed

Tested against the actual uploaded 100-question GSM8K-format dataset.
3 of the 100 gold answers use comma-formatted numbers (`114,200`,
`276,000`, `5,600`). `is_correct()`'s numeric-match regex
(`-?\d+\.?\d*`) doesn't allow commas, so it silently fell through to a
substring match that fails for a correctly-formatted model answer like
`114200` against gold `114,200` — a guaranteed false negative on those 3
questions regardless of model quality. `normalize_answer()` in
`evaluation/experiment.py` now strips thousands-separator commas between
digits before comparison. Verified against all 100 questions: zero
non-numeric gold values remain after normalization, and the comma casesma cases
now match correctly while unrelated wrong-answer cases still fail as
expected.

Your dataset is included at `datasets/gsm8k.json` in this drop (named so
the pipeline's filename-based dataset-type detection correctly tags it as
`numeric`, which drives the `SymbolicValidator`'s dataset-specific format
check).
