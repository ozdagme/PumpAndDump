# Code-only reproducibility package

This folder contains source code for the manuscript's model and selected
reproducibility analyses. It intentionally contains code only: no dataset,
forum post, account identifier, source-site name or URL, firm/event mapping,
generated result, prediction file, log, cache, or model checkpoint is included.

## Contents

- `v4_pump_and_dump.py` — model, preprocessing, baseline, and main experiment
  implementation.
- `v5_pump_and_dump.py` — interruption-resilient runner and local caches.
- `full_v4_runner.py` — entry point for the main full-sample run.
- `activity_controlled_q1_runner.py` — activity-controlled experiment entry
  point.
- `temporal_ablation_q1_runner.py` — temporal ablation entry point.
- `q1_posthoc_analysis.py` — post-hoc metrics and statistical analyses.
- `q1_completion_analyses.py` — training-loss and two-way bootstrap analyses.
- `matched_negative_event_analysis.py` — matched event-window analysis.

## Inputs and execution

The experiment code expects locally supplied forum and market CSV files named
`forum_verisi_v4.csv` and `piyasa_verisi_v4.csv` by default. These files are not
part of this release. Their required columns and validation rules are defined
in `prepare_forum` and `prepare_market` in `v4_pump_and_dump.py`. Obtain and use
any inputs only where you have the applicable rights and permissions. Do not
commit input files or generated outputs to a public repository.

Install the packages listed in `requirements.txt` in an existing Python
environment. A CUDA-capable GPU is optional but materially reduces model
runtime. The original computational environment used Python 3.12, PyTorch,
Transformers, Sentence-Transformers, NumPy, pandas, and scikit-learn; package
versions may need adjustment for the local hardware and CUDA installation.

From this directory, run the main experiment with:

```bash
python full_v4_runner.py
```

The other experiment entry points are listed above. The statistical-analysis
scripts consume the corresponding locally generated prediction/financial
outputs. They do not bundle those outputs.

## Scope and privacy notes

The code uses pseudonymous firm codes supplied in local inputs; no real-ticker
reverse mapping is included. Event-specific exclusion profiles that require a
private firm/event mapping are disabled and omitted from this release. The
package does not contain a forum collection script or information identifying
the forum website. It is not a data release, model-weight release, or public
inference service.

The software license has not yet been selected. Do not assume permission to
redistribute or reuse the code beyond applicable law and the authors' eventual
license choice.