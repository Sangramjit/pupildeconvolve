# pupildeconvolve

Dual-kernel pupil deconvolution for estimating attentional pulse dynamics from pupil-size recordings.

---

# Overview

`pupildeconvolve` is a Python package for modeling pupil responses using a dual-kernel deconvolution framework inspired by pupil deconvolution approaches in cognitive neuroscience.

The package estimates:

* attentional pulse amplitudes
* slow drift/slope components
* participant-level attentional profiles


The package is designed primarily for:

* cognitive neuroscience
* attention research
* emotional face processing
* attentional blink experiments
* Faced Paced Stimuli Presenttaion like RSVP, Video
* pupillometry experiments

---

# Key Features

* Dual-kernel pupil modeling
* Participant-wise averaging before fitting
* Attention pulse estimation
* PLR (pupillary light reflex) modeling
* Drift/slope estimation
* Confidence interval plotting
* Multi-participant support
* DataFrame-based workflow
* Synthetic data simulation support
* Compatible with DataMatrix workflows

---

# Installation

## Local editable installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/pupildeconvolve.git
```

Move into the repository:

```bash
cd pupildeconvolve
```

Install locally:

```bash
pip install -e .
```

The `-e` flag installs the package in editable mode.

This means:

* changes in source files update immediately
* no reinstall required after edits
* useful during development

---

# Dependencies

The package depends on:

* numpy
* pandas
* scipy
* matplotlib
* tqdm

These are installed automatically.

---

# Basic Workflow

The recommended workflow is:

```text
Trials
→ Participant average
→ Dual-kernel fitting
→ Attention pulse estimation
→ Group statistics/plotting
```

The package intentionally performs:

```text
average first
fit second
```

instead of:

```text
fit each trial separately
average afterward
```

because participant-level averaging produces:

* more stable fits
* reduced noise
* biologically plausible pulse profiles
* better late-pulse estimation

---

# Quick Start

## Step 1 — Import

```python
import numpy as np
import pandas as pd

from pupildeconvolve import (
    fit_dataframe,
    plot_pupil_and_pulses
)
```

---

## Step 2 — Create a DataFrame

The package expects:

* one row = one trial
* timepoints as columns
* participant column

Example:

```python
rows = []

for trial in range(20):

    row = {}

    row["participant"] = "P1"

    for t in range(200):
        row[t] = np.random.randn()

    rows.append(row)


df = pd.DataFrame(rows)
```

Define pupil columns:

```python
pupil_cols = list(range(200))
```

---

## Step 3 — Fit the model

```python
res_df = fit_dataframe(

    df,

    pupil_cols=pupil_cols,

    participant_col="participant",

    sampling_rate=100,

    pulse_interval=150,

    n_runs=10
)
```

---

## Step 4 — View results

```python
print(res_df.head())
```

Output columns:

| Column          | Description                  |
| --------------- | ---------------------------- |
| participant     | participant ID               |
| slope           | slow drift estimate          |
| plr_latency_ms  | estimated PLR latency        |
| Pulse1...PulseN | attentional pulse amplitudes |

---

# Plotting

## Example

```python
plot_pupil_and_pulses(

    pupil=pupil_matrix,

    result={
        "attention_amplitude": attention_matrix,
        "attention_time": attention_time
    },

    time=time
)
```

The plotting function computes:

* grand mean
* confidence intervals
* participant-level averaging

for both:

* pupil traces
* attentional pulses

---

# DataFrame API

## fit_dataframe()

```python
fit_dataframe(
    df,
    pupil_cols,
    participant_col,
    condition_col=None,
    sampling_rate=100,
    pulse_interval=150,
    pulse_times=None,
    save_csv=False,
    output_path="results.csv",
    show_progress=True,
    verbose=True,
    n_runs=1
)
```

---

## Parameters

### df

Pandas DataFrame containing trial-wise pupil recordings.

---

### pupil_cols

List of columns containing pupil timepoints.

Example:

```python
pupil_cols = list(range(243))
```

---

### participant_col

Column containing participant IDs.

Example:

```python
participant_col="participant"
```

---

### condition_col

Optional condition column.

Example:

```python
condition_col="Condition"
```

If provided:

* fitting occurs separately for each condition
* output contains one row per participant × condition

---

### sampling_rate

Sampling rate in Hz.

Example:

```python
sampling_rate=100
```

---

### pulse_interval

Spacing between attentional pulses in milliseconds.

Example:

```python
pulse_interval=150
```

---

### pulse_times

Optional custom pulse times.

Example:

```python
pulse_times=[0,100,400,700]
```

If omitted:

regular pulse spacing is generated automatically.

---

### n_runs

Number of optimization restarts.

Higher values:

* improve stability
* increase runtime

Recommended:

| Dataset size        | Recommended n_runs |
| ------------------- | ------------------ |
| small               | 5–10               |
| medium              | 10–25              |
| publication-quality | 25–50              |

---

# Core Model

The model contains:

## 1. PLR kernel

Models early constriction dynamics.

---

## 2. Attention kernel

Models slower attentional recruitment.

---

## 3. Drift/slope term

Models slow pupil drift.

---

# Optimization

The package uses:

```text
L-BFGS-B optimization
```

with:

* bounded optimization
* multiple restarts
* participant-level averaging

---

# Important Modeling Choice

The package intentionally uses:

```text
participant-average fitting
```

instead of:

```text
trial-wise fitting
```

because trial-wise deconvolution produced:

* unstable late pulses
* noisy pulse estimates
* unrealistic optimization collapse
* poor identifiability

Participant averaging substantially improves:

* robustness
* physiological interpretability
* stability of attentional profiles

---

# Handling Different Recording Lengths

Participants may have:

* different recording durations
* different trial lengths
* missing tails

The package handles this automatically using:

```text
NaN padding
```

and:

```python
np.nanmean()
```

during participant averaging.

Missing segments are not treated as zero.

---

# Working with DataMatrix

Example:

```python
from datamatrix import io
```

Load:

```python
dm = io.readbin("participant.dm")
```

Extract pupil matrix:

```python
pupil = np.asarray(dm["ptrace_RSVP"], dtype=float)
```

Convert to DataFrame:

```python
rows = []

for i in range(len(dm)):

    row = {}

    row["participant"] = "P1"

    for t in range(pupil.shape[1]):
        row[t] = pupil[i, t]

    rows.append(row)


df = pd.DataFrame(rows)
```

Then run:

```python
fit_dataframe(...)
```

---

# Example Scripts

The repository contains:

| File                    | Description               |
| ----------------------- | ------------------------- |
| examples/basic_usage.py | synthetic dataset example |
| examples/dm_example.py  | DataMatrix workflow       |
| tests/test_api.py       | API tests                 |

---

# Running Tests

Install pytest:

```bash
pip install pytest
```

Run:

```bash
pytest
```

---

# Example Output

Typical outputs include:

* participant-wise pulse amplitudes
* condition-level attentional dynamics
* grand-average pupil traces
* confidence intervals

---

# Citation

If you use this package in research, please cite:

```text
Maity, S. (2026).
pupildeconvolve: Dual-kernel pupil deconvolution for attentional pulse estimation.
```

---

# License

MIT License.

---

# Author

Sangramjit Maity

Cognitive Neuroscience / Computational Modeling
