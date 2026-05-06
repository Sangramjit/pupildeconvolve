import os

import numpy as np
import pandas as pd

from datamatrix import io

from pupildeconvolve import (
    fit_dataframe,
    plot_pupil_and_pulses
)

# =========================================================
# SETTINGS
# =========================================================

# ---------------------------------------------------------
# ADD MULTIPLE DM FILES HERE
# ---------------------------------------------------------
dm_paths = [

    r"C:\Users\sangr\Downloads\P1_dm_preprocessed.dm",

    r"C:\Users\sangr\Downloads\P2_dm_preprocessed.dm",

    r"C:\Users\sangr\Downloads\P3_dm_preprocessed.dm",

    r"C:\Users\sangr\Downloads\P4_dm_preprocessed.dm",

    r"C:\Users\sangr\Downloads\P5_dm_preprocessed.dm",

    r"C:\Users\sangr\Downloads\P6_dm_preprocessed.dm",
]

pupil_col = "ptrace_RSVP"

optional_meta_cols = [

    "pair",

    "T1",

    "T2",

    "lag",

    "two_targets",
]

sampling_rate = 100

pulse_interval = 150

# =========================================================
# STORAGE
# =========================================================
all_rows = []

# IMPORTANT:
# global maximum length
max_timepoints = 0

# =========================================================
# FIRST PASS
# FIND MAXIMUM LENGTH
# =========================================================
print("\n=================================================")
print("Scanning DataMatrix files...")

for dm_path in dm_paths:

    dm = io.readbin(dm_path)

    pupil = np.asarray(
        dm[pupil_col],
        dtype=float
    )

    if pupil.ndim != 2:

        raise ValueError(
            f"{pupil_col} must be "
            f"trial x time"
        )

    _, n_tp = pupil.shape

    max_timepoints = max(
        max_timepoints,
        n_tp
    )

print(
    "\nMaximum timepoints detected:",
    max_timepoints
)

# =========================================================
# SECOND PASS
# BUILD GRAND DATAFRAME
# =========================================================
for dm_path in dm_paths:

    print("\n=================================================")
    print(f"Loading: {dm_path}")

    dm = io.readbin(dm_path)

    print("Length:", len(dm))

    # -----------------------------------------------------
    # participant ID FROM FILENAME
    # -----------------------------------------------------
    pid = os.path.basename(
        dm_path
    ).split(
        "_dm_preprocessed.dm"
    )[0]

    print("Participant ID:", pid)

    # -----------------------------------------------------
    # extract pupil
    # -----------------------------------------------------
    pupil = np.asarray(

        dm[pupil_col],

        dtype=float
    )

    if pupil.ndim != 2:

        raise ValueError(
            f"{pupil_col} must be "
            f"trial x time. "
            f"Found: {pupil.shape}"
        )

    n_trials, n_tp = pupil.shape

    print("Pupil shape:", pupil.shape)

    # =====================================================
    # BUILD ROWS
    # =====================================================
    for i, row in enumerate(dm):

        row_dict = {}

        # -------------------------------------------------
        # pupil columns
        # -------------------------------------------------
        for t in range(max_timepoints):

            # ---------------------------------------------
            # shorter recordings
            # ---------------------------------------------
            if t >= n_tp:

                row_dict[t] = np.nan

            else:

                val = pupil[i, t]

                row_dict[t] = val

        # -------------------------------------------------
        # participant
        # -------------------------------------------------
        row_dict["participant"] = pid

        # -------------------------------------------------
        # trial
        # -------------------------------------------------
        row_dict["trial"] = i

        # -------------------------------------------------
        # optional metadata
        # -------------------------------------------------
        for col in optional_meta_cols:

            if col in dm.column_names:

                row_dict[col] = getattr(
                    row,
                    col
                )

        all_rows.append(
            row_dict
        )

# =========================================================
# CREATE GRAND DATAFRAME
# =========================================================
print("\n=================================================")
print("Creating GRAND dataframe...")

df = pd.DataFrame(
    all_rows
)

meta_cols = [

    c for c in
    ["participant", "trial"] +
    optional_meta_cols

    if c in df.columns
]

pupil_cols = list(
    range(max_timepoints)
)

df = df[
    meta_cols + pupil_cols
]

print("\nGRAND DataFrame shape:")
print(df.shape)

print("\nUnique participants:")
print(df["participant"].unique())

print(
    "\nTotal participants:",
    df["participant"].nunique()
)

print("\nFirst rows:")
print(df.head())

# =========================================================
# RUN MODEL
# =========================================================
print("\n=================================================")
print("Running pupildeconvolve...")

res_df = fit_dataframe(

    df,

    pupil_cols=pupil_cols,

    participant_col="participant",

    sampling_rate=sampling_rate,

    pulse_interval=pulse_interval,

    save_csv=True,

    output_path="grand_attention_output.csv",

    verbose=True,

    show_progress=True,

    n_runs=50
)

# =========================================================
# OUTPUT
# =========================================================
print("\n=================================================")
print("RESULT")

print(res_df.head())

print("\nResult shape:")
print(res_df.shape)

# =========================================================
# TIME AXIS
# =========================================================
dt = 1000 / sampling_rate

time = np.arange(

    0,

    max_timepoints * dt,

    dt
)

# =========================================================
# ATTENTION PULSES
# =========================================================
pulse_cols = [

    c for c in res_df.columns

    if str(c).startswith(
        "Pulse"
    )
]

pulse_times = np.arange(

    0,

    len(pulse_cols) * pulse_interval,

    pulse_interval
)

attention_time = np.searchsorted(
    time,
    pulse_times
)

# =========================================================
# GRAND AVERAGE PUPIL
# =========================================================
print("\n=================================================")
print("Computing grand averages...")

participant_pupil_means = []

# ---------------------------------------------------------
# compute participant averages
# ---------------------------------------------------------
for pid in df["participant"].unique():

    df_pid = df[
        df["participant"] == pid
    ]

    pupil_matrix = (
        df_pid[pupil_cols]
        .astype(float)
        .values
    )

    # participant mean
    pupil_avg = np.nanmean(

        pupil_matrix,

        axis=0
    )

    participant_pupil_means.append(
        pupil_avg
    )

# ---------------------------------------------------------
# participants x time
# ---------------------------------------------------------
grand_pupil_matrix = np.vstack(
    participant_pupil_means
)

print(
    "\nGrand pupil matrix shape:",
    grand_pupil_matrix.shape
)

# =========================================================
# ATTENTION MATRIX
# =========================================================
attention_matrix = (
    res_df[pulse_cols]
    .astype(float)
    .values
)

print(
    "Attention matrix shape:",
    attention_matrix.shape
)

# =========================================================
# GRAND AVERAGE PLOT
# =========================================================
print("\n=================================================")
print("Plotting grand averages...")

plot_pupil_and_pulses(

    pupil=grand_pupil_matrix,

    result={

        "attention_amplitude":
        attention_matrix,

        "attention_time":
        attention_time
    },

    time=time,

    title=(
        "Grand Average "
        "Pupil + Attention"
    ),

    show_individual=False
)

print("\nDONE")