import numpy as np
import pandas as pd

from pupildeconvolve import (
    fit_dataframe,
    plot_pupil_and_pulses
)

# =====================================================
# SETTINGS
# =====================================================
sampling_rate = 100

dt = 1000 / sampling_rate

time = np.arange(
    0,
    3000,
    dt
)

n_participants = 10

n_trials_per_condition = 20

conditions = [
    "Happy",
    "Angry"
]

pulse_interval = 150

# =====================================================
# GAMMA KERNEL
# =====================================================
def gamma_kernel(
    t,
    n,
    t_max
):

    h = (
        t ** n
    ) * np.exp(
        -n * t / t_max
    )

    h[0] = 0

    return h / np.max(h)

kernel_t = np.arange(
    len(time)
) * dt

# =====================================================
# BASE KERNELS
# =====================================================
base_plr = gamma_kernel(

    kernel_t,

    n=6,

    t_max=350
)

base_attn = gamma_kernel(

    kernel_t,

    n=10,

    t_max=900
)

# =====================================================
# PULSE TIMES
# =====================================================
pulse_times = np.arange(

    0,

    3000,

    pulse_interval
)

# =====================================================
# SYNTHETIC DATAFRAME
# =====================================================
rows = []

print("\nGenerating synthetic dataset...")

for pid in range(
    n_participants
):

    participant_id = (
        f"P{pid+1}"
    )

    # -------------------------------------------------
    # participant variability
    # -------------------------------------------------
    participant_gain = np.random.uniform(
        0.8,
        1.2
    )

    participant_noise = np.random.uniform(
        0.01,
        0.04
    )

    for cond in conditions:

        # =============================================
        # CONDITION DIFFERENCE
        # =============================================
        if cond == "Happy":

            attention_profile = np.array([

                0.2,
                0.5,
                0.8,
                1.1,
                1.4,
                1.6,
                1.7,
                1.6,
                1.4,
                1.2,
                1.0,
                0.8,
                0.6,
                0.4,
                0.3,
                0.2,
                0.1,
                0.05,
                0.02,
                0.01
            ])

            plr_scale = -0.5

        else:

            attention_profile = np.array([

                0.3,
                0.7,
                1.1,
                1.5,
                1.9,
                2.2,
                2.4,
                2.3,
                2.1,
                1.8,
                1.5,
                1.2,
                0.9,
                0.7,
                0.5,
                0.3,
                0.2,
                0.1,
                0.05,
                0.02
            ])

            plr_scale = -0.7

        for trial in range(
            n_trials_per_condition
        ):

            # -------------------------------------------------
            # pulse train
            # -------------------------------------------------
            stim = np.zeros_like(
                time
            )

            for j, pt in enumerate(
                pulse_times
            ):

                idx = np.argmin(
                    np.abs(time - pt)
                )

                amp = (
                    attention_profile[j]
                    * participant_gain
                    * np.random.uniform(
                        0.8,
                        1.2
                    )
                )

                stim[idx] = amp

            # -------------------------------------------------
            # kernels
            # -------------------------------------------------
            plr_kernel = (
                plr_scale *
                base_plr
            )

            attn_kernel = (
                1.0 *
                base_attn
            )

            biphasic_kernel = (
                plr_kernel +
                attn_kernel
            )

            # -------------------------------------------------
            # generate pupil
            # -------------------------------------------------
            pupil_clean = np.convolve(

                stim,

                biphasic_kernel,

                mode="full"

            )[:len(time)]

            # -------------------------------------------------
            # drift
            # -------------------------------------------------
            drift = np.linspace(

                0,

                np.random.uniform(
                    -0.2,
                    0.2
                ),

                len(time)
            )

            # -------------------------------------------------
            # noise
            # -------------------------------------------------
            noise = np.random.normal(

                0,

                participant_noise,

                len(time)
            )

            pupil = (
                pupil_clean +
                drift +
                noise
            )

            # -------------------------------------------------
            # dataframe row
            # -------------------------------------------------
            row = {}

            row["participant"] = (
                participant_id
            )

            row["Condition"] = cond

            row["trial"] = trial

            for i in range(
                len(time)
            ):

                row[i] = pupil[i]

            rows.append(row)

# =====================================================
# DATAFRAME
# =====================================================
df = pd.DataFrame(
    rows
)

pupil_cols = list(
    range(len(time))
)

print("\n=== DATAFRAME ===")

print(df.head())

print("\nShape:", df.shape)

print(
    "\nParticipants:",
    df["participant"].nunique()
)

print(
    "Conditions:",
    df["Condition"].unique()
)

# =====================================================
# FIT MODEL
# =====================================================
print("\nRunning pupildeconvolve...")

res_df = fit_dataframe(

    df,

    pupil_cols=pupil_cols,

    participant_col="participant",

    condition_col="Condition",

    sampling_rate=sampling_rate,

    pulse_interval=pulse_interval,

    n_runs=10,

    verbose=True,

    show_progress=True
)

print("\n=== RESULT ===")

print(res_df.head())

print("\nResult shape:")

print(res_df.shape)

# =====================================================
# ATTENTION TIME
# =====================================================
attention_time = np.searchsorted(

    time,

    pulse_times
)

# =====================================================
# PLOT CONDITIONS
# =====================================================
for cond in conditions:

    print(f"\nPlotting {cond}...")

    # -------------------------------------------------
    # pupil data
    # -------------------------------------------------
    participant_means = []

    for pid in df["participant"].unique():

        df_sub = df[
            (df["participant"] == pid)
            &
            (df["Condition"] == cond)
        ]

        pupil_matrix = (
            df_sub[pupil_cols]
            .astype(float)
            .values
        )

        pupil_avg = np.nanmean(

            pupil_matrix,

            axis=0
        )

        participant_means.append(
            pupil_avg
        )

    pupil_matrix = np.vstack(
        participant_means
    )

    # -------------------------------------------------
    # attention pulses
    # -------------------------------------------------
    res_cond = res_df[
        res_df["condition"] == cond
    ]

    pulse_cols = [

        c for c in res_cond.columns

        if str(c).startswith(
            "Pulse"
        )
    ]

    attention_pulses = (
        res_cond[pulse_cols]
        .astype(float)
        .values
    )

    # -------------------------------------------------
    # plot
    # -------------------------------------------------
    plot_pupil_and_pulses(

        pupil=pupil_matrix,

        result={

            "attention_amplitude":
            attention_pulses,

            "attention_time":
            attention_time
        },

        time=time,

        title=f"{cond} Condition",

        show_individual=False
    )

print("\nDONE")