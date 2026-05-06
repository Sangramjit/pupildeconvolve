import numpy as np
import pandas as pd
import time

from tqdm import tqdm

from .core import deconvolve_dual


# =========================================================
# MAIN FUNCTION
# =========================================================
def fit_dataframe(

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
):

    """
    ======================================================

    PARTICIPANT-WISE PUPIL DECONVOLUTION

    Logic:
    ------------------------------------------------------

    1. Gather all trials for each participant
    2. Average trials participant-wise
    3. Fit ONE model on averaged signal
    4. Output attentional pulse amplitudes

    ------------------------------------------------------

    Trial-wise fitting has been REMOVED.

    ======================================================
    """

    start_time = time.time()

    # =====================================================
    # VALIDATION
    # =====================================================
    if participant_col is None:

        raise ValueError(
            "participant_col is required"
        )

    if participant_col not in df.columns:

        raise ValueError(
            f"{participant_col} not found"
        )

    missing = [

        c for c in pupil_cols

        if c not in df.columns
    ]

    if len(missing) > 0:

        raise ValueError(
            f"Pupil columns missing: {missing}"
        )

    if condition_col is not None:

        if condition_col not in df.columns:

            raise ValueError(
                f"{condition_col} not found"
            )

    # =====================================================
    # GROUPING
    # =====================================================
    if condition_col is not None:

        grouped = df.groupby(
            [participant_col, condition_col]
        )

    else:

        grouped = df.groupby(
            participant_col
        )

    iterator = grouped

    if show_progress:

        iterator = tqdm(
            grouped,
            total=len(grouped),
            desc="Participant fitting"
        )

    results = []

    # =====================================================
    # FIT PARTICIPANT AVERAGES
    # =====================================================
    for group_key, group in iterator:

        # -------------------------------------------------
        # unpack keys
        # -------------------------------------------------
        if condition_col is not None:

            pid, cond = group_key

        else:

            pid = group_key
            cond = None

        # -------------------------------------------------
        # extract pupil matrix
        # -------------------------------------------------
        pupil_matrix = (
            group[pupil_cols]
            .astype(float)
            .values
        )

        # =================================================
        # IMPORTANT:
        # Replace NaNs with 0
        # =================================================
        pupil_matrix = np.nan_to_num(
            pupil_matrix,
            nan=0.0
        )

        # =================================================
        # AVERAGE TRIALS
        # =================================================
        pupil_avg = np.mean(

            pupil_matrix,

            axis=0
        )

        # -------------------------------------------------
        # FIT MODEL ONCE
        # -------------------------------------------------
        res = deconvolve_dual(

            pupil_avg,

            sampling_rate=sampling_rate,

            pulse_interval=pulse_interval,

            pulse_times=pulse_times,

            n_runs=n_runs
        )

        if res is None:

            if verbose:

                print(
                    f"[WARNING] "
                    f"Fit failed for {pid}"
                )

            continue

        # =================================================
        # OUTPUT ROW
        # =================================================
        row_dict = {}

        row_dict["participant"] = pid

        if condition_col is not None:

            row_dict["condition"] = cond

        # -------------------------------------------------
        # model parameters
        # -------------------------------------------------
        row_dict["slope"] = (
            res.get(
                "slope",
                np.nan
            )
        )

        row_dict["plr_latency_ms"] = (
            res.get(
                "plr_latency_ms",
                np.nan
            )
        )

        # -------------------------------------------------
        # attentional pulses
        # -------------------------------------------------
        attn = np.asarray(

            res["attention_amplitude"],

            dtype=float
        )

        # =================================================
        # IMPORTANT:
        # NaN pulse amplitudes -> 0
        # =================================================
        attn = np.nan_to_num(
            attn,
            nan=0.0
        )

        for j, val in enumerate(attn):

            row_dict[
                f"Pulse{j+1}"
            ] = val

        results.append(
            row_dict
        )

    # =====================================================
    # OUTPUT DATAFRAME
    # =====================================================
    df_out = pd.DataFrame(
        results
    )

    # =====================================================
    # SAVE CSV
    # =====================================================
    if save_csv:

        df_out.to_csv(
            output_path,
            index=False
        )

        if verbose:

            print(
                f"\n[INFO] Saved CSV:"
            )

            print(output_path)

    # =====================================================
    # TIMER
    # =====================================================
    elapsed = (
        time.time() - start_time
    )

    # =====================================================
    # SUMMARY
    # =====================================================
    if verbose:

        print("\n=== OUTPUT SUMMARY ===")

        print(
            f"Participants fitted: "
            f"{len(df_out)}"
        )

        print(
            f"Columns: "
            f"{df_out.shape[1]}"
        )

        print(
            f"Total time: "
            f"{elapsed:.2f} sec"
        )

        if len(df_out) > 0:

            print(
                f"Avg per participant: "
                f"{elapsed / len(df_out):.3f} sec"
            )

    return df_out