import numpy as np
import pandas as pd
import pytest

from pupildeconvolve import (
    deconvolve_dual,
    fit_dataframe
)


# =====================================================
# 1. CORE SMOKE TEST
# =====================================================
def test_deconvolve_runs():

    pupil = np.random.randn(200)

    res = deconvolve_dual(
        pupil,
        sampling_rate=100
    )

    assert isinstance(res, dict)

    assert "attention_amplitude" in res


# =====================================================
# 2. BASIC DATAFRAME TEST
# =====================================================
def test_fit_dataframe_basic():

    df = pd.DataFrame(
        np.random.randn(5, 200)
    )

    df["participant"] = [
        "P1",
        "P1",
        "P2",
        "P2",
        "P3"
    ]

    pupil_cols = df.columns[:200]

    res = fit_dataframe(

        df,

        pupil_cols=pupil_cols,

        participant_col="participant"
    )

    assert isinstance(
        res,
        pd.DataFrame
    )

    assert len(res) > 0


# =====================================================
# 3. CONDITION TEST
# =====================================================
def test_condition_handling():

    df = pd.DataFrame(
        np.random.randn(6, 200)
    )

    df["participant"] = [
        "P1",
        "P1",
        "P2",
        "P2",
        "P3",
        "P3"
    ]

    df["Condition"] = [
        "Happy",
        "Angry",
        "Happy",
        "Angry",
        "Happy",
        "Angry"
    ]

    pupil_cols = df.columns[:200]

    res = fit_dataframe(

        df,

        pupil_cols=pupil_cols,

        participant_col="participant",

        condition_col="Condition"
    )

    # 3 participants × 2 conditions
    assert len(res) == 6


# =====================================================
# 4. OUTPUT STRUCTURE TEST
# =====================================================
def test_output_contains_pulses():

    df = pd.DataFrame(
        np.random.randn(4, 200)
    )

    df["participant"] = [
        "P1",
        "P1",
        "P2",
        "P2"
    ]

    pupil_cols = df.columns[:200]

    res = fit_dataframe(

        df,

        pupil_cols=pupil_cols,

        participant_col="participant"
    )

    pulse_cols = [

        c for c in res.columns

        if str(c).startswith(
            "Pulse"
        )
    ]

    assert len(pulse_cols) > 0


# =====================================================
# 5. INVALID INPUT TEST
# =====================================================
def test_invalid_input():

    with pytest.raises(ValueError):

        deconvolve_dual(
            [1, 2, 3],
            sampling_rate=100
        )


# =====================================================
# 6. MISSING COLUMN TEST
# =====================================================
def test_missing_columns():

    df = pd.DataFrame(
        np.random.randn(5, 200)
    )

    with pytest.raises(ValueError):

        fit_dataframe(

            df,

            pupil_cols=["wrong_col"],

            participant_col="participant"
        )


# =====================================================
# 7. CSV SAVE TEST
# =====================================================
def test_csv_output(tmp_path):

    df = pd.DataFrame(
        np.random.randn(4, 200)
    )

    df["participant"] = [
        "P1",
        "P1",
        "P2",
        "P2"
    ]

    pupil_cols = df.columns[:200]

    output_file = (
        tmp_path / "test.csv"
    )

    fit_dataframe(

        df,

        pupil_cols=pupil_cols,

        participant_col="participant",

        save_csv=True,

        output_path=str(output_file)
    )

    assert output_file.exists()


# =====================================================
# 8. REPRODUCIBILITY TEST
# =====================================================
def test_deterministic():

    pupil = np.random.randn(200)

    res1 = deconvolve_dual(

        pupil,

        sampling_rate=100,

        random_state=42
    )

    res2 = deconvolve_dual(

        pupil,

        sampling_rate=100,

        random_state=42
    )

    assert np.allclose(

        res1["attention_amplitude"],

        res2["attention_amplitude"]
    )