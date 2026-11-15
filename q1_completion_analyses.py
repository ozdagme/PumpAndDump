#!/usr/bin/env python3
"""Complete low-cost Q1 checks without retraining any model.

Produces (1) a fixed-three-epoch training-loss audit from the archived log and
(2) 10,000-draw two-way date/company bootstrap comparisons from saved
seed-ensemble predictions.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from q1_posthoc_analysis import average_precision, grouped_average_precision, holm_adjust


OUT = Path("q1_completion_results")
FULL_ZIP = Path("prism-uploads/full_v4_final_results.zip")
ACTIVITY_ZIP = Path("prism-uploads/result_logs_v4_activity_controlled.zip")
REPETITIONS = 10_000
BATCH_SIZE = 100


def zip_bytes(path: Path, member: str) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read(member)


def zip_csv(path: Path, member: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(zip_bytes(path, member)), **kwargs)


def convergence_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    text = zip_bytes(FULL_ZIP, "result_logs_v4_full_v4/experiment.log").decode(
        "utf-8", errors="replace"
    ).replace("\x00", "")
    year = None
    stage = "unknown"
    fold = np.nan
    active: dict[int, dict] = {}
    rows: list[dict] = []
    run_id = 0
    year_re = re.compile(r"Dış fold full_v4/(20\d{2})")
    rolling_re = re.compile(r"Rolling-origin seed=(\d+) fold=(\d+)")
    epoch_re = re.compile(
        r"Seed (\d+) bce epoch ([123])/3: TGN loss=([0-9.]+), observed_prior=([0-9.]+)"
    )
    for line in text.splitlines():
        match = year_re.search(line)
        if match:
            year = int(match.group(1))
        match = rolling_re.search(line)
        if match:
            stage = "rolling_validation_fit"
            fold = int(match.group(2))
        if "Rolling-origin model seçimi" in line and "100.00%" in line:
            stage = "final_outer_fit"
            fold = np.nan
        match = epoch_re.search(line)
        if not match or year is None:
            continue
        seed, epoch, loss, prior = (
            int(match.group(1)), int(match.group(2)), float(match.group(3)), float(match.group(4))
        )
        if epoch == 1:
            run_id += 1
            active[seed] = {
                "Run_ID": run_id,
                "Test_Year": year,
                "Stage": stage,
                "Fold": fold,
                "Seed": seed,
                "Observed_Prior": prior,
            }
        record = active.get(seed)
        if record is None:
            continue
        record[f"Loss_Epoch_{epoch}"] = loss
        if epoch == 3:
            record["Absolute_Change_1_to_3"] = loss - record["Loss_Epoch_1"]
            record["Relative_Change_1_to_3"] = loss / record["Loss_Epoch_1"] - 1.0
            record["Monotonic_Decrease"] = bool(
                record["Loss_Epoch_1"] >= record["Loss_Epoch_2"] >= record["Loss_Epoch_3"]
            )
            rows.append(record.copy())
            del active[seed]
    detail = pd.DataFrame(rows).drop_duplicates(
        ["Test_Year", "Stage", "Fold", "Seed", "Loss_Epoch_1", "Loss_Epoch_3"]
    )
    summary = (
        detail.groupby(["Test_Year", "Stage"], as_index=False)
        .agg(
            Runs=("Run_ID", "count"),
            Epoch1_Mean=("Loss_Epoch_1", "mean"),
            Epoch2_Mean=("Loss_Epoch_2", "mean"),
            Epoch3_Mean=("Loss_Epoch_3", "mean"),
            Relative_Change_Mean=("Relative_Change_1_to_3", "mean"),
            Monotonic_Decrease_Rate=("Monotonic_Decrease", "mean"),
        )
        .sort_values(["Test_Year", "Stage"])
    )
    return detail.sort_values(["Test_Year", "Stage", "Seed", "Fold"]), summary


def ensemble_predictions() -> pd.DataFrame:
    full = zip_csv(
        FULL_ZIP,
        "result_logs_v4_full_v4/all_test_predictions.csv",
        dtype={"Olay_ID": "string"},
    )
    controlled = zip_csv(
        ACTIVITY_ZIP,
        "result_logs_v4_activity_controlled/all_test_predictions.csv",
        dtype={"Olay_ID": "string"},
    )
    baselines = zip_csv(
        FULL_ZIP,
        "result_logs_v4_full_v4/strong_baselines/baseline_predictions.csv",
        dtype={"Olay_ID": "string"},
    )
    keys = ["Test_Year", "Tarih", "Hisse_Kodu", "Label"]
    deep = full.groupby(keys, as_index=False).agg(
        TGN=("TGN_Prob", "mean"), Text=("Text_Prob", "mean"), Static=("Graph_Prob", "mean")
    )
    activity_tgn = controlled.groupby(keys, as_index=False).agg(
        Controlled_TGN=("TGN_Prob", "mean")
    )
    baseline = (
        baselines.groupby(keys + ["Model"], as_index=False)
        .agg(Probability=("Probability", "mean"))
        .pivot(index=keys, columns="Model", values="Probability")
        .reset_index()
    )
    merged = deep.merge(activity_tgn, on=keys, validate="one_to_one").merge(
        baseline, on=keys, validate="one_to_one"
    )
    return merged


def holm_by_family(frame: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for _, group in frame.groupby("Family", sort=False):
        adjusted = holm_adjust(group["P_Value"])
        part = group.copy()
        part["P_Holm_Family"] = adjusted
        outputs.append(part)
    return pd.concat(outputs, ignore_index=True)


def daily_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    model_columns = [
        "TGN", "Text", "Static", "Train_Prior", "Activity_Logit",
        "Activity_HistGB", "TFIDF_WordChar_Logit", "TFIDF_Activity_Logit",
        "Controlled_TGN",
    ]
    comparisons = [
        ("internal", "TGN", "Text"),
        ("internal", "TGN", "Static"),
        *[("strong", "TGN", model) for model in model_columns[3:8]],
        ("activity_control", "Controlled_TGN", "TGN"),
        ("activity_control", "Controlled_TGN", "Activity_Logit"),
    ]
    output: list[dict] = []
    for year, group in frame.groupby("Test_Year"):
        group = group.reset_index(drop=True)
        labels = group["Label"].to_numpy(dtype=int)
        dates, date_index = np.unique(group["Tarih"].astype(str), return_inverse=True)
        companies, company_index = np.unique(group["Hisse_Kodu"].astype(str), return_inverse=True)
        observed = {
            model: average_precision(labels, group[model].to_numpy(dtype=float))
            for model in model_columns
        }
        distributions = {
            model: np.empty(REPETITIONS, dtype=float) for model in model_columns
        }
        rng = np.random.default_rng(20260904 + int(year))
        cursor = 0
        while cursor < REPETITIONS:
            size = min(BATCH_SIZE, REPETITIONS - cursor)
            date_counts = rng.multinomial(len(dates), np.full(len(dates), 1 / len(dates)), size=size)
            company_counts = rng.multinomial(
                len(companies), np.full(len(companies), 1 / len(companies)), size=size
            )
            weights = date_counts[:, date_index] * company_counts[:, company_index]
            for model in model_columns:
                distributions[model][cursor : cursor + size] = grouped_average_precision(
                    labels, group[model].to_numpy(dtype=float), weights
                )
            cursor += size
        for family, first, second in comparisons:
            differences = distributions[first] - distributions[second]
            differences = differences[np.isfinite(differences)]
            lower = (np.sum(differences <= 0) + 1) / (len(differences) + 1)
            upper = (np.sum(differences >= 0) + 1) / (len(differences) + 1)
            output.append(
                {
                    "Test_Year": int(year),
                    "Family": family,
                    "Comparison": f"{first}-{second}",
                    "N": len(group),
                    "Observed_Difference": observed[first] - observed[second],
                    "Bootstrap_Mean_Difference": float(np.mean(differences)),
                    "CI_Low": float(np.quantile(differences, 0.025)),
                    "CI_High": float(np.quantile(differences, 0.975)),
                    "P_Value": float(min(1.0, 2 * min(lower, upper))),
                    "Bootstrap_Valid": len(differences),
                    "Repetitions": REPETITIONS,
                }
            )
    return holm_by_family(pd.DataFrame(output)).sort_values(
        ["Family", "Test_Year", "Comparison"]
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    detail, convergence = convergence_audit()
    detail.to_csv(OUT / "training_loss_by_run.csv", index=False)
    convergence.to_csv(OUT / "training_loss_summary.csv", index=False)
    bootstrap = daily_bootstrap(ensemble_predictions())
    bootstrap.to_csv(OUT / "daily_two_way_bootstrap_10000.csv", index=False)
    print(f"training runs={len(detail)}; bootstrap comparisons={len(bootstrap)}")
    print(convergence.to_string(index=False))
    print("Holm-significant daily comparisons:", int((bootstrap.P_Holm_Family < 0.05).sum()))


if __name__ == "__main__":
    main()