"""Q1 hakem kontrolü için eğitim gerektirmeyen son analizler.

Üretilen çıktılar:

* ensemble_auprc_table.csv
* event_unit_model_metrics.csv
* event_cluster_bootstrap_comparisons.csv
* economic_seed_bootstrap.csv
* economic_daily_block_seed42.csv
* analysis_manifest.txt

Betik tamamlanmış üç sonuç arşivini veya bunların açılmış klasörlerini kullanır;
model eğitmez ve mevcut sonuç dosyalarını değiştirmez.
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = [42, 123, 2024, 555, 999]
STRONG_MODELS = [
    "Train_Prior",
    "Activity_Logit",
    "Activity_HistGB",
    "TFIDF_WordChar_Logit",
    "TFIDF_Activity_Logit",
]


def canonical_event_id(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def average_precision(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    weights = np.ones((1, len(labels_array)), dtype=float)
    return float(grouped_average_precision(labels_array, scores_array, weights)[0])


def locate_default(candidates: list[str]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError("Dosya/klasör bulunamadı: " + ", ".join(candidates))


def materialize(source: Path, expected_name: str, temp_root: Path) -> Path:
    if source.is_dir():
        if source.name == expected_name:
            return source
        direct = source / expected_name
        if direct.is_dir():
            return direct
        matches = list(source.rglob(expected_name))
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(f"{expected_name}, {source} içinde bulunamadı")
    if source.suffix.lower() != ".zip":
        raise ValueError(f"ZIP veya klasör bekleniyordu: {source}")
    target = temp_root / expected_name
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        archive.extractall(target)
    matches = [p for p in target.rglob(expected_name) if p.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{source} içinde tek bir {expected_name} bulunamadı: {matches}"
        )
    return matches[0]


def holm_adjust(p_values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        adjusted = min(1.0, (count - rank) * float(value))
        running = max(running, adjusted)
        output.loc[index] = running
    return output


def p_value_two_sided(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    lower = (np.sum(values <= 0) + 1) / (len(values) + 1)
    upper = (np.sum(values >= 0) + 1) / (len(values) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def load_ensemble(full_dir: Path, activity_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_path = full_dir / "all_test_predictions.csv"
    controlled_path = activity_dir / "all_test_predictions.csv"
    baseline_path = full_dir / "strong_baselines" / "baseline_predictions.csv"

    full = pd.read_csv(full_path, dtype={"Olay_ID": "string"})
    controlled = pd.read_csv(controlled_path, dtype={"Olay_ID": "string"})
    baselines = pd.read_csv(baseline_path, dtype={"Olay_ID": "string"})
    for frame in (full, controlled, baselines):
        frame["Olay_ID"] = canonical_event_id(frame["Olay_ID"])
        frame["Tarih"] = frame["Tarih"].astype(str)

    keys = ["Test_Year", "Tarih", "Hisse_Kodu", "Label", "Olay_ID"]
    full_ensemble = (
        full.groupby(keys, as_index=False, dropna=False)
        .agg(TGN=("TGN_Prob", "mean"), Text=("Text_Prob", "mean"), Static=("Graph_Prob", "mean"))
    )
    controlled_ensemble = (
        controlled.groupby(keys, as_index=False, dropna=False)
        .agg(Controlled_TGN=("TGN_Prob", "mean"))
    )
    baseline_ensemble = (
        baselines.groupby(keys + ["Model"], as_index=False, dropna=False)
        .agg(Probability=("Probability", "mean"))
        .pivot(index=keys, columns="Model", values="Probability")
        .reset_index()
    )
    ensemble = full_ensemble.merge(controlled_ensemble, on=keys, validate="one_to_one")
    ensemble = ensemble.merge(baseline_ensemble, on=keys, validate="one_to_one")

    if ensemble[["TGN", "Text", "Static", "Controlled_TGN", *STRONG_MODELS]].isna().any().any():
        raise ValueError("Model tahminleri birleştirilirken eksik değer oluştu")
    return ensemble, pd.concat(
        [
            seed_metrics_from_wide(full, {"TGN": "TGN_Prob", "Text": "Text_Prob", "Static": "Graph_Prob"}),
            seed_metrics_from_wide(controlled, {"Controlled_TGN": "TGN_Prob"}),
            seed_metrics_from_long(baselines),
        ],
        ignore_index=True,
    )


def seed_metrics_from_wide(frame: pd.DataFrame, models: dict[str, str]) -> pd.DataFrame:
    rows = []
    for (year, seed), group in frame.groupby(["Test_Year", "Seed"]):
        for model, column in models.items():
            rows.append(
                {
                    "Test_Year": int(year),
                    "Seed": int(seed),
                    "Model": model,
                    "AUPRC": average_precision(group["Label"], group[column]),
                }
            )
    return pd.DataFrame(rows)


def seed_metrics_from_long(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, seed, model), group in frame.groupby(["Test_Year", "Seed", "Model"]):
        rows.append(
            {
                "Test_Year": int(year),
                "Seed": int(seed),
                "Model": str(model),
                "AUPRC": average_precision(group["Label"], group["Probability"]),
            }
        )
    return pd.DataFrame(rows)


def make_event_units(frame: pd.DataFrame, model_columns: list[str]) -> pd.DataFrame:
    work = frame.copy()
    positive_missing = work["Label"].eq(1) & work["Olay_ID"].eq("")
    if positive_missing.any():
        raise ValueError("Pozitif satırlarda olay kimliği eksik")
    work["Evaluation_Unit"] = np.where(
        work["Label"].eq(1),
        "EVENT::" + work["Hisse_Kodu"].astype(str) + "::" + work["Olay_ID"],
        "NEG::" + work["Hisse_Kodu"].astype(str) + "::" + work["Tarih"],
    )
    aggregations = {"Label": "max", **{column: "mean" for column in model_columns}}
    return work.groupby("Evaluation_Unit", as_index=False).agg(aggregations)


def grouped_average_precision(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Bootstrap ağırlıkları için bağları doğru ele alan vektörize AP."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(float)
    sorted_weights = weights[:, order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    group_total = np.add.reduceat(sorted_weights, starts, axis=1)
    group_positive = np.add.reduceat(sorted_weights * sorted_labels, starts, axis=1)
    cumulative_total = np.cumsum(group_total, axis=1)
    cumulative_positive = np.cumsum(group_positive, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    total_positive = cumulative_positive[:, -1]
    return np.divide(
        np.sum(precision * group_positive, axis=1),
        total_positive,
        out=np.full(len(weights), np.nan),
        where=total_positive > 0,
    )


def event_unit_bootstrap(
    units: pd.DataFrame,
    model_columns: list[str],
    repetitions: int,
    seed: int,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    labels = units["Label"].to_numpy(dtype=int)
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    if len(positive_indices) < 2 or len(negative_indices) < 2:
        raise ValueError("Bootstrap için her iki sınıfta en az iki değerlendirme birimi gerekli")

    rng = np.random.default_rng(seed)
    distributions = {model: np.empty(repetitions, dtype=float) for model in model_columns}
    probabilities_positive = np.repeat(1.0 / len(positive_indices), len(positive_indices))
    probabilities_negative = np.repeat(1.0 / len(negative_indices), len(negative_indices))

    cursor = 0
    while cursor < repetitions:
        size = min(batch_size, repetitions - cursor)
        weights = np.zeros((size, len(units)), dtype=np.float32)
        weights[:, positive_indices] = rng.multinomial(
            len(positive_indices), probabilities_positive, size=size
        )
        weights[:, negative_indices] = rng.multinomial(
            len(negative_indices), probabilities_negative, size=size
        )
        for model in model_columns:
            distributions[model][cursor : cursor + size] = grouped_average_precision(
                labels, units[model].to_numpy(dtype=float), weights
            )
        cursor += size

    metric_rows = []
    for model in model_columns:
        values = distributions[model]
        metric_rows.append(
            {
                "Model": model,
                "Event_Unit_AUPRC": average_precision(labels, units[model]),
                "Bootstrap_Mean": float(np.nanmean(values)),
                "CI_Low": float(np.nanpercentile(values, 2.5)),
                "CI_High": float(np.nanpercentile(values, 97.5)),
                "Positive_Events": int(len(positive_indices)),
                "Negative_Company_Days": int(len(negative_indices)),
            }
        )
    return pd.DataFrame(metric_rows), distributions


def comparison_specification() -> list[tuple[str, str, str]]:
    comparisons = []
    for baseline in ("Text", "Static"):
        comparisons.append(("internal", "TGN", baseline))
    for baseline in STRONG_MODELS:
        comparisons.append(("strong", "TGN", baseline))
    comparisons.extend(
        [
            ("activity_control", "Controlled_TGN", "TGN"),
            ("activity_control", "Controlled_TGN", "Activity_Logit"),
        ]
    )
    return comparisons


def ensemble_table(
    ensemble: pd.DataFrame, seed_metrics: pd.DataFrame, model_columns: list[str]
) -> pd.DataFrame:
    seed_summary = (
        seed_metrics.groupby(["Test_Year", "Model"], as_index=False)
        .agg(Seed_Mean_AUPRC=("AUPRC", "mean"), Seed_SD_AUPRC=("AUPRC", "std"), Seed_N=("Seed", "nunique"))
    )
    rows = []
    for year, group in ensemble.groupby("Test_Year"):
        units = make_event_units(group, model_columns)
        for model in model_columns:
            rows.append(
                {
                    "Test_Year": int(year),
                    "Model": model,
                    "Ensemble_Day_AUPRC": average_precision(group["Label"], group[model]),
                    "Ensemble_Event_Unit_AUPRC": average_precision(units["Label"], units[model]),
                    "Test_Days": int(len(group)),
                    "Positive_Days": int(group["Label"].sum()),
                    "Positive_Events": int(units["Label"].sum()),
                }
            )
    result = pd.DataFrame(rows).merge(seed_summary, on=["Test_Year", "Model"], how="left")
    columns = [
        "Test_Year",
        "Model",
        "Seed_Mean_AUPRC",
        "Seed_SD_AUPRC",
        "Seed_N",
        "Ensemble_Day_AUPRC",
        "Ensemble_Event_Unit_AUPRC",
        "Test_Days",
        "Positive_Days",
        "Positive_Events",
    ]
    return result[columns].sort_values(["Test_Year", "Model"])


def detection_analysis(
    ensemble: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    output_dir: Path,
    repetitions: int,
    seed: int,
    batch_size: int,
) -> None:
    model_columns = ["TGN", "Text", "Static", "Controlled_TGN", *STRONG_MODELS]
    ensemble_table(ensemble, seed_metrics, model_columns).to_csv(
        output_dir / "ensemble_auprc_table.csv", index=False
    )

    metric_frames = []
    comparison_rows = []
    specifications = comparison_specification()
    for year, group in ensemble.groupby("Test_Year"):
        units = make_event_units(group, model_columns)
        metrics, distributions = event_unit_bootstrap(
            units,
            model_columns,
            repetitions=repetitions,
            seed=seed + int(year),
            batch_size=batch_size,
        )
        metrics.insert(0, "Test_Year", int(year))
        metric_frames.append(metrics)
        for family, first, second in specifications:
            difference = distributions[first] - distributions[second]
            comparison_rows.append(
                {
                    "Family": family,
                    "Test_Year": int(year),
                    "Comparison": f"{first}-{second}",
                    "Observed_Difference": float(
                        average_precision(units["Label"], units[first])
                        - average_precision(units["Label"], units[second])
                    ),
                    "Bootstrap_Mean_Difference": float(np.nanmean(difference)),
                    "CI_Low": float(np.nanpercentile(difference, 2.5)),
                    "CI_High": float(np.nanpercentile(difference, 97.5)),
                    "P_Raw": p_value_two_sided(difference),
                    "Repetitions": repetitions,
                    "Positive_Events": int(units["Label"].sum()),
                }
            )

    comparisons = pd.DataFrame(comparison_rows)
    comparisons["P_Holm_Family"] = np.nan
    for _, indices in comparisons.groupby("Family").groups.items():
        comparisons.loc[list(indices), "P_Holm_Family"] = holm_adjust(
            comparisons.loc[list(indices), "P_Raw"]
        )
    comparisons["P_Holm_Global_45"] = holm_adjust(comparisons["P_Raw"])
    comparisons.to_csv(output_dir / "event_cluster_bootstrap_comparisons.csv", index=False)
    pd.concat(metric_frames, ignore_index=True).to_csv(
        output_dir / "event_unit_model_metrics.csv", index=False
    )


def paired_seed_bootstrap(
    financial_path: Path, output_dir: Path, repetitions: int, seed: int
) -> None:
    frame = pd.read_csv(financial_path)
    frame = frame[np.isclose(frame["Cost"], 0.004)].copy()
    frame = frame[frame["Strategy"].isin(["Raw_LongCash", "Hard_Filter_LongCash"])]
    rng = np.random.default_rng(seed)
    rows = []
    for year, group in frame.groupby("Test_Year"):
        pivot = group.pivot(index="Seed", columns="Strategy", values=["Return", "Sharpe", "MDD"])
        if sorted(pivot.index.tolist()) != sorted(SEEDS):
            raise ValueError(f"{year} için beklenen beş seed bulunamadı")
        for metric in ("Return", "Sharpe", "MDD"):
            differences = (
                pivot[(metric, "Hard_Filter_LongCash")]
                - pivot[(metric, "Raw_LongCash")]
            ).to_numpy(dtype=float)
            sampled = rng.choice(differences, size=(repetitions, len(differences)), replace=True).mean(axis=1)
            rows.append(
                {
                    "Test_Year": int(year),
                    "Metric": metric,
                    "Difference": "Hard_Filter_LongCash-Raw_LongCash",
                    "Observed_Mean_Difference": float(differences.mean()),
                    "CI_Low": float(np.percentile(sampled, 2.5)),
                    "CI_High": float(np.percentile(sampled, 97.5)),
                    "P_Raw": p_value_two_sided(sampled),
                    "Seed_N": len(differences),
                    "Repetitions": repetitions,
                    "Uncertainty_Source": "paired_seed_initialization_only",
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "economic_seed_bootstrap.csv", index=False)


def financial_metrics(returns: np.ndarray, risk_free: np.ndarray) -> tuple[float, float, float]:
    wealth = np.cumprod(1.0 + returns)
    total_return = float(wealth[-1] - 1.0)
    excess = returns - risk_free
    standard_deviation = np.std(excess, ddof=1)
    sharpe = float(np.sqrt(252) * np.mean(excess) / standard_deviation) if standard_deviation > 0 else np.nan
    running_maximum = np.maximum.accumulate(wealth)
    mdd = float(np.max(1.0 - wealth / running_maximum))
    return total_return, sharpe, mdd


def moving_block_indices(rng: np.random.Generator, n: int, block: int) -> np.ndarray:
    number_of_blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=number_of_blocks)
    indices = np.concatenate([(start + np.arange(block)) % n for start in starts])
    return indices[:n]


def daily_block_bootstrap(
    primary_dir: Path,
    output_dir: Path,
    repetitions: int,
    seed: int,
    block_length: int,
) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(2021, 2026):
        experiment_dir = primary_dir / "keep_all" / f"test_{year}" / "bce"
        raw = pd.read_csv(experiment_dir / "curve_Raw_LongCash_cost_0.004.csv")
        filtered = pd.read_csv(experiment_dir / "curve_Hard_Filter_LongCash_cost_0.004.csv")
        merged = raw[["Tarih", "Net_Return", "Risk_Free"]].merge(
            filtered[["Tarih", "Net_Return", "Risk_Free"]],
            on="Tarih",
            suffixes=("_Raw", "_Filtered"),
            validate="one_to_one",
        )
        raw_values = merged["Net_Return_Raw"].to_numpy(dtype=float)
        filtered_values = merged["Net_Return_Filtered"].to_numpy(dtype=float)
        risk_free = merged["Risk_Free_Raw"].to_numpy(dtype=float)
        observed_raw = financial_metrics(raw_values, risk_free)
        observed_filtered = financial_metrics(filtered_values, risk_free)
        distributions = np.empty((repetitions, 3), dtype=float)
        for repetition in range(repetitions):
            indices = moving_block_indices(rng, len(merged), block_length)
            raw_metrics = financial_metrics(raw_values[indices], risk_free[indices])
            filtered_metrics = financial_metrics(filtered_values[indices], risk_free[indices])
            distributions[repetition] = np.asarray(filtered_metrics) - np.asarray(raw_metrics)
        for position, metric in enumerate(("Return", "Sharpe", "MDD")):
            values = distributions[:, position]
            rows.append(
                {
                    "Test_Year": year,
                    "Metric": metric,
                    "Difference": "Hard_Filter_LongCash-Raw_LongCash",
                    "Observed_Difference": float(observed_filtered[position] - observed_raw[position]),
                    "CI_Low": float(np.nanpercentile(values, 2.5)),
                    "CI_High": float(np.nanpercentile(values, 97.5)),
                    "P_Raw": p_value_two_sided(values),
                    "Trading_Days": len(merged),
                    "Block_Length": block_length,
                    "Repetitions": repetitions,
                    "Seed": 42,
                    "Uncertainty_Source": "moving_block_time_uncertainty_seed42_only",
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "economic_daily_block_seed42.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path)
    parser.add_argument("--activity", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("q1_posthoc_results"))
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--block-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 1_000:
        raise ValueError("Q1 analizi için en az 1.000 tekrar kullanın")
    full_source = args.full or locate_default(
        ["full_v4_final_results.zip", "prism-uploads/full_v4_final_results.zip", "result_logs_v4_full_v4"]
    )
    activity_source = args.activity or locate_default(
        [
            "result_logs_v4_activity_controlled.zip",
            "prism-uploads/result_logs_v4_activity_controlled.zip",
            "result_logs_v4_activity_controlled",
        ]
    )
    primary_source = args.primary or locate_default(
        ["result_logs_v4_primary.zip", "prism-uploads/result_logs_v4_primary.zip", "result_logs_v4_primary"]
    )

    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="q1_posthoc_") as temporary:
        temporary_root = Path(temporary)
        full_dir = materialize(full_source, "result_logs_v4_full_v4", temporary_root)
        activity_dir = materialize(
            activity_source, "result_logs_v4_activity_controlled", temporary_root
        )
        primary_dir = materialize(primary_source, "result_logs_v4_primary", temporary_root)

        ensemble, seed_metrics = load_ensemble(full_dir, activity_dir)
        detection_analysis(
            ensemble,
            seed_metrics,
            output_dir,
            repetitions=args.repetitions,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        paired_seed_bootstrap(
            full_dir / "all_financial_results_by_seed.csv",
            output_dir,
            repetitions=args.repetitions,
            seed=args.seed + 100,
        )
        daily_block_bootstrap(
            primary_dir,
            output_dir,
            repetitions=args.repetitions,
            seed=args.seed + 200,
            block_length=args.block_length,
        )

    manifest = "\n".join(
        [
            "Q1 post-hoc analysis completed",
            f"repetitions={args.repetitions}",
            f"random_seed={args.seed}",
            f"moving_block_length={args.block_length}",
            "event_inference=stratified event-unit bootstrap",
            "positive_unit=company-event mean probability",
            "negative_unit=company-day probability",
            "global_multiplicity=Holm across 45 comparisons",
            "economic_seed_ci=paired five-seed bootstrap",
            "economic_time_ci=paired moving-block bootstrap for saved seed-42 curves",
        ]
    )
    (output_dir / "analysis_manifest.txt").write_text(manifest + "\n", encoding="utf-8")
    print(f"Tamamlandı: {output_dir}")


if __name__ == "__main__":
    main()