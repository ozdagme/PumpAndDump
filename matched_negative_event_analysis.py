"""Symmetric event-window evaluation using matched negative company windows.

No model is trained.  Five-seed ensemble predictions are read from the archived
Q1 result packages.  Each positive company-event is matched to other companies
that are negative throughout the same calendar span and have observations on at
least 80% of the positive event dates.  Each bootstrap replicate resamples
positive events and draws one eligible negative company-window per event, so
both classes use the same calendar-window unit and have equal support.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from q1_posthoc_analysis import (
    STRONG_MODELS,
    canonical_event_id,
    grouped_average_precision,
    holm_adjust,
    p_value_two_sided,
)


FULL_ARCHIVE = Path("prism-uploads/full_v4_final_results.zip")
ACTIVITY_ARCHIVE = Path("prism-uploads/result_logs_v4_activity_controlled.zip")
TEMPORAL_ARCHIVE = Path("prism-uploads/result_logs_v4_temporal_ablations.zip")
OUTPUT_DIR = Path("q1_matched_event_results")
REPETITIONS = 10_000
MIN_DATE_COVERAGE = 0.80
RANDOM_SEED = 20260904

MAIN_MODELS = ["TGN", "Text", "Static", "Controlled_TGN", *STRONG_MODELS]
TEMPORAL_MODELS = ["Standard_TGN", "No_Time_Encoding", "Daily_Memory_Reset"]


def read_zip_csv(archive_path: Path, member: str, **kwargs: object) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        return pd.read_csv(io.BytesIO(archive.read(member)), **kwargs)


def load_main_ensemble() -> pd.DataFrame:
    full = read_zip_csv(
        FULL_ARCHIVE,
        "result_logs_v4_full_v4/all_test_predictions.csv",
        dtype={"Olay_ID": "string"},
    )
    controlled = read_zip_csv(
        ACTIVITY_ARCHIVE,
        "result_logs_v4_activity_controlled/all_test_predictions.csv",
        dtype={"Olay_ID": "string"},
    )
    baselines = read_zip_csv(
        FULL_ARCHIVE,
        "result_logs_v4_full_v4/strong_baselines/baseline_predictions.csv",
        dtype={"Olay_ID": "string"},
    )
    for frame in (full, controlled, baselines):
        frame["Olay_ID"] = canonical_event_id(frame["Olay_ID"])
        frame["Tarih"] = pd.to_datetime(frame["Tarih"])
    keys = ["Test_Year", "Tarih", "Hisse_Kodu", "Label", "Olay_ID"]
    full = full.groupby(keys, as_index=False, dropna=False).agg(
        TGN=("TGN_Prob", "mean"),
        Text=("Text_Prob", "mean"),
        Static=("Graph_Prob", "mean"),
    )
    controlled = controlled.groupby(keys, as_index=False, dropna=False).agg(
        Controlled_TGN=("TGN_Prob", "mean")
    )
    baselines = (
        baselines.groupby(keys + ["Model"], as_index=False, dropna=False)
        .agg(Probability=("Probability", "mean"))
        .pivot(index=keys, columns="Model", values="Probability")
        .reset_index()
    )
    result = full.merge(controlled, on=keys, validate="one_to_one")
    result = result.merge(baselines, on=keys, validate="one_to_one")
    if result[MAIN_MODELS].isna().any().any():
        raise ValueError("Ana ensemble tahminlerinde eksik model değeri oluştu")
    return result


def load_temporal_ensemble() -> pd.DataFrame:
    frame = read_zip_csv(
        TEMPORAL_ARCHIVE,
        "result_logs_v4_temporal_ablations/q1_comparison/"
        "temporal_ablation_ensemble_predictions.csv",
        dtype={"Olay_ID": "string"},
    )
    frame["Olay_ID"] = canonical_event_id(frame["Olay_ID"])
    frame["Tarih"] = pd.to_datetime(frame["Tarih"])
    if frame[TEMPORAL_MODELS].isna().any().any():
        raise ValueError("Zamansal ensemble tahminlerinde eksik değer var")
    return frame


def build_catalog(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    positive = frame[frame["Label"].eq(1)]
    for (year, company, event_id), event in positive.groupby(
        ["Test_Year", "Hisse_Kodu", "Olay_ID"], sort=True
    ):
        if not event_id:
            raise ValueError("Pozitif olay kimliği eksik")
        event_dates = pd.Index(sorted(event["Tarih"].unique()))
        start = event_dates.min()
        end = event_dates.max()
        year_frame = frame[frame["Test_Year"].eq(year)]
        positive_scores = event[models].mean()
        candidate_count = 0
        for candidate_company, candidate_span in year_frame[
            year_frame["Hisse_Kodu"].ne(company)
            & year_frame["Tarih"].between(start, end)
        ].groupby("Hisse_Kodu", sort=True):
            if candidate_span["Label"].eq(1).any():
                continue
            matched = candidate_span[candidate_span["Tarih"].isin(event_dates)]
            coverage = matched["Tarih"].nunique() / len(event_dates)
            if coverage + 1e-12 < MIN_DATE_COVERAGE:
                continue
            candidate_count += 1
            row: dict[str, object] = {
                "Test_Year": int(year),
                "Event_Company": str(company),
                "Event_ID": str(event_id),
                "Event_Key": f"{int(year)}::{company}::{event_id}",
                "Event_Start": start.date().isoformat(),
                "Event_End": end.date().isoformat(),
                "Positive_Dates": int(len(event_dates)),
                "Negative_Company": str(candidate_company),
                "Negative_Dates": int(matched["Tarih"].nunique()),
                "Date_Coverage": float(coverage),
            }
            for model in models:
                row[f"Positive_{model}"] = float(positive_scores[model])
                row[f"Negative_{model}"] = float(matched[model].mean())
            rows.append(row)
        if candidate_count == 0:
            raise ValueError(
                f"{year} {company} {event_id} için %{MIN_DATE_COVERAGE*100:.0f} "
                "kapsamlı negatif pencere bulunamadı"
            )
    catalog = pd.DataFrame(rows)
    expected_events = positive.groupby(["Test_Year", "Hisse_Kodu", "Olay_ID"]).ngroups
    if catalog["Event_Key"].nunique() != expected_events:
        raise ValueError("Bütün pozitif olaylar eşleştirme kataloğunda temsil edilmiyor")
    return catalog.sort_values(["Test_Year", "Event_Key", "Negative_Company"])


def weighted_catalog_ap(catalog: pd.DataFrame, model: str) -> float:
    positive = catalog.drop_duplicates("Event_Key")[["Event_Key", f"Positive_{model}"]]
    positive_labels = np.ones(len(positive), dtype=int)
    positive_scores = positive[f"Positive_{model}"].to_numpy(dtype=float)
    positive_weights = np.ones(len(positive), dtype=float)

    candidate_counts = catalog.groupby("Event_Key")["Negative_Company"].transform("count")
    negative_labels = np.zeros(len(catalog), dtype=int)
    negative_scores = catalog[f"Negative_{model}"].to_numpy(dtype=float)
    negative_weights = 1.0 / candidate_counts.to_numpy(dtype=float)

    labels = np.concatenate([positive_labels, negative_labels])
    scores = np.concatenate([positive_scores, negative_scores])
    weights = np.concatenate([positive_weights, negative_weights])[None, :]
    return float(grouped_average_precision(labels, scores, weights)[0])


def bootstrap_year(
    catalog: pd.DataFrame,
    models: list[str],
    repetitions: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    event_keys = catalog["Event_Key"].drop_duplicates().tolist()
    grouped = {key: group.reset_index(drop=True) for key, group in catalog.groupby("Event_Key")}
    rng = np.random.default_rng(seed)
    distributions = {model: np.empty(repetitions, dtype=float) for model in models}
    number_of_events = len(event_keys)

    for repetition in range(repetitions):
        sampled_keys = rng.choice(event_keys, size=number_of_events, replace=True)
        positive_scores = {model: [] for model in models}
        negative_scores = {model: [] for model in models}
        for key in sampled_keys:
            candidates = grouped[str(key)]
            chosen = candidates.iloc[int(rng.integers(0, len(candidates)))]
            for model in models:
                positive_scores[model].append(float(chosen[f"Positive_{model}"]))
                negative_scores[model].append(float(chosen[f"Negative_{model}"]))
        labels = np.r_[np.ones(number_of_events, dtype=int), np.zeros(number_of_events, dtype=int)]
        weights = np.ones((1, len(labels)), dtype=float)
        for model in models:
            scores = np.asarray(positive_scores[model] + negative_scores[model], dtype=float)
            distributions[model][repetition] = grouped_average_precision(labels, scores, weights)[0]

    metric_rows = []
    for model in models:
        values = distributions[model]
        metric_rows.append(
            {
                "Model": model,
                "Weighted_All_Candidates_AUPRC": weighted_catalog_ap(catalog, model),
                "Bootstrap_Mean_AUPRC": float(np.mean(values)),
                "CI_Low": float(np.percentile(values, 2.5)),
                "CI_High": float(np.percentile(values, 97.5)),
                "Positive_Events": number_of_events,
                "Matched_Negative_Per_Replicate": number_of_events,
                "Candidate_Negative_Windows": int(len(catalog)),
                "Matched_Prevalence": 0.5,
            }
        )
    return distributions, pd.DataFrame(metric_rows)


def main_comparison_specification() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    rows.extend(("internal", "TGN", baseline) for baseline in ("Text", "Static"))
    rows.extend(("strong", "TGN", baseline) for baseline in STRONG_MODELS)
    rows.extend(
        [
            ("activity_control", "Controlled_TGN", "TGN"),
            ("activity_control", "Controlled_TGN", "Activity_Logit"),
        ]
    )
    return rows


def compare_catalog(
    catalog: pd.DataFrame,
    models: list[str],
    specification: list[tuple[str, str, str]],
    global_column: str,
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    comparison_rows = []
    for year, year_catalog in catalog.groupby("Test_Year"):
        distributions, metrics = bootstrap_year(
            year_catalog,
            models,
            repetitions=REPETITIONS,
            seed=RANDOM_SEED + seed_offset + int(year),
        )
        metrics.insert(0, "Test_Year", int(year))
        metric_frames.append(metrics)
        for family, first, second in specification:
            difference = distributions[first] - distributions[second]
            comparison_rows.append(
                {
                    "Family": family,
                    "Test_Year": int(year),
                    "Comparison": f"{first}-{second}",
                    "Weighted_Observed_Difference": weighted_catalog_ap(year_catalog, first)
                    - weighted_catalog_ap(year_catalog, second),
                    "Bootstrap_Mean_Difference": float(np.mean(difference)),
                    "CI_Low": float(np.percentile(difference, 2.5)),
                    "CI_High": float(np.percentile(difference, 97.5)),
                    "P_Raw": p_value_two_sided(difference),
                    "Repetitions": REPETITIONS,
                    "Positive_Events": int(year_catalog["Event_Key"].nunique()),
                    "Candidate_Negative_Windows": int(len(year_catalog)),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["P_Holm_Family"] = np.nan
    for _, indices in comparisons.groupby("Family").groups.items():
        comparisons.loc[list(indices), "P_Holm_Family"] = holm_adjust(
            comparisons.loc[list(indices), "P_Raw"]
        )
    comparisons[global_column] = holm_adjust(comparisons["P_Raw"])
    return pd.concat(metric_frames, ignore_index=True), comparisons


def coverage_table(catalog: pd.DataFrame) -> pd.DataFrame:
    event_rows = (
        catalog.groupby(
            ["Test_Year", "Event_Key", "Event_Company", "Event_ID", "Event_Start", "Event_End", "Positive_Dates"],
            as_index=False,
        )
        .agg(
            Eligible_Negative_Windows=("Negative_Company", "nunique"),
            Minimum_Coverage=("Date_Coverage", "min"),
            Mean_Coverage=("Date_Coverage", "mean"),
            Maximum_Coverage=("Date_Coverage", "max"),
        )
    )
    return event_rows.sort_values(["Test_Year", "Event_Key"])


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    main_ensemble = load_main_ensemble()
    temporal_ensemble = load_temporal_ensemble()

    main_catalog = build_catalog(main_ensemble, MAIN_MODELS)
    temporal_catalog = build_catalog(temporal_ensemble, TEMPORAL_MODELS)
    main_keys = set(main_catalog[["Event_Key", "Negative_Company"]].itertuples(index=False, name=None))
    temporal_keys = set(temporal_catalog[["Event_Key", "Negative_Company"]].itertuples(index=False, name=None))
    if main_keys != temporal_keys:
        raise ValueError("Ana ve zamansal paketlerde eşleştirme adayları aynı değil")

    main_metrics, main_comparisons = compare_catalog(
        main_catalog,
        MAIN_MODELS,
        main_comparison_specification(),
        "P_Holm_Global_45",
        seed_offset=0,
    )
    temporal_specification = [
        ("primary_temporal_ablation", "No_Time_Encoding", "Standard_TGN"),
        ("primary_temporal_ablation", "Daily_Memory_Reset", "Standard_TGN"),
        ("diagnostic_ablation", "No_Time_Encoding", "Daily_Memory_Reset"),
    ]
    temporal_metrics, temporal_comparisons = compare_catalog(
        temporal_catalog,
        TEMPORAL_MODELS,
        temporal_specification,
        "P_Holm_All_15",
        seed_offset=10_000,
    )

    main_catalog.to_csv(OUTPUT_DIR / "matched_window_catalog.csv", index=False)
    coverage_table(main_catalog).to_csv(OUTPUT_DIR / "matched_window_coverage.csv", index=False)
    main_metrics.to_csv(OUTPUT_DIR / "matched_window_model_metrics.csv", index=False)
    main_comparisons.to_csv(
        OUTPUT_DIR / "matched_window_bootstrap_comparisons.csv", index=False
    )
    temporal_metrics.to_csv(OUTPUT_DIR / "matched_temporal_model_metrics.csv", index=False)
    temporal_comparisons.to_csv(
        OUTPUT_DIR / "matched_temporal_bootstrap_comparisons.csv", index=False
    )
    manifest = {
        "analysis": "matched negative company-window event evaluation",
        "generated_on": "2026-09-04",
        "repetitions": REPETITIONS,
        "random_seed": RANDOM_SEED,
        "minimum_event_date_coverage": MIN_DATE_COVERAGE,
        "positive_unit": "company-event mean over positive event dates",
        "negative_unit": "other-company window on the same event dates",
        "negative_eligibility": "no positive label in the full event calendar span",
        "bootstrap": "resample events and draw one eligible negative company-window per sampled event",
        "matched_prevalence": 0.5,
        "main_global_holm_tests": 45,
        "temporal_holm_tests": 15,
    }
    (OUTPUT_DIR / "matched_window_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Main coverage")
    print(
        coverage_table(main_catalog)
        .groupby("Test_Year")
        .agg(
            Events=("Event_Key", "nunique"),
            Min_Candidates=("Eligible_Negative_Windows", "min"),
            Median_Candidates=("Eligible_Negative_Windows", "median"),
            Max_Candidates=("Eligible_Negative_Windows", "max"),
        )
        .to_string()
    )
    print("\nMain comparisons sorted by global Holm")
    print(main_comparisons.sort_values("P_Holm_Global_45").head(15).to_string(index=False))
    print("\nTemporal comparisons sorted by Holm")
    print(temporal_comparisons.sort_values("P_Holm_All_15").to_string(index=False))


if __name__ == "__main__":
    main()