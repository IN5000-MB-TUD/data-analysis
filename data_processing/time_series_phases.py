import json
import logging
from math import ceil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

from connection import mo
from data_processing.t2f.model.clustering import ClusterWrapper
from utils.data import get_stargazers_time_series, get_metric_time_series
from utils.time_series import group_metric_by_month, time_series_phases

# Setup logging
log = logging.getLogger(__name__)

STATISTICAL_SETTINGS = ComprehensiveFCParameters()
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def extrapolate_phases_properties(metric_phases, metric_by_month):
    """
    Extrapolate the given metric phases statistical properties.

    :param metric_phases: The metric phases indexes.
    :param metric_by_month: The metric values by month time series.
    :return: The dataframe of the extracted features for the metric.
    """
    phases_time_series = []
    phase_idx_cumulative = 0
    timestamp_time_series = []
    for phase_idx, phase in enumerate(metric_phases):
        phases_time_series.extend([phase_idx + 1] * (phase - phase_idx_cumulative))
        timestamp_time_series.extend(list(range(0, (phase - phase_idx_cumulative))))
        phase_idx_cumulative = phase

    # Build data frame
    df_rows = []
    for metric_idx, metric_data in enumerate(metric_by_month):
        if isinstance(metric_data, tuple):
            metric_value = metric_data[1]
        else:
            metric_value = metric_data

        df_rows.append(
            (
                phases_time_series[metric_idx],
                timestamp_time_series[metric_idx],
                metric_value,
            )
        )

    df_metric = pd.DataFrame(df_rows, columns=["phase", "time", "value"])
    df_metric["value"] = (
        df_metric.groupby("phase")["value"]
        .apply(lambda v: (v - v.min()) / (v.max() - v.min()))
        .reset_index(level=0, drop=True)
    )
    df_metric = df_metric.fillna(0)
    extracted_features = extract_features(
        df_metric,
        column_id="phase",
        column_sort="time",
        default_fc_parameters={
            "friedrich_coefficients": STATISTICAL_SETTINGS["friedrich_coefficients"],
            "standard_deviation": STATISTICAL_SETTINGS["standard_deviation"],
            "skewness": STATISTICAL_SETTINGS["skewness"],
            "autocorrelation": STATISTICAL_SETTINGS["autocorrelation"],
        },
    )
    extracted_features = extracted_features.fillna(0)
    extracted_features = extracted_features.rename_axis("phase_order").reset_index()

    return extracted_features


def _prepare_repository_clustering_phases_files(
    repository_metrics_phases_count, phases_features
):
    """
    Prepare files for the clustering script.

    :param repository_metrics_phases_count: The phases count for each repository metric.
    :param phases_features: The phases feature dataframe.
    """
    df_repository_phases_clustering_rows = []
    repository_metrics_phases = {}
    phases_rows_counter = 0
    for (
        repository_full_name,
        repository_metrics,
    ) in repository_metrics_phases_count.items():
        metrics_phases_sequence = {}
        repository_metrics_phases[repository_full_name] = {}
        for metric, metric_phases_count in repository_metrics.items():
            repository_metrics_phases[repository_full_name][metric] = []
            phases_average = 0
            for i in range(0, metric_phases_count):
                item_value = phases_features["phase_order"][phases_rows_counter + i]
                phases_average += item_value
                metrics_phases_sequence[f"metric_{metric}_phase_{i}"] = item_value
                repository_metrics_phases[repository_full_name][metric].append(
                    int(item_value)
                )

                # Fill until max phases count with phases mean for the current metric
            phases_average /= metric_phases_count
            for i in range(metric_phases_count, max_clusters):
                metrics_phases_sequence[f"metric_{metric}_phase_{i}"] = phases_average
            phases_rows_counter += metric_phases_count
        df_repository_phases_clustering_rows.append(metrics_phases_sequence)

    df_repository_phases_clustering = pd.DataFrame(df_repository_phases_clustering_rows)
    df_repository_phases_clustering["id"] = list(repository_metrics_phases_count.keys())
    df_repository_phases_clustering = df_repository_phases_clustering.reindex(
        sorted(df_repository_phases_clustering.columns), axis=1
    )
    df_repository_phases_clustering.to_csv(
        "../data/time_series_clustering_phases.csv", index=False
    )
    with open("../data/repository_metrics_phases.json", "w") as outfile:
        json.dump(repository_metrics_phases, outfile, indent=4)


if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval from Database")

    # Get the repositories in the database
    repositories = mo.db["repositories_data"].find({"statistics": {"$exists": True}})
    repository_metrics_phases_count = {}

    if not Path("../data/time_series_phases.csv").exists():
        phases_features = pd.DataFrame()

        for idx, repository in enumerate(repositories):
            log.info("Analyzing repository {}".format(repository["full_name"]))

            repository_metrics_phases_count[repository["full_name"]] = {}

            repository_age_months = ceil(repository["age"] / 2629746)
            repository_age_start = repository["created_at"].replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )

            stargazers_dates, _ = get_stargazers_time_series(repository)

            issues_dates, _ = get_metric_time_series(
                repository, "statistics_issues", "issues", "created_at", "open_issues"
            )

            commits_dates, _ = get_metric_time_series(
                repository, "statistics_commits", "commits", "date", "commits"
            )

            contributors_dates, _ = get_metric_time_series(
                repository, "statistics_commits", "contributors", "first_commit", None
            )

            deployments_dates, _ = get_metric_time_series(
                repository, "statistics_deployments", "deployments", "created_at", None
            )

            forks_dates, _ = get_metric_time_series(
                repository, "statistics_forks", "forks", "created_at", "forks_count"
            )

            pull_requests_dates, _ = get_metric_time_series(
                repository,
                "statistics_pull_requests",
                "pull_requests",
                "created_at",
                None,
            )

            workflows_dates, _ = get_metric_time_series(
                repository, "statistics_workflow_runs", "workflows", "created_at", None
            )

            # Compute time series phases
            time_series_dates = {
                "stargazers": stargazers_dates,
                "issues": issues_dates,
                "commits": commits_dates,
                "contributors": contributors_dates,
                "deployments": deployments_dates,
                "forks": forks_dates,
                "pull_requests": pull_requests_dates,
                "workflows": workflows_dates,
            }

            time_series_phases_idxs = {}
            time_series_metrics_by_month = {}

            for metric, metric_dates in time_series_dates.items():
                metric_by_month = group_metric_by_month(
                    metric_dates, repository_age_months, repository_age_start
                )
                metric_phases_idxs = time_series_phases(metric_by_month)
                time_series_phases_idxs[metric] = metric_phases_idxs
                time_series_metrics_by_month[metric] = metric_by_month

                repository_metrics_phases_count[repository["full_name"]][metric] = len(
                    metric_phases_idxs
                )
                log.info(f"{metric} phases: {metric_phases_idxs}")

            # Extrapolate metrics time series phases statistical properties
            for metric in time_series_dates.keys():
                metric_phases = time_series_phases_idxs[metric]
                metric_by_month = time_series_metrics_by_month[metric]

                extracted_features = extrapolate_phases_properties(
                    metric_phases, metric_by_month
                )

                phases_features = pd.concat(
                    [phases_features, extracted_features], ignore_index=True
                )

            log.info("----------------------")

        # Store data
        phases_features.to_csv("../data/time_series_phases.csv", index=False)
        with open("../data/repository_metrics_phases_count.json", "w") as outfile:
            json.dump(repository_metrics_phases_count, outfile, indent=4)
    else:
        # Load data
        phases_features = pd.read_csv("../data/time_series_phases.csv")
        with open("../data/repository_metrics_phases_count.json") as json_file:
            repository_metrics_phases_count = json.load(json_file)

    log.info("Cluster phases")

    transform_type = "std"  # preprocessing step
    model_type = "Hierarchical"  # clustering model

    # Clustering
    max_clusters = phases_features["phase_order"].max()
    df_phases = phases_features.drop(columns=["phase_order"])

    # Check if model exists
    if not Path("../models/phases/mts_phases.pickle").exists():
        best_fit = -1
        clusters = 3
        for n_cluster in range(3, max_clusters):
            model = ClusterWrapper(
                n_clusters=n_cluster,
                model_type=model_type,
                transform_type=transform_type,
            )
            model.model.fit(df_phases)
            cluster_score = silhouette_score(df_phases, model.model.labels_)
            if cluster_score > best_fit:
                best_fit = cluster_score
                clusters = n_cluster
        log.info(
            f"Optimal number of clusters is: {clusters} with silhouette_score: {best_fit}"
        )

        # Save model
        model = ClusterWrapper(
            n_clusters=clusters, model_type=model_type, transform_type=transform_type
        )
        model.model.fit(df_phases)
        joblib.dump(
            model,
            "../models/phases/mts_phases.pickle",
        )
    else:
        model = joblib.load("../models/phases/mts_phases.pickle")

    # Print clustered repos
    clustered_phases = model.fit_predict(df_phases)
    phases_features["phase_order"] = clustered_phases

    # Store repository phases sequence per metric
    _prepare_repository_clustering_phases_files(
        repository_metrics_phases_count, phases_features
    )

    # Store polynomial coefficients
    phases_features = phases_features.groupby(["phase_order"]).mean()

    # Plot phases curves
    x = list(range(0, 13))

    for idx, row in phases_features.iterrows():
        # Store phase properties
        mo.db["evolution_phases"].update_one(
            {"phase_order": idx},
            {"$set": row.to_dict()},
            upsert=True,
        )

        # Plot phase
        poly_coefficients = [
            row[3],
            row[2],
            row[1],
            row[0],
        ]
        y = np.polyval(poly_coefficients, x)
        plt.plot(x, y, label=f"Phase {idx + 1}")

    plt.title(f"Repository Metrics Evolution Phases")
    plt.xlabel("Time (Months)")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.show()
