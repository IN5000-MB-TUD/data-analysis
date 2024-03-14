import logging
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
from itertools import groupby

from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

from connection import mo
from data_processing.t2f.model.clustering import ClusterWrapper
from utils.data import get_stargazers_time_series, get_metric_time_series

# Setup logging
log = logging.getLogger(__name__)

STATISTICAL_SETTINGS = ComprehensiveFCParameters()
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


# Utility functions to form groupings
def group_util(date, min_date):
    return (date - min_date).days // 31


def group_metric_by_month(dates, total_months, min_date):
    if not dates:
        return []

    dates_grouped = []
    dates.sort()

    for key, val in groupby(dates, key=lambda date: group_util(date, min_date)):
        dates_grouped.append((key, list(val)))

    time_series_cumulative_by_month = []
    metric_counter = -1
    dates_grouped_idx = 0
    grouped_months_count = len(dates_grouped)
    for month_idx in range(total_months):
        if (
            dates_grouped_idx < grouped_months_count
            and month_idx == dates_grouped[dates_grouped_idx][0]
        ):
            metric_counter += len(dates_grouped[dates_grouped_idx][1])
            dates_grouped_idx += 1

        time_series_cumulative_by_month.append(
            (min_date + relativedelta(months=month_idx), metric_counter)
        )

    return time_series_cumulative_by_month


def time_series_phases(time_series, show_plot=False):
    time_series_np = np.array([value for _, value in time_series], dtype="int")

    model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
    pen = np.log(time_series_np.shape[0]) * 1 * time_series_np.std() ** 2

    algo = rpt.Window(width=min(12, time_series_np.shape[0] - 1), model=model).fit(
        time_series_np
    )
    phases_break_points = algo.predict(pen=pen)

    if show_plot:
        rpt.show.display(
            time_series_np, phases_break_points, phases_break_points, figsize=(10, 6)
        )
        plt.show()

    return phases_break_points


if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval from Database")

    # Get the repositories in the database
    repositories = mo.db["repositories_data"].find({"statistics": {"$exists": True}})

    if not Path("../data/time_series_phases.csv").exists():
        phases_features = pd.DataFrame()

        for idx, repository in enumerate(repositories):
            log.info("Analyzing repository {}".format(repository["full_name"]))

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
                log.info(f"{metric} phases: {metric_phases_idxs}")

            # Extrapolate metrics time series phases statistical properties
            for metric in time_series_dates.keys():
                metric_phases = time_series_phases_idxs[metric]
                metric_by_month = time_series_metrics_by_month[metric]

                phases_time_series = []
                phase_idx_cumulative = 0
                timestamp_time_series = []
                for phase_idx, phase in enumerate(metric_phases):
                    phases_time_series.extend(
                        [phase_idx + 1] * (phase - phase_idx_cumulative)
                    )
                    timestamp_time_series.extend(
                        list(range(0, (phase - phase_idx_cumulative)))
                    )
                    phase_idx_cumulative = phase

                # Build data frame
                df_rows = []
                for metric_idx, metric_tuple in enumerate(metric_by_month):
                    df_rows.append(
                        (
                            phases_time_series[metric_idx],
                            timestamp_time_series[metric_idx],
                            metric_tuple[1],
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
                        "friedrich_coefficients": STATISTICAL_SETTINGS[
                            "friedrich_coefficients"
                        ],
                        "standard_deviation": STATISTICAL_SETTINGS[
                            "standard_deviation"
                        ],
                        "skewness": STATISTICAL_SETTINGS["skewness"],
                        "autocorrelation": STATISTICAL_SETTINGS["autocorrelation"],
                    },
                )
                extracted_features = extracted_features.fillna(0)
                extracted_features = extracted_features.rename_axis(
                    "phase_order"
                ).reset_index()
                phases_features = pd.concat(
                    [phases_features, extracted_features], ignore_index=True
                )

            log.info("----------------------")

        # Store data
        phases_features.to_csv("../data/time_series_phases.csv", index=False)
    else:
        # Load data frame
        phases_features = pd.read_csv("../data/time_series_phases.csv")

    log.info("Cluster phases")

    transform_type = "std"  # preprocessing step
    model_type = "Hierarchical"  # clustering model

    # Clustering
    max_clusters = phases_features["phase_order"].max()
    df_phases = phases_features.drop(columns=["phase_order"])

    best_fit = -1
    clusters = 3
    for n_cluster in range(3, max_clusters):
        model = ClusterWrapper(
            n_clusters=n_cluster, model_type=model_type, transform_type=transform_type
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

    # Print clustered repos
    clustered_phases = model.fit_predict(df_phases)
    phases_features["phase_order"] = clustered_phases

    phases_features = phases_features.groupby(["phase_order"]).mean()

    # Plot phases curves
    x = list(range(0, 13))

    for idx, row in phases_features.iterrows():
        poly_coefficients = [
            row[3],
            row[2],
            row[1],
            row[0],
        ]
        y = np.polyval(poly_coefficients, x)
        plt.plot(x, y, "-r")
        plt.title(f"Phase {idx + 1}")
        plt.show()
