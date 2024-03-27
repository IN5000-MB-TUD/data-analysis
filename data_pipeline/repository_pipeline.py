import logging
from math import ceil
from pathlib import Path

import joblib
import pandas as pd

from connection import mo
from data_mining.repositories import update_github_repository_data
from data_mining.statistics import update_repository_statistics_data
from data_mining.statistics_fill_gaps import fill_repository_statistics_gaps
from data_processing.t2f.extraction.extractor import extract_pair_series_features
from data_processing.time_series_phases import extrapolate_phases_properties
from data_processing.time_series_plot import create_plot
from utils.data import (
    get_stargazers_time_series,
    get_metrics_information,
    get_metric_time_series,
)
from utils.main import normalize
from utils.time_series import group_metric_by_month, time_series_phases

# Setup logging
log = logging.getLogger(__name__)

REPOSITORY_FULL_NAME = "kubernetes/kubernetes"


if __name__ == "__main__":
    log.info(f"Running pipeline for repository {REPOSITORY_FULL_NAME}\n")

    # STEP 1: DATA COLLECTION
    log.info("STEP 1: DATA COLLECTION")
    # Check if the data in te DB exists. Otherwise, collect it.
    repository_url_split = REPOSITORY_FULL_NAME.split("/")
    repos_owner = repository_url_split[-2]
    repos_name = repository_url_split[-1]

    repository_db_record = mo.db["repositories_data"].find_one(
        {"full_name": REPOSITORY_FULL_NAME}
    )
    if not repository_db_record:
        log.info(
            f"No record found for repository {REPOSITORY_FULL_NAME}, start collecting data..."
        )

        # Collect data for repository
        if update_github_repository_data(repos_owner, repos_name):
            log.info("Collected repository data")
        else:
            log.warning("Data collection was not possible, please try again")
            exit()

        # Collect the stored data
        repository_db_record = mo.db["repositories_data"].find_one(
            {"full_name": REPOSITORY_FULL_NAME}
        )

        # Collect statistics data
        if update_repository_statistics_data(repository_db_record):
            log.info("Collected repository statistics data")
        else:
            log.warning("Statistics data collection was not possible, please try again")
            exit()

        # Fill eventual gaps in the collected statistics
        fill_repository_statistics_gaps(repository_db_record)
    else:
        log.info(f"Record found for repository {REPOSITORY_FULL_NAME} in the DB")

    log.info("---------------------------------------------------\n")

    # STEP 2: STATISTICS DATA PREPARATION
    log.info("STEP 2: STATISTICS DATA PREPARATION")

    # Group metrics data by month
    log.info("Grouping metrics statistical data by month...")
    repository_age_months = ceil(repository_db_record["age"] / 2629746)
    repository_age_start = repository_db_record["created_at"].replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )

    metrics_time_series = {}
    stargazers_dates, _ = get_stargazers_time_series(repository_db_record)
    stargazers_by_month = group_metric_by_month(
        stargazers_dates, repository_age_months, repository_age_start
    )
    stargazers_by_month_dates, stargazers_by_month_values = zip(*stargazers_by_month)
    stargazers_by_month_dates = list(stargazers_by_month_dates)
    stargazers_by_month_values = list(stargazers_by_month_values)

    metrics_time_series["stargazers"] = {
        "dates": stargazers_by_month_dates,
        "values": stargazers_by_month_values,
    }

    for metric in get_metrics_information():
        metric_dates, _ = get_metric_time_series(
            repository_db_record,
            metric[0],
            metric[1],
            metric[2],
            metric[3],
        )

        metric_by_month = group_metric_by_month(
            metric_dates, repository_age_months, repository_age_start
        )

        metric_by_month_dates, metric_by_month_values = zip(*metric_by_month)
        metric_by_month_dates = list(metric_by_month_dates)
        metric_by_month_values = list(metric_by_month_values)

        metrics_time_series[metric[1]] = {
            "dates": metric_by_month_dates,
            "values": metric_by_month_values,
        }

    # Plot metrics curves
    # for metric, metric_data in metrics_time_series.items():
    #     log.info(f"Plotting metric {metric} curve")
    #     create_plot(
    #         "{} {}".format(metric, repository_db_record["full_name"]),
    #         "Total: {}".format(metric_data["values"][-1]),
    #         "Date",
    #         "Count",
    #         metric_data["dates"],
    #         [metric_data["values"]],
    #     )

    log.info("---------------------------------------------------\n")

    # STEP 3: PHASES CLUSTERING
    log.info("STEP 3: PHASES CLUSTERING")

    log.info("Computing metrics phases...")
    metrics_phases = {}
    for metric, metric_data in metrics_time_series.items():
        metric_phases_idxs = time_series_phases(metric_data["values"])
        metrics_phases[metric] = {
            "phases": metric_phases_idxs,
            "phases_count": len(metric_phases_idxs),
        }

    log.info("Extrapolating metrics time series phases statistical properties...")
    phases_features = pd.DataFrame()
    for metric, metric_data in metrics_time_series.items():
        metric_phases = metrics_phases[metric]["phases"]
        metric_by_month = metric_data["values"]

        extracted_features = extrapolate_phases_properties(
            metric_phases, metric_by_month
        )

        phases_features = pd.concat(
            [phases_features, extracted_features], ignore_index=True
        )

    log.info("Clustering the metrics phases...")
    df_phases = phases_features.drop(columns=["phase_order"])
    if not Path("../models/phases/mts_phases.pickle").exists():
        log.warning(
            "The phases clustering model does not exists in the /models/phases folder. "
            "Run the data_processing/time_series_phases.py script to create it."
        )
        exit()
    phases_clustering_model = joblib.load("../models/phases/mts_phases.pickle")

    clustered_phases = phases_clustering_model.fit_predict(df_phases)
    phases_features["phase_cluster"] = clustered_phases

    log.info("Building metrics phases sequences...")
    phases_rows_counter = 0
    for metric, metric_phases_data in metrics_phases.items():
        metrics_phases[metric]["phases_sequence"] = []
        for i in range(0, metric_phases_data["phases_count"]):
            metrics_phases[metric]["phases_sequence"].append(
                phases_features["phase_cluster"][phases_rows_counter + i]
            )

        metrics_phases[metric]["phases_average"] = (
            sum(metrics_phases[metric]["phases_sequence"])
            / metric_phases_data["phases_count"]
        )
        phases_rows_counter += metric_phases_data["phases_count"]

        log.info(
            f"Phases sequence for metric {metric}: {metrics_phases[metric]['phases_sequence']}"
        )

    log.info("---------------------------------------------------\n")

    # STEP 4: REPOSITORY CLUSTERING
    log.info("STEP 3: REPOSITORY CLUSTERING")

    log.info("Building cluster vector...")

    log.info("Process metrics phases sequences...")
    max_phases = 6
    metrics_phases_sequence = {}
    for metric, metric_phases_data in metrics_phases.items():
        for i in range(0, max_phases):
            if i < metric_phases_data["phases_count"]:
                item_value = metrics_phases[metric]["phases_sequence"][i]
            else:
                item_value = metrics_phases[metric]["phases_average"]
            metrics_phases_sequence[f"metric_{metric}_phase_{i}"] = item_value

    df_repository_phases_clustering = pd.DataFrame([metrics_phases_sequence])
    df_repository_phases_clustering = df_repository_phases_clustering.reindex(
        sorted(df_repository_phases_clustering.columns), axis=1
    )

    log.info("Compute pattern distance between metrics...")
    metrics_values_pairs = []
    for metric, metric_data in metrics_time_series.items():
        metric_by_month_values_normalized = normalize(metric_data["values"], 0, 1)
        metrics_values_pairs.append(
            [
                (metric_data["dates"][i], metric_by_month_values_normalized[i])
                for i in range(len(metric_data["dates"]))
            ]
        )

    features_pair = extract_pair_series_features(metrics_values_pairs)
    ts_features_df = pd.DataFrame([features_pair])

    log.info("Cluster the repository...")
    df_feats = pd.concat([df_repository_phases_clustering, ts_features_df], axis=1)
    if not Path(f"../models/clustering/mts_clustering.pickle").exists():
        log.warning(
            "The repositories clustering model does not exists in the /models/clustering folder."
            "Run the data_processing/time_series_clustering.py script to create it."
        )
        exit()
    repos_clustering_model = joblib.load("../models/clustering/mts_clustering.pickle")

    clustered_repository = repos_clustering_model.fit_predict(df_feats)
    log.info(f"The repository was assigned to cluster {clustered_repository}")

    log.info("---------------------------------------------------\n")

    # STEP 5: METRICS FORECASTING
    log.info("STEP 5: METRICS FORECASTING")

    log.info("---------------------------------------------------\n")
