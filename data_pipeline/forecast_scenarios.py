import logging
from math import ceil
from pathlib import Path

import joblib
import pandas as pd

from connection import mo
from utils.data import (
    get_stargazers_time_series,
    get_releases_time_series,
    get_size_time_series,
    get_metrics_information,
    get_metric_time_series,
)
from utils.pipeline import log_forecast_values
from utils.time_series import group_metric_by_month, group_size_by_month

# Setup logging
log = logging.getLogger(__name__)

REPOSITORY_FULL_NAME = "saltstack/salt"
REPOSITORY_CLUSTER = 0
METRICS = [
    "stargazers",
    "releases",
    "commits",
    "contributors",
    "deployments",
    "issues",
    "forks",
    "pull_requests",
    "workflows",
    "size",
]
PHASES_LABELS = ["Steep", "Shallow", "Plateau"]

if __name__ == "__main__":
    log.info(f"Running forecast scenarios for repository {REPOSITORY_FULL_NAME}\n")

    if not Path("../models/phases/mts_phases_classifier.pickle").exists():
        log.warning(
            "The phases clustering classifier model does not exists in the /models/phases folder. "
            "Run the data_processing/time_series_phases.py script to create it."
        )
        exit()
    phases_clustering_model = joblib.load(
        "../models/phases/mts_phases_classifier.pickle"
    )

    repository_db_record = mo.db["repositories_data"].find_one(
        {"full_name": REPOSITORY_FULL_NAME}
    )
    if not repository_db_record:
        log.info(
            f"No record found for repository {REPOSITORY_FULL_NAME}, run the pipeline script to import the data"
        )
        exit()

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

    releases_dates, _ = get_releases_time_series(repository_db_record)
    releases_by_month = group_metric_by_month(
        releases_dates, repository_age_months, repository_age_start
    )
    releases_by_month_dates, releases_by_month_values = zip(*releases_by_month)
    releases_by_month_dates = list(releases_by_month_dates)
    releases_by_month_values = list(releases_by_month_values)

    metrics_time_series["releases"] = {
        "dates": releases_by_month_dates,
        "values": releases_by_month_values,
    }

    (
        repository_actions_dates,
        repository_actions_total,
        _,
    ) = get_size_time_series(repository_db_record)
    size_by_month = group_size_by_month(
        repository_actions_dates,
        repository_actions_total,
        repository_age_months,
        repository_age_start,
    )
    size_by_month_dates, size_by_month_values = zip(*size_by_month)
    size_by_month_dates = list(size_by_month_dates)
    size_by_month_values = list(size_by_month_values)

    metrics_time_series["size"] = {
        "dates": size_by_month_dates,
        "values": size_by_month_values,
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

    # Build data frame
    log.info("Building the metrics time series dataframe...")
    df_multi_time_series = pd.DataFrame(
        columns=[
            "ds",
            "unique_id",
            "stargazers",
            "issues",
            "commits",
            "contributors",
            "deployments",
            "forks",
            "pull_requests",
            "workflows",
            "releases",
            "size",
        ],
    )

    df_multi_time_series["ds"] = list(range(repository_age_months))
    df_multi_time_series["unique_id"] = [REPOSITORY_FULL_NAME] * repository_age_months
    for metric, metric_data in metrics_time_series.items():
        df_multi_time_series[metric] = metric_data["values"]

    # Scale values between min and max
    metrics_columns = df_multi_time_series.columns.difference(["ds", "unique_id"])
    df_multi_time_series[metrics_columns] = (
        df_multi_time_series[metrics_columns]
        .apply(lambda v: (v - v.min()) / (v.max() - v.min()))
        .reset_index(level=0, drop=True)
    )
    df_multi_time_series = df_multi_time_series.fillna(0)

    models_path = f"../models/forecasting/cluster_{REPOSITORY_CLUSTER}"

    # Retrieve phases from database
    evolution_phases = mo.db["evolution_phases"].find(
        projection={"_id": 0, "phase_name": 0}
    )
    phases_statistical_properties = {}
    for phase in evolution_phases:
        phases_statistical_properties[f"phase_{phase['phase_id']}"] = {
            "coeff_0": phase["value__friedrich_coefficients__coeff_0__m_3__r_30"],
            "coeff_1": phase["value__friedrich_coefficients__coeff_1__m_3__r_30"],
            "coeff_2": phase["value__friedrich_coefficients__coeff_2__m_3__r_30"],
            "coeff_3": phase["value__friedrich_coefficients__coeff_3__m_3__r_30"],
        }

    # Scenario 1
    log.info(
        "SCENARIO 1: New feature implementation over 24 months, how many more contributors are needed?"
    )

    log_forecast_values(
        scenario_metric_target="contributors",
        scenario_months=24,
        repository_full_name=REPOSITORY_FULL_NAME,
        metrics_time_series=metrics_time_series,
        df_multi_time_series=df_multi_time_series,
        metrics_pattern_hypothesis={
            "stargazers": 1,
            "releases": 1,
            "commits": 0,
            "deployments": 2,
            "issues": 1,
            "forks": 1,
            "pull_requests": 0,
            "workflows": 2,
            "size": 0,
        },
        phases_statistical_properties=phases_statistical_properties,
        models_path=models_path,
        phases_labels=PHASES_LABELS,
        phases_clustering_model=phases_clustering_model,
    )

    log.info("---------------------------------------------------\n")

    # Scenario 2
    log.info(
        "SCENARIO 2: Major refactoring over 6 months, how many releases should be planned?"
    )

    log_forecast_values(
        scenario_metric_target="releases",
        scenario_months=6,
        repository_full_name=REPOSITORY_FULL_NAME,
        metrics_time_series=metrics_time_series,
        df_multi_time_series=df_multi_time_series,
        metrics_pattern_hypothesis={
            "stargazers": 2,
            "contributors": 2,
            "commits": 0,
            "deployments": 2,
            "issues": 0,
            "forks": 0,
            "pull_requests": 0,
            "workflows": 2,
            "size": 0,
        },
        phases_statistical_properties=phases_statistical_properties,
        models_path=models_path,
        phases_labels=PHASES_LABELS,
        phases_clustering_model=phases_clustering_model,
    )

    log.info("---------------------------------------------------\n")

    # Scenario 3
    log.info(
        "SCENARIO 3: Major bug fixing over 12 months after users reporting them, how many code changes are expected?"
    )

    log_forecast_values(
        scenario_metric_target="size",
        scenario_months=12,
        repository_full_name=REPOSITORY_FULL_NAME,
        metrics_time_series=metrics_time_series,
        df_multi_time_series=df_multi_time_series,
        metrics_pattern_hypothesis={
            "stargazers": 2,
            "contributors": 2,
            "commits": 0,
            "deployments": 2,
            "issues": 0,
            "forks": 0,
            "pull_requests": 0,
            "workflows": 2,
            "releases": 1,
        },
        phases_statistical_properties=phases_statistical_properties,
        models_path=models_path,
        phases_labels=PHASES_LABELS,
        phases_clustering_model=phases_clustering_model,
    )

    log.info("---------------------------------------------------\n")

    # Scenario 4
    log.info(
        "SCENARIO 4: Major architectural change in the codebase planned in the next 36 months, how many new issues is this expected to trigger?"
    )

    log_forecast_values(
        scenario_metric_target="issues",
        scenario_months=36,
        repository_full_name=REPOSITORY_FULL_NAME,
        metrics_time_series=metrics_time_series,
        df_multi_time_series=df_multi_time_series,
        metrics_pattern_hypothesis={
            "stargazers": 1,
            "contributors": 1,
            "commits": 0,
            "deployments": 2,
            "size": 0,
            "forks": 0,
            "pull_requests": 0,
            "workflows": 2,
            "releases": 0,
        },
        phases_statistical_properties=phases_statistical_properties,
        models_path=models_path,
        phases_labels=PHASES_LABELS,
        phases_clustering_model=phases_clustering_model,
    )

    log.info("---------------------------------------------------\n")

    # Scenario 5
    log.info(
        "SCENARIO 5: As the community of contributors is growing and the project is being forked more, how many pull requests are expected to be opened in the next 12 months?"
    )

    log_forecast_values(
        scenario_metric_target="pull_requests",
        scenario_months=12,
        repository_full_name=REPOSITORY_FULL_NAME,
        metrics_time_series=metrics_time_series,
        df_multi_time_series=df_multi_time_series,
        metrics_pattern_hypothesis={
            "stargazers": 0,
            "contributors": 0,
            "commits": 1,
            "deployments": 2,
            "size": 1,
            "forks": 0,
            "issues": 1,
            "workflows": 2,
            "releases": 2,
        },
        phases_statistical_properties=phases_statistical_properties,
        models_path=models_path,
        phases_labels=PHASES_LABELS,
        phases_clustering_model=phases_clustering_model,
    )

    log.info("---------------------------------------------------\n")
