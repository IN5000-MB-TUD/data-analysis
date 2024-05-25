import logging
from math import ceil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import r2_score

from connection import mo
from data_processing.time_series_forecasting import TRAINING_SETTINGS
from data_processing.time_series_phases import extrapolate_phases_properties
from data_processing.time_series_plot import create_plot
from utils.data import (
    get_stargazers_time_series,
    get_releases_time_series,
    get_size_time_series,
    get_metrics_information,
    get_metric_time_series,
)
from utils.time_series import (
    group_metric_by_month,
    group_size_by_month,
    time_series_phases,
)

# Setup logging
log = logging.getLogger(__name__)

REPOSITORY_FULL_NAME = "saltstack/salt"
REPOSITORY_CLUSTER = 0
SHOW_PLOTS = False
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
    log.info(f"Running forecast validation for repository {REPOSITORY_FULL_NAME}\n")

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
    forecast_horizon = repository_age_months - 12

    if not Path("../models/phases/mts_phases_classifier.pickle").exists():
        log.warning(
            "The phases clustering classifier model does not exists in the /models/phases folder. "
            "Run the data_processing/time_series_phases.py script to create it."
        )
        exit()
    phases_clustering_model = joblib.load(
        "../models/phases/mts_phases_classifier.pickle"
    )

    for feature_target, dynamic_features in TRAINING_SETTINGS.items():
        metric_model_path = f"{models_path}/mts_forecast_{feature_target}.pickle"

        # Check if model exists
        if not Path(metric_model_path).exists():
            log.warning(
                f"The {feature_target} forecasting model not exists in the /models/forecasting/cluster_{REPOSITORY_CLUSTER} folder."
                "Run the data_processing/time_series_forecasting.py script to create it."
            )
            continue

        log.info(f"Forecasting for {feature_target} metric...")

        # Load the model
        forecasting_model = joblib.load(metric_model_path)

        # Build the data frames
        df_time_series = df_multi_time_series.rename(
            columns={feature_target: "y"}
        ).reset_index(drop=True)

        df_forecast = forecasting_model.cross_validation(
            df_time_series,
            n_windows=forecast_horizon,
            h=1,
            step_size=1,
            refit=True,
            fitted=True,
            keep_last_n=12,
            dropna=False,
        )

        # Evaluate forecasted phases
        forecasted_metric_values = df_forecast["XGBRegressor"].tolist()
        history_metrics_values = df_time_series.head(-forecast_horizon)["y"].tolist()

        full_values = history_metrics_values + forecasted_metric_values
        full_values = [
            int(
                round(
                    x
                    * (
                        metrics_time_series[feature_target]["values"][-1]
                        - metrics_time_series[feature_target]["values"][0]
                    )
                    + metrics_time_series[feature_target]["values"][0]
                )
            )
            for x in full_values
        ]

        forecasted_metric_phases = time_series_phases(full_values)
        log.info(
            f"The forecasted phases breakpoints for the metric {feature_target} in the next {forecast_horizon} months are: {[metrics_time_series[feature_target]['dates'][i - 1] for i in forecasted_metric_phases]}"
        )

        df_forecasted_metric_phases_features = extrapolate_phases_properties(
            forecasted_metric_phases, full_values
        )

        forecasted_clustered_phases = phases_clustering_model.predict(
            df_forecasted_metric_phases_features.drop(columns=["phase_order"])
        )
        log.info(
            f"The forecasted phases for the metric {feature_target} in the next {forecast_horizon} months are: {[PHASES_LABELS[phase_id] for phase_id in forecasted_clustered_phases]}"
        )

        # Compute R2
        r2_value = r2_score(metrics_time_series[feature_target]["values"], full_values)
        log.info(f"R2: {r2_value}")

        if SHOW_PLOTS:
            log.info(f"Plotting forecasted curve for metric {feature_target}...\n")
            full_months = list(range(len(metrics_time_series[feature_target]["dates"])))

            forecast_metric_plot = create_plot(
                "Forecasted {} {}".format(
                    feature_target, repository_db_record["full_name"]
                ),
                "R2 Score: {}".format(round(r2_value, 3)),
                "Date",
                "Count",
                full_months,
                [full_values, metrics_time_series[feature_target]["values"]],
            )
            forecast_metric_plot.axvline(
                x=len(history_metrics_values),
                color="g",
                label="axvline - full height",
            )
            forecast_metric_plot.show()

        log.info("--------------------------------------------------------------\n")
