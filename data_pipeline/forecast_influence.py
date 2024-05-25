import logging
from math import ceil
from pathlib import Path

import joblib
import pandas as pd

from connection import mo
from data_processing.time_series_phases import extrapolate_phases_properties
from data_processing.time_series_plot import create_plot
from utils.data import (
    get_metric_time_series,
    get_metrics_information,
    get_size_time_series,
    get_releases_time_series,
    get_stargazers_time_series,
)
from utils.main import normalize
from utils.pipeline import forecast_scenario_values
from utils.time_series import (
    group_metric_by_month,
    group_size_by_month,
    time_series_phases,
)

# Setup logging
log = logging.getLogger(__name__)

REPOSITORY_FULL_NAME = "saltstack/salt"
REPOSITORY_CLUSTER = 0
FORECAST_HORIZON = 24
SHOW_PLOTS = True
METRICS_INFLUENCE = {
    "stargazers": {
        "releases": 0,
        "commits": 0,
        "contributors": 1,
        "deployments": 1,
        "issues": 1,
        "forks": 2,
        "pull_requests": 2,
        "workflows": 1,
        "size": 1,
    },
    "issues": {
        "stargazers": 1,
        "releases": 1,
        "commits": 0,
        "contributors": 1,
        "deployments": 1,
        "forks": 1,
        "pull_requests": 0,
        "workflows": 1,
        "size": 0,
    },
    "commits": {
        "stargazers": 2,
        "releases": 0,
        "issues": 0,
        "contributors": 0,
        "deployments": 2,
        "forks": 2,
        "pull_requests": 0,
        "workflows": 2,
        "size": 0,
    },
}
PHASES_LABELS = ["Steep", "Shallow", "Plateau"]


if __name__ == "__main__":
    log.info(f"Running forecast influence for repository {REPOSITORY_FULL_NAME}\n")

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

    models_path = f"../models/forecasting/cluster_{REPOSITORY_CLUSTER}"

    if not Path("../models/phases/mts_phases_classifier.pickle").exists():
        log.warning(
            "The phases clustering classifier model does not exists in the /models/phases folder. "
            "Run the data_processing/time_series_phases.py script to create it."
        )
        exit()
    phases_clustering_model = joblib.load(
        "../models/phases/mts_phases_classifier.pickle"
    )

    for metric, metrics_patterns in METRICS_INFLUENCE.items():
        log.info(f"Validate influence on {metric}")

        # Estimate further growth for the other metrics
        influence_metrics_values = {}
        for influence_metric, influence_phase in metrics_patterns.items():
            metric_forecast_values = forecast_scenario_values(
                metrics_time_series[influence_metric]["values"][-1],
                [FORECAST_HORIZON],
                [influence_phase],
                phases_statistical_properties,
            )
            influence_metrics_values[influence_metric] = (
                metrics_time_series[influence_metric]["values"] + metric_forecast_values
            )
            influence_metrics_values[influence_metric] = normalize(
                influence_metrics_values[influence_metric], 0, 1
            )

        # Balance the normalization of the target metric
        scenario_ds = [repository_age_months + x for x in range(0, FORECAST_HORIZON)]
        normalization_ratio = repository_age_months / scenario_ds[-1]
        influence_metrics_values[metric] = normalize(
            metrics_time_series[metric]["values"], 0, 1
        )

        # Build data frame
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
        df_multi_time_series["unique_id"] = [
            REPOSITORY_FULL_NAME
        ] * repository_age_months
        for influence_metric, metric_data in influence_metrics_values.items():
            df_multi_time_series[influence_metric] = metric_data[:repository_age_months]
        df_time_series = df_multi_time_series.rename(columns={metric: "y"}).reset_index(
            drop=True
        )

        df_hypothesis = {
            "ds": scenario_ds,
            "unique_id": [REPOSITORY_FULL_NAME] * FORECAST_HORIZON,
        }
        for influence_metric, metric_data in influence_metrics_values.items():
            if influence_metric != metric:
                df_hypothesis[influence_metric] = metric_data[repository_age_months:]
        df_hypothesis = pd.DataFrame(df_hypothesis)

        # Forecast target metric
        df_forecast_input = df_time_series._append(
            df_hypothesis, ignore_index=True
        ).drop(columns=["y"])

        forecasting_model = joblib.load(f"{models_path}/mts_forecast_{metric}.pickle")
        df_forecast = (
            forecasting_model.predict(
                FORECAST_HORIZON + 12,
                ids=[REPOSITORY_FULL_NAME],
                X_df=df_forecast_input,
            )
            .tail(FORECAST_HORIZON)
            .reset_index(drop=True)
        )

        # Evaluate forecasted patterns
        forecasted_metric_values = df_forecast["XGBRegressor"].tolist()
        forecasted_metric_phases = time_series_phases(forecasted_metric_values)
        df_forecasted_metric_phases_features = extrapolate_phases_properties(
            forecasted_metric_phases, forecasted_metric_values
        )

        forecasted_clustered_phases = phases_clustering_model.predict(
            df_forecasted_metric_phases_features.drop(columns=["phase_order"])
        )

        log.info(
            f"Forecasted pattern for metric {metric}: {[PHASES_LABELS[i] for i in forecasted_clustered_phases]}"
        )

        if SHOW_PLOTS:
            log.info(f"Plotting forecasted curve for metric {metric}...\n")
            full_months = list(range(repository_age_months)) + [
                repository_age_months + x for x in range(1, FORECAST_HORIZON + 1)
            ]

            history_metrics_values = df_time_series["y"].tolist()
            full_values = history_metrics_values + forecasted_metric_values

            forecast_metric_plot = create_plot(
                "Forecasted {} {}".format(metric, REPOSITORY_FULL_NAME),
                "",
                "Date",
                "Trend",
                full_months,
                [full_values],
            )
            forecast_metric_plot.axvline(
                x=repository_age_months - 1,
                color="g",
                label="axvline - full height",
            )
            forecast_metric_plot.show()

        log.info("--------------------------------\n")
