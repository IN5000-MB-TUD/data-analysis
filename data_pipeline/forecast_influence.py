import logging
from math import ceil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression

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
from utils.time_series import (
    group_metric_by_month,
    group_size_by_month,
    time_series_phases,
)

# Setup logging
log = logging.getLogger(__name__)

REPOSITORY_FULL_NAME = "SuperEvilMegacorp/vainglory-assets"
CUT_MONTH = 20
SHOW_PLOTS = True
METRICS_INFLUENCE = {
    "commits": {
        "stargazers": 1,
        "releases": 0,
        "issues": 0,
        "contributors": 0,
        "deployments": 2,
        "forks": 1,
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

    if not Path("../models/phases/mts_phases_classifier.pickle").exists():
        log.warning(
            "The phases clustering classifier model does not exists in the /models/phases folder. "
            "Run the data_processing/time_series_phases.py script to create it."
        )
        exit()
    phases_clustering_model = joblib.load(
        "../models/phases/mts_phases_classifier.pickle"
    )

    forecast_horizon = repository_age_months - CUT_MONTH

    for metric, metrics_patterns in METRICS_INFLUENCE.items():
        log.info(f"Validate influence on {metric}")

        # Estimate further growth for the other metrics
        influence_metrics_values = {}
        for influence_metric, influence_phase in metrics_patterns.items():
            if influence_phase != 2:
                poly_coefficients = [
                    phases_statistical_properties[f"phase_{influence_phase}"][
                        "coeff_3"
                    ],
                    phases_statistical_properties[f"phase_{influence_phase}"][
                        "coeff_2"
                    ],
                    phases_statistical_properties[f"phase_{influence_phase}"][
                        "coeff_1"
                    ],
                    phases_statistical_properties[f"phase_{influence_phase}"][
                        "coeff_0"
                    ],
                ]
            else:
                poly_coefficients = [0, 0, 0, 0]
            y = np.polyval(poly_coefficients, list(range(0, forecast_horizon))).tolist()
            y_normalized = normalize(y, 0, 1)

            normalized_metric_values = normalize(
                metrics_time_series[influence_metric]["values"][:CUT_MONTH], 0, 1
            )
            y_adjusted = [
                y_value + normalized_metric_values[-1] for y_value in y_normalized
            ]

            influence_metrics_values[influence_metric] = (
                normalized_metric_values[:CUT_MONTH] + y_adjusted
            )
            influence_metrics_values[influence_metric] = normalize(
                influence_metrics_values[influence_metric], 0, 1
            )

        # Balance the normalization of the target metric
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
            df_multi_time_series[influence_metric] = metric_data

        df_time_series = df_multi_time_series.rename(columns={metric: "y"}).reset_index(
            drop=True
        )

        # Split dataset
        df_training = df_time_series.head(CUT_MONTH).reset_index(drop=True)
        df_validation = (
            df_time_series.tail(forecast_horizon)
            .drop(columns=["y"])
            .reset_index(drop=True)
        )

        model = MLForecast(
            models=LinearRegression(),
            freq=1,
        )

        # Fit training data to the model
        model.fit(
            df_training,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            static_features=[],
        )
        df_forecast = model.predict(
            h=forecast_horizon,
            X_df=df_validation,
            ids=[REPOSITORY_FULL_NAME],
        )

        # Evaluate forecasted patterns
        forecasted_metric_values = df_forecast["LinearRegression"].tolist()
        full_values = (
            influence_metrics_values[metric][:CUT_MONTH] + forecasted_metric_values
        )
        full_values = normalize(full_values, 0, 1)

        metric_phases = time_series_phases(forecasted_metric_values)
        df_metric_phases_features = extrapolate_phases_properties(
            metric_phases, forecasted_metric_values
        )
        forecasted_clustered_phases = phases_clustering_model.predict(
            df_metric_phases_features.drop(columns=["phase_order"])
        )
        log.info(
            f"Forecasted pattern for metric {metric}: {[PHASES_LABELS[i] for i in forecasted_clustered_phases]}"
        )

        if SHOW_PLOTS:
            log.info(f"Plotting forecasted curve for metric {metric}...\n")
            full_months = list(range(repository_age_months))

            forecast_metric_plot = create_plot(
                "Forecasted {} {}".format(metric, REPOSITORY_FULL_NAME),
                "",
                "Date",
                "Trend",
                full_months,
                [full_values, influence_metrics_values[metric]],
            )
            forecast_metric_plot.axvline(
                x=CUT_MONTH,
                color="g",
                label="axvline - full height",
            )
            forecast_metric_plot.show()

        log.info("--------------------------------\n")
