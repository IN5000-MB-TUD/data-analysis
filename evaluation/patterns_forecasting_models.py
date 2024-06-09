import json
import logging
from math import ceil
from pathlib import Path

import joblib
import pandas as pd
from mlforecast import MLForecast
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from connection import mo
from data_processing.time_series_phases import extrapolate_phases_properties
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

REPOSITORIES = [
    "saltstack/salt",
    "patternfly/patternfly-react",
    "conan-io/conan",
    "nextcloud/server",
    "ansible/awx",
    "pypa/pip",
    "cockroachdb/cockroach",
    "NixOS/nixpkgs",
    "WordPress/gutenberg",
    "woocommerce/woocommerce",
]
REPOSITORY_CLUSTER = 0
N_MONTHS = [2, 4, 6, 12, 24, 48, 72]
PHASES_LABELS = ["Steep", "Shallow", "Plateau"]
METRICS = [
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
]


# Define helpers
def _compute_monthly_patterns(metric_patterns, metric_patterns_idxs):
    """Return a sequence of each monthly pattern"""
    metric_patterns_sequence = []
    patterns_boundaries = list(zip(metric_patterns_idxs[:-1], metric_patterns_idxs[1:]))
    for pattern_idx, pattern in enumerate(metric_patterns):
        metric_patterns_sequence += [pattern] * (
            patterns_boundaries[pattern_idx][1] - patterns_boundaries[pattern_idx][0]
        )
    return metric_patterns_sequence


if __name__ == "__main__":
    if not Path("../data/n_patterns_forecast_accuracy.json").exists():
        n_months_accuracy = {}
    else:
        with open("../data/n_patterns_forecast_accuracy.json") as json_file:
            n_months_accuracy = json.load(json_file)

    for repository_full_name in REPOSITORIES:
        log.info(
            f"Evaluate N-months predictions to find the most suitable N for repository: {repository_full_name}\n"
        )

        repository_db_record = mo.db["repositories_data"].find_one(
            {"full_name": repository_full_name}
        )
        if not repository_db_record:
            log.info(
                f"No record found for repository {repository_full_name}, run the pipeline script to import the data"
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
        stargazers_by_month_dates, stargazers_by_month_values = zip(
            *stargazers_by_month
        )
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
            repository_full_name
        ] * repository_age_months
        for metric, metric_data in metrics_time_series.items():
            df_multi_time_series[metric] = normalize(metric_data["values"], 0, 1)

        # Open full data and remove data from the analyzed repo
        df_repos_time_series = pd.read_csv("../data/time_series_data.csv")
        df_repos_time_series["ds"] = pd.to_numeric(
            df_repos_time_series["ds"], downcast="integer"
        )
        df_repos_time_series = df_repos_time_series.loc[
            df_repos_time_series["cluster"] == REPOSITORY_CLUSTER
        ]
        df_repos_time_series = df_repos_time_series.drop(columns=["cluster"])
        df_repos_time_series = df_repos_time_series.loc[
            df_repos_time_series["unique_id"] != repository_full_name
        ]
        metrics_columns = df_repos_time_series.columns.difference(["ds", "unique_id"])
        df_repos_time_series[metrics_columns] = (
            df_repos_time_series[metrics_columns]
            .groupby("repository")[metrics_columns]
            .apply(lambda v: (v - v.min()) / (v.max() - v.min()))
            .reset_index(level=0, drop=True)
        )
        df_repos_time_series = df_repos_time_series.fillna(0)
        df_repos_time_series = df_repos_time_series.drop(
            columns=["repository"]
        ).reset_index(drop=True)

        if not Path("../models/phases/mts_phases_classifier.pickle").exists():
            log.warning(
                "The phases clustering classifier model does not exists in the /models/phases folder. "
                "Run the data_processing/time_series_phases.py script to create it."
            )
            exit()
        phases_clustering_model = joblib.load(
            "../models/phases/mts_phases_classifier.pickle"
        )

        with open("../data/repository_metrics_phases.json") as json_file:
            repository_metrics_phases = json.load(json_file)
        with open("../data/repository_metrics_phases_idxs.json") as json_file:
            repository_metrics_phases_idxs = json.load(json_file)

        repository_patterns = repository_metrics_phases[repository_full_name]
        repository_patterns_idxs = repository_metrics_phases_idxs[repository_full_name]

        n_months_accuracy[repository_full_name] = {}
        for n in N_MONTHS:
            log.info(f"Evaluating {n} months training")
            total_predictions = 0
            correct_predictions = 0
            deviation_predictions = 0
            r2_average = 0

            cut_month = n
            forecast_horizon = repository_age_months - cut_month

            for target_metric in METRICS:
                df_time_series = df_multi_time_series.rename(
                    columns={target_metric: "y"}
                ).reset_index(drop=True)

                # Split dataset
                df_training = df_time_series.head(cut_month).reset_index(drop=True)
                df_validation = (
                    df_time_series.tail(forecast_horizon)
                    .drop(columns=["y"])
                    .reset_index(drop=True)
                )

                # Append training df to full data
                df_repos_time_series_training = df_repos_time_series.rename(
                    columns={target_metric: "y"}
                ).reset_index(drop=True)
                df_training = df_repos_time_series_training._append(
                    df_training, ignore_index=True
                )

                model = MLForecast(
                    models=XGBRegressor(
                        random_state=0, n_estimators=100, device="cuda"
                    ),
                    freq=1,
                )

                # Fit training data to the model
                model.fit(
                    df_training,
                    id_col="unique_id",
                    time_col="ds",
                    target_col="y",
                    static_features=[],
                    keep_last_n=n,
                )
                df_forecast = model.predict(
                    h=forecast_horizon,
                    X_df=df_validation,
                    ids=[repository_full_name],
                )

                # Evaluate forecasted patterns
                forecasted_metric_values = df_forecast["XGBRegressor"].tolist()
                forecasted_metric_values = normalize(
                    forecasted_metric_values,
                    metrics_time_series[target_metric]["values"][cut_month:][0],
                    metrics_time_series[target_metric]["values"][cut_month:][-1],
                )
                forecasted_metric_values = [
                    int(round(x, 0)) for x in forecasted_metric_values
                ]
                full_values = (
                    metrics_time_series[target_metric]["values"][:cut_month]
                    + forecasted_metric_values
                )

                metric_phases = time_series_phases(full_values)
                df_metric_phases_features = extrapolate_phases_properties(
                    metric_phases, full_values
                )
                forecasted_clustered_phases = phases_clustering_model.predict(
                    df_metric_phases_features.drop(columns=["phase_order"])
                )

                # Count correct predictions
                actual_patterns_sequence = _compute_monthly_patterns(
                    repository_patterns[target_metric],
                    [0] + repository_patterns_idxs[target_metric],
                )
                forecasted_patterns_sequence = _compute_monthly_patterns(
                    forecasted_clustered_phases, [0] + metric_phases
                )

                for idx, pred_values in enumerate(forecasted_patterns_sequence):
                    # Increase predictions counter
                    total_predictions += 1
                    # Count correct predictions
                    if pred_values == actual_patterns_sequence[idx]:
                        correct_predictions += 1
                    # Compute deviation
                    deviation_predictions += abs(
                        pred_values - actual_patterns_sequence[idx]
                    )

                # Compute scores
                r2_value = r2_score(
                    metrics_time_series[target_metric]["values"], full_values
                )

                r2_average += r2_value

                log.info(target_metric)
                log.info(f"R2: {r2_value}")
                log.info("-----------------")

            if total_predictions > 0:
                n_months_accuracy[repository_full_name][f"{n}_months"] = {
                    "correct_predictions": correct_predictions,
                    "total_predictions": total_predictions,
                    "performance": correct_predictions / total_predictions,
                    "deviation": deviation_predictions / total_predictions,
                    "r2": r2_average / len(METRICS),
                }

            log.info("----------------------------------------------\n")

        log.info(n_months_accuracy[repository_full_name])

        best_accuracy = -1
        best_n = ""
        for n_key, n_items in n_months_accuracy[repository_full_name].items():
            if n_items["performance"] > best_accuracy:
                best_accuracy = n_items["performance"]
                best_n = n_key
        log.info(f"Best N is {best_n} with accuracy {round(best_accuracy, 3)}")

    # Compute average scores
    average_scores = {}
    for repository, months_values in n_months_accuracy.items():
        for month_key, month_val in months_values.items():
            if month_key not in average_scores:
                average_scores[month_key] = {
                    "performance": 0,
                    "deviation": 0,
                    "r2": 0,
                }
            average_scores[month_key]["performance"] += month_val["performance"]
            average_scores[month_key]["deviation"] += month_val["deviation"]
            average_scores[month_key]["r2"] += min(1, max(0, month_val["r2"]))

    total_projects = len(REPOSITORIES)
    for month_key in average_scores.keys():
        average_scores[month_key]["performance"] /= total_projects
        average_scores[month_key]["deviation"] /= total_projects
        average_scores[month_key]["r2"] /= total_projects

    n_months_accuracy["average_scores"] = average_scores

    with open("../data/n_patterns_forecast_accuracy.json", "w") as outfile:
        json.dump(n_months_accuracy, outfile, indent=4)

    log.info("Done!")
