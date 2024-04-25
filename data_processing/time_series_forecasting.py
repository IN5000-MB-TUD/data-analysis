import json
import logging
from math import ceil
from pathlib import Path

import joblib
import pandas as pd
from mlforecast import MLForecast
from utilsforecast.plotting import plot_series
from utilsforecast.losses import mse
from xgboost import XGBRegressor

from connection import mo
from utils.data import (
    get_stargazers_time_series,
    get_metric_time_series,
    get_releases_time_series,
    get_size_time_series,
)
from utils.time_series import group_metric_by_month, group_size_by_month

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
FORECAST_HORIZON_MONTHS = 12
MINIMUM_AGE_MONTHS = FORECAST_HORIZON_MONTHS * 2
TRAINING_SETTINGS = {
    "stargazers": [
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
    "issues": [
        "stargazers",
        "commits",
        "contributors",
        "deployments",
        "forks",
        "pull_requests",
        "workflows",
        "releases",
        "size",
    ],
    "commits": [
        "stargazers",
        "issues",
        "contributors",
        "deployments",
        "forks",
        "pull_requests",
        "workflows",
        "releases",
        "size",
    ],
    "contributors": [
        "stargazers",
        "issues",
        "commits",
        "deployments",
        "forks",
        "pull_requests",
        "workflows",
        "releases",
        "size",
    ],
    "deployments": [
        "stargazers",
        "issues",
        "commits",
        "contributors",
        "forks",
        "pull_requests",
        "workflows",
        "releases",
        "size",
    ],
    "forks": [
        "stargazers",
        "issues",
        "commits",
        "contributors",
        "deployments",
        "pull_requests",
        "workflows",
        "releases",
        "size",
    ],
    "pull_requests": [
        "stargazers",
        "issues",
        "commits",
        "contributors",
        "deployments",
        "forks",
        "workflows",
        "releases",
        "size",
    ],
    "workflows": [
        "stargazers",
        "issues",
        "commits",
        "contributors",
        "deployments",
        "forks",
        "pull_requests",
        "releases",
        "size",
    ],
    "releases": [
        "stargazers",
        "issues",
        "commits",
        "contributors",
        "deployments",
        "forks",
        "pull_requests",
        "workflows",
        "size",
    ],
    "size": [
        "stargazers",
        "issues",
        "commits",
        "contributors",
        "deployments",
        "forks",
        "pull_requests",
        "workflows",
        "releases",
    ],
}


if __name__ == "__main__":
    log.info("Start repositories metrics forecasting models training")

    if not Path("../data/time_series_data.csv").exists():
        # Get cluster information for each repository
        if not Path("../data/repository_clusters.json").exists():
            log.warning(
                "The repository_clusters.json file is not present in the data folder. Please run the time_series_clustering.py script first!"
            )
            exit()

        with open("../data/repository_clusters.json") as json_file:
            repository_clusters = json.load(json_file)

        repos_by_cluster = {}
        for cluster, repos in repository_clusters.items():
            cluster_id = int(cluster.split("_")[-1])
            repos_by_cluster.update(**{repo_name: cluster_id for repo_name in repos})

        log.info("Start GitHub statistics retrieval from Database")

        # Get the repositories in the database
        repositories = mo.db["repositories_data"].find(
            {"statistics": {"$exists": True}}
        )

        df_time_series_rows = []

        for idx, repository in enumerate(repositories):
            log.info("Analyzing repository {}".format(repository["full_name"]))

            repository_age_months = ceil(repository["age"] / 2629746)
            if repository_age_months < MINIMUM_AGE_MONTHS:
                log.warning(
                    "At least {} months of age are needed for the forecasting, excluding {} since it's {} moths old".format(
                        MINIMUM_AGE_MONTHS,
                        repository["full_name"],
                        repository_age_months,
                    )
                )
                continue

            repository_age_start = repository["created_at"].replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )

            stargazers_dates, _ = get_stargazers_time_series(repository)
            stargazers_time_series = group_metric_by_month(
                stargazers_dates, repository_age_months, repository_age_start
            )

            issues_dates, _ = get_metric_time_series(
                repository, "statistics_issues", "issues", "created_at", "open_issues"
            )
            issues_time_series = group_metric_by_month(
                issues_dates, repository_age_months, repository_age_start
            )

            commits_dates, _ = get_metric_time_series(
                repository, "statistics_commits", "commits", "date", "commits"
            )
            commits_time_series = group_metric_by_month(
                commits_dates, repository_age_months, repository_age_start
            )

            contributors_dates, _ = get_metric_time_series(
                repository, "statistics_commits", "contributors", "first_commit", None
            )
            contributors_time_series = group_metric_by_month(
                contributors_dates, repository_age_months, repository_age_start
            )

            deployments_dates, _ = get_metric_time_series(
                repository, "statistics_deployments", "deployments", "created_at", None
            )
            deployments_time_series = group_metric_by_month(
                deployments_dates, repository_age_months, repository_age_start
            )

            forks_dates, _ = get_metric_time_series(
                repository, "statistics_forks", "forks", "created_at", "forks_count"
            )
            forks_time_series = group_metric_by_month(
                forks_dates, repository_age_months, repository_age_start
            )

            pull_requests_dates, _ = get_metric_time_series(
                repository,
                "statistics_pull_requests",
                "pull_requests",
                "created_at",
                None,
            )
            pull_requests_time_series = group_metric_by_month(
                pull_requests_dates, repository_age_months, repository_age_start
            )

            workflows_dates, _ = get_metric_time_series(
                repository, "statistics_workflow_runs", "workflows", "created_at", None
            )
            workflows_time_series = group_metric_by_month(
                workflows_dates, repository_age_months, repository_age_start
            )

            releases_dates, _ = get_releases_time_series(repository)
            releases_time_series = group_metric_by_month(
                releases_dates, repository_age_months, repository_age_start
            )

            (
                repository_actions_dates,
                repository_actions_total,
                _,
            ) = get_size_time_series(repository)
            size_time_series = group_size_by_month(
                repository_actions_dates,
                repository_actions_total,
                repository_age_months,
                repository_age_start,
            )

            for idx_month in range(0, repository_age_months):
                df_time_series_rows.append(
                    {
                        "ds": idx_month,
                        "unique_id": repository["full_name"],
                        "cluster": repos_by_cluster[repository["full_name"]],
                        "repository": idx,
                        "stargazers": stargazers_time_series[idx_month][1],
                        "issues": issues_time_series[idx_month][1],
                        "commits": commits_time_series[idx_month][1],
                        "contributors": contributors_time_series[idx_month][1],
                        "deployments": deployments_time_series[idx_month][1],
                        "forks": forks_time_series[idx_month][1],
                        "pull_requests": pull_requests_time_series[idx_month][1],
                        "workflows": workflows_time_series[idx_month][1],
                        "releases": releases_time_series[idx_month][1],
                        "size": size_time_series[idx_month][1],
                    }
                )

        # Build data frame
        df_multi_time_series = pd.DataFrame(
            df_time_series_rows,
            columns=[
                "ds",
                "unique_id",
                "cluster",
                "repository",
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
        df_multi_time_series.to_csv("../data/time_series_data.csv", index=False)
    else:
        # Load data frame
        df_multi_time_series = pd.read_csv(
            "../data/time_series_data.csv", parse_dates=["ds"]
        )

    log.info("Dataset loaded, prepare for model training")
    df_multi_time_series["ds"] = pd.to_numeric(
        df_multi_time_series["ds"], downcast="integer"
    )

    # Initialize settings
    h = FORECAST_HORIZON_MONTHS
    n_clusters = df_multi_time_series["cluster"].max() + 1
    cluster_features_importance = {}

    # Train models by cluster group
    for cluster_id in range(n_clusters):
        log.info(f"Train models for cluster {cluster_id}...\n")

        models_path = f"../models/forecasting/cluster_{cluster_id}"
        Path(models_path).mkdir(parents=True, exist_ok=True)

        cluster_features_importance[f"cluster_{cluster_id}"] = {}

        for feature_target, dynamic_features in TRAINING_SETTINGS.items():
            log.info(f"Train model for {feature_target} forecasting")
            # Train for specific metric forecast
            df_time_series = df_multi_time_series.loc[
                df_multi_time_series["cluster"] == cluster_id
            ].rename(columns={feature_target: "y"})
            df_time_series = df_time_series.drop(columns=["cluster"])

            # Scale values between min and max
            metrics_columns = df_time_series.columns.difference(["ds", "unique_id"])
            df_time_series[metrics_columns] = (
                df_time_series[metrics_columns]
                .groupby("repository")[metrics_columns]
                .apply(lambda v: (v - v.min()) / (v.max() - v.min()))
                .reset_index(level=0, drop=True)
            )
            df_time_series = df_time_series.fillna(0)
            df_time_series["repository"] = df_multi_time_series.loc[
                df_multi_time_series["cluster"] == cluster_id
            ]["repository"].copy()

            metric_model_path = f"{models_path}/mts_forecast_{feature_target}.pickle"

            # Split dataset
            df_training = (
                df_time_series.groupby("repository")
                .head(-FORECAST_HORIZON_MONTHS)
                .drop(columns=["repository"])
                .reset_index(drop=True)
            )
            df_validation = (
                df_time_series.groupby("repository")
                .tail(FORECAST_HORIZON_MONTHS)
                .drop(columns=["repository"])
                .reset_index(drop=True)
            )

            # Setup the forecasting model
            log.info("Setup model")

            # Check if model exists
            if not Path(metric_model_path).exists():
                models = [
                    XGBRegressor(random_state=0, n_estimators=100, device="cuda"),
                ]

                model = MLForecast(
                    models=models,
                    freq=1,
                    lags=[1, FORECAST_HORIZON_MONTHS],
                )

                # Fit training data to the model
                log.info("Train model on training set")
                model.fit(
                    df_training,
                    id_col="unique_id",
                    time_col="ds",
                    target_col="y",
                    static_features=[],
                )

                # Save the model
                joblib.dump(
                    model,
                    metric_model_path,
                )
            else:
                # Load the model
                model = joblib.load(metric_model_path)

            # Predict data
            log.info("Make predictions on validation set")
            predictions = model.predict(
                h=h,
                X_df=df_validation.drop(columns=["y"]),
            )
            predictions = predictions.merge(
                df_validation[["unique_id", "ds", "y"]],
                on=["unique_id", "ds"],
                how="left",
            )

            # Print MSE
            model_mse_df = mse(
                predictions, models=["XGBRegressor"], id_col="unique_id", target_col="y"
            )
            model_mse = model_mse_df["XGBRegressor"].mean()
            log.info(f"MSE: {model_mse}")

            # Plot time frames
            fig = plot_series(df_training, predictions)
            fig.show()

            # Store the features importance of the model
            cluster_features_importance[f"cluster_{cluster_id}"][feature_target] = (
                pd.Series(
                    model.models_["XGBRegressor"].feature_importances_,
                    index=model.ts.features_order_,
                )
                .sort_values(ascending=False)
                .to_dict()
            )

            log.info("--------------")

        log.info("----------------------------\n")

    # Process and print the features importance per cluster
    cluster_metrics_importance = {}
    for cluster, models in cluster_features_importance.items():
        cluster_metrics_importance[cluster] = {}
        for model, features in models.items():
            filtered_dict = {
                key: value for key, value in features.items() if "lag" not in key
            }
            normalize_factor = sum(filtered_dict.values())
            if normalize_factor == 0:
                normalize_factor = 1 / len(filtered_dict)
            else:
                normalize_factor = 1 / normalize_factor

            for feature_key, feature_value in filtered_dict.items():
                cluster_metrics_importance[cluster][feature_key] = (
                    feature_value * normalize_factor
                )

    log.info(cluster_metrics_importance)

    log.info("Multivariate time series forecasting completed!")
