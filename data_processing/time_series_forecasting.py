import logging
from datetime import timedelta
from math import ceil
from pathlib import Path

import pandas as pd
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from utilsforecast.plotting import plot_series
from xgboost import XGBRegressor

from connection import mo
from utils.data import (
    get_stargazers_time_series,
    get_issues_time_series,
)
from utils.time_series import (
    build_time_series,
    merge_time_series,
)

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
FORECAST_HORIZON_MONTHS = 12


if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval from Database")

    # Get the repositories in the database
    repositories = mo.db["repositories_data"].find({"statistics": {"$exists": True}})

    df_time_series_rows = []

    if not Path("../data/time_series_data.csv").exists():
        for idx, repository in enumerate(repositories):
            log.info("Analyzing repository {}".format(repository["full_name"]))

            if repository["statistics"].get("stargazers", []):
                stargazers, stargazers_cumulative = get_stargazers_time_series(
                    repository
                )
            else:
                stargazers, stargazers_cumulative = build_time_series(
                    repository, "stargazers_count"
                )

            stargazers_time_series = list(zip(stargazers, stargazers_cumulative))

            if repository["statistics"].get("issues", []):
                issues_dates, issues_cumulative = get_issues_time_series(repository)
            else:
                issues_dates, issues_cumulative = build_time_series(
                    repository, "open_issues"
                )

            issues_time_series = list(zip(issues_dates, issues_cumulative))

            commits_dates, commits_cumulative = build_time_series(
                repository, "commits"
            )

            commits_time_series = list(zip(commits_dates, commits_cumulative))

            # Combine the time series
            stargazers_time_series, issues_time_series = merge_time_series(
                stargazers_time_series, issues_time_series
            )
            stargazers_time_series, commits_time_series = merge_time_series(
                stargazers_time_series, commits_time_series
            )
            issues_time_series, commits_time_series = merge_time_series(
                issues_time_series, commits_time_series
            )

            # Sample by month
            repository_age_months = ceil(repository["age"] / 2629746)
            repository_age_start = stargazers_time_series[0][0].replace(
                day=1, hour=0, second=0, microsecond=0
            )
            time_series_idx_increment = max(
                1, int(len(stargazers_time_series) / repository_age_months)
            )
            time_series_idx = 0

            for idx_month in range(0, repository_age_months):
                repository_age_start += timedelta(days=31)
                repository_age_start = repository_age_start.replace(
                    day=1, hour=0, second=0, microsecond=0
                )

                df_time_series_rows.append(
                    {
                        "ds": repository_age_start,
                        "unique_id": repository["full_name"],
                        "repository": idx,
                        "stargazers": stargazers_time_series[time_series_idx][1],
                        "issues": issues_time_series[time_series_idx][1],
                        "commits": commits_time_series[time_series_idx][1],
                    }
                )

                if time_series_idx < len(stargazers_time_series):
                    time_series_idx += time_series_idx_increment
                else:
                    time_series_idx = -1

        # Build data frame
        df_multi_time_series = pd.DataFrame(
            df_time_series_rows,
            columns=[
                "ds",
                "unique_id",
                "repository",
                "stargazers",
                "issues",
                "commits",
            ],
        )
        df_multi_time_series.to_csv("../data/time_series_data.csv", index=False)
    else:
        # Load data frame
        df_multi_time_series = pd.read_csv(
            "../data/time_series_data.csv", parse_dates=["ds"]
        )

    log.info("Dataset loaded, prepare for model training")

    # Initialize settings
    h = FORECAST_HORIZON_MONTHS
    training_settings = {
        "stargazers": ["issues", "commits"],
        "issues": ["stargazers", "commits"],
        "commits": ["stargazers", "issues"],
    }

    for feature_target, dynamic_features in training_settings.items():
        log.info(f"Train model for {feature_target} forecasting")
        # Train for specific metric forecast
        df_time_series = df_multi_time_series.rename(columns={feature_target: "y"})

        # Split dataset
        idx_split = int(df_time_series["repository"].nunique() * 0.8)
        df_training = (
            df_time_series.loc[df_time_series["repository"] < idx_split]
            .drop(columns=["repository"])
            .reset_index(drop=True)
            .set_index(dynamic_features, append=True)
        )
        df_test = (
            df_time_series.loc[df_time_series["repository"] >= idx_split]
            .groupby("repository")
            .head(-FORECAST_HORIZON_MONTHS)
            .drop(columns=["repository"])
            .reset_index(drop=True)
            .set_index(dynamic_features, append=True)
        )
        df_validation = (
            df_time_series.loc[df_time_series["repository"] >= idx_split]
            .groupby("repository")
            .tail(FORECAST_HORIZON_MONTHS)
            .drop(columns=["repository"])
            .reset_index(drop=True)
            .set_index(dynamic_features, append=True)
        )

        # Setup the forecasting model
        log.info("Setup model")
        models = [
            # make_pipeline(
            #     SimpleImputer(),
            #     RandomForestRegressor(random_state=0, n_estimators=100)
            # ),
            XGBRegressor(random_state=0, n_estimators=100, device="cuda"),
        ]

        model = MLForecast(
            models=models,
            freq="MS",
            lags=range(1, 13),
            target_transforms=[Differences([1, 12])],
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

        # Predict data
        log.info("Make predictions on validation set")
        predictions = model.predict(
            h=h,
            new_df=df_test,
        )
        predictions = predictions.merge(
            df_validation[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left"
        )

        # Plot time frames
        fig = plot_series(df_test, predictions)
        fig.show()

        log.info("--------------")

    log.info("Multivariate time series forecasting completed!")
