import logging
from pathlib import Path

import pandas as pd
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from window_ops.rolling import rolling_mean, rolling_min, rolling_max
from xgboost import XGBRegressor

from connection import mo
from data_processing.utils import get_stargazers_time_series, build_time_series, get_issues_time_series, \
    get_additions_deletions_time_series, merge_time_series

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DATE_SPLIT = "2023-01-01"
FORECAST_HORIZON_WEEKS = 4


if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval from Database")

    # Get the repositories in the database
    repositories = mo.db["repositories_data"].find({"statistics": {"$exists": True}})

    df_time_series_rows = []

    if not Path("../data/time_series_data.csv").exists():
        for idx, repository in enumerate(repositories):
            log.info("Analyzing repository {}".format(repository["full_name"]))

            if repository["statistics"].get("stargazers", []):
                stargazers, stargazers_cumulative = get_stargazers_time_series(repository)
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

            if repository["statistics"].get("commits_weekly", []):
                (
                    commits_dates,
                    commits_cumulative,
                    _,
                    _,
                ) = get_additions_deletions_time_series(repository)
            else:
                commits_dates, commits_cumulative = build_time_series(repository, "commits")

            commits_time_series = list(zip(commits_dates, commits_cumulative))

            # Combine the time series
            stargazers_time_series, issues_time_series = merge_time_series(stargazers_time_series, issues_time_series)
            stargazers_time_series, commits_time_series = merge_time_series(stargazers_time_series, commits_time_series)
            issues_time_series, commits_time_series = merge_time_series(issues_time_series, commits_time_series)

            for time_series_idx in range(0, len(stargazers_time_series)):
                df_time_series_rows.append(
                    {
                        "ds": stargazers_time_series[time_series_idx][0].replace(microsecond=0),
                        "unique_id": idx,
                        "repository": repository["full_name"],
                        "stargazers": stargazers_time_series[time_series_idx][1],
                        "issues": issues_time_series[time_series_idx][1],
                        "commits": commits_time_series[time_series_idx][1],
                    }
                )

        # Build data frame
        df_time_series = pd.DataFrame(df_time_series_rows, columns=["ds", "unique_id", "repository", "stargazers", "issues", "commits"])
        df_time_series.to_csv("../data/time_series_data.csv", index=False)
    else:
        # Load data frame
        df_time_series = pd.read_csv("../data/time_series_data.csv", parse_dates=["ds"])

    log.info("Dataset loaded, prepare for model training")
    # Train for specific metric forecast
    df_time_series = df_time_series.rename(columns={"stargazers": "y"})
    df_time_series = df_time_series.drop(columns=["repository"])

    # Align dates and aggregate duplicate dates
    df_time_series["ds"] = df_time_series["ds"].dt.to_period("W-SAT").dt.start_time

    last_date_by_repository = df_time_series.groupby("repository")["ds"].last()
    repository_with_full_data = last_date_by_repository[last_date_by_repository == df_time_series["ds"].max()].index
    df_time_series = df_time_series.loc[df_time_series["repository"].isin(repository_with_full_data)]

    dynamic_features = ["issues", "commits"]
    static_features = []

    # Split dataset
    df_training = df_time_series.loc[df_time_series["ds"] < DATE_SPLIT]
    df_validation = df_time_series.loc[df_time_series["ds"] >= DATE_SPLIT]

    # Set forecast horizon, in weeks
    h = FORECAST_HORIZON_WEEKS

    # Setup the forecasting model
    log.info("Setup model")
    # models = [
    #     make_pipeline(
    #         SimpleImputer(),
    #         RandomForestRegressor(random_state=0, n_estimators=100)
    #     ),
    #     XGBRegressor(random_state=0, n_estimators=100, device="cuda"),
    # ]

    model = MLForecast(
        models=XGBRegressor(random_state=0, n_estimators=100, device="cuda"),
        freq="W",
        lags=[1, 2, 4],
        lag_transforms={
            1: [(rolling_mean, 4), (rolling_min, 4), (rolling_max, 4)],
        },
        date_features=["week", "month"],
        num_threads=6,
    )

    # Fit training data to the model
    log.info("Train model on training set")
    model.fit(
        df_training,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=static_features,
        max_horizon=h
    )

    # Predict data
    log.info("Make predictions on validation set")
    predictions = model.predict(
        h=h,
        new_df=df_validation,
    )
    predictions = predictions.merge(df_validation[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")
    log.info(predictions.head())
