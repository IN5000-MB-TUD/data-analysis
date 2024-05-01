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
from data_processing.time_series_forecasting import TRAINING_SETTINGS
from data_processing.time_series_phases import extrapolate_phases_properties
from data_processing.time_series_plot import create_plot
from utils.data import (
    get_stargazers_time_series,
    get_metrics_information,
    get_metric_time_series,
    get_releases_time_series,
    get_size_time_series,
)
from utils.main import normalize
from utils.time_series import (
    group_metric_by_month,
    time_series_phases,
    group_size_by_month,
    merge_time_series_patterns,
)

# Setup logging
log = logging.getLogger(__name__)

REPOSITORY_FULL_NAME = "saltstack/salt"
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
CLUSTERS_LABELS = ["Steep", "Semi-Shallow", "Shallow"]
PATTERNS_WEIGHTS = [0.5, 0.35, 0.15]


if __name__ == "__main__":
    log.info(f"Running pipeline for repository {REPOSITORY_FULL_NAME}\n")

    # STEP 1: DATA COLLECTION
    log.info("STEP 1: DATA COLLECTION")

    # Get all repos names
    repository_names = list(
        mo.db["repositories_data"].find(projection={"full_name": 1})
    )

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

    # Plot metrics curves
    if SHOW_PLOTS:
        for metric, metric_data in metrics_time_series.items():
            log.info(f"Plotting metric {metric} curve")
            metric_plot = create_plot(
                "{} {}".format(metric, repository_db_record["full_name"]),
                "Total: {}".format(metric_data["values"][-1]),
                "Date",
                "Count",
                metric_data["dates"],
                [metric_data["values"]],
            )
            metric_plot.show()

    log.info("---------------------------------------------------\n")

    # STEP 3: PHASES CLUSTERING
    log.info("STEP 3: PHASES CLUSTERING")

    log.info("Computing metrics phases...")
    metrics_phases = {}
    for metric, metric_data in metrics_time_series.items():
        metric_phases_idxs = time_series_phases(
            metric_data["values"],
            show_plot=SHOW_PLOTS,
            plot_title=f"{REPOSITORY_FULL_NAME} {metric}",
        )
        metrics_phases[metric] = {
            "phases": metric_phases_idxs,
            "phases_count": len(metric_phases_idxs),
            "phases_dates": [metric_data["dates"][i - 1] for i in metric_phases_idxs],
        }

        log.info(
            f"{metric} {metrics_phases[metric]['phases_count']} phases breakpoints: {metrics_phases[metric]['phases']}, {metrics_phases[metric]['phases_dates']}"
        )

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
    if not Path("../models/phases/mts_phases_classifier.pickle").exists():
        log.warning(
            "The phases clustering classifier model does not exists in the /models/phases folder. "
            "Run the data_processing/time_series_phases.py script to create it."
        )
        exit()
    phases_clustering_model = joblib.load(
        "../models/phases/mts_phases_classifier.pickle"
    )

    clustered_phases = phases_clustering_model.predict(df_phases)
    phases_features["phase_cluster"] = clustered_phases

    log.info("Building metrics phases sequences...")
    phases_rows_counter = 0
    for metric, metric_phases_data in metrics_phases.items():
        metrics_phases[metric]["phases_sequence"] = []
        metrics_phases[metric]["phases_sequence_label"] = []
        for i in range(0, metric_phases_data["phases_count"]):
            phase_id = phases_features["phase_cluster"][phases_rows_counter + i]
            metrics_phases[metric]["phases_sequence"].append(phase_id)
            metrics_phases[metric]["phases_sequence_label"].append(
                PHASES_LABELS[phase_id]
            )

        phases_rows_counter += metric_phases_data["phases_count"]

        log_sequences = list(
            zip(
                [
                    segment_date.strftime("%d/%m/%Y")
                    for segment_date in metrics_phases[metric]["phases_dates"]
                ],
                metrics_phases[metric]["phases_sequence_label"],
            )
        )

        log.info(f"Phases sequence for metric {metric}: {log_sequences}")

    # Build unique representation curve for repository evolution history
    phases_idxs = set()

    for metric, metric_phases_data in metrics_phases.items():
        phases_idxs.update(metric_phases_data["phases"])
    phases_idxs = list(phases_idxs)
    phases_idxs.sort()

    metrics_phases_aligned = {}
    for metric, metric_stats in metrics_phases.items():
        metrics_phases_aligned[metric] = []
        pattern_idx = 0
        for phase_idx in phases_idxs:
            if phase_idx > metric_stats["phases"][pattern_idx]:
                pattern_idx += 1

            metrics_phases_aligned[metric].append(
                metric_stats["phases_sequence"][pattern_idx]
            )

    phases_idxs = [0] + phases_idxs
    phases_bounds = list(zip(phases_idxs[:-1], phases_idxs[1:]))

    unique_time_series = merge_time_series_patterns(
        METRICS,
        PATTERNS_WEIGHTS,
        metrics_time_series,
        metrics_phases_aligned,
        phases_bounds,
    )

    if SHOW_PLOTS:
        log.info(f"Plotting metric merged curve")
        metric_plot = create_plot(
            "Merged Time Series",
            "",
            "Month",
            "Trend",
            list(range(len(unique_time_series))),
            [unique_time_series],
        )

        for phase_idx in phases_idxs[1:-1]:
            metric_plot.axvline(
                x=phase_idx,
                color="g",
            )

        metric_plot.show()

    # Map merged time series to phases
    unique_time_series_features = extrapolate_phases_properties(
        phases_idxs[1:], unique_time_series
    )
    df_unique_time_series_phases = unique_time_series_features.drop(
        columns=["phase_order"]
    )
    unique_time_series_phases = phases_clustering_model.predict(
        df_unique_time_series_phases
    )

    merged_phases_sequence = [
        PHASES_LABELS[pattern_id] for pattern_id in unique_time_series_phases
    ]
    log.info(
        f"Merged phases sequence for {REPOSITORY_FULL_NAME}: {merged_phases_sequence}"
    )

    log.info("---------------------------------------------------\n")

    # STEP 4: REPOSITORY CLUSTERING
    log.info("STEP 4: REPOSITORY CLUSTERING")

    # Retrieve phases from database
    evolution_phases = mo.db["evolution_phases"].find(
        projection={"_id": 0, "phase_name": 0}
    )
    phases_statistical_properties = {}
    for phase in evolution_phases:
        phases_statistical_properties[f"phase_{phase['phase_id']}"] = phase
        del phases_statistical_properties[f"phase_{phase['phase_id']}"]["phase_id"]

    log.info("Building cluster vector...")

    log.info("Process metrics phases sequences...")
    metrics_phases_sequence = {}

    for metric in METRICS:
        metric_phases_data = metrics_phases[metric]
        metric_phases_rows = []
        for phase in metric_phases_data["phases_sequence"]:
            row_dict = {
                f"{metric}_{key}": value
                for key, value in phases_statistical_properties[
                    f"phase_{phase}"
                ].items()
            }
            metric_phases_rows.append(row_dict)
        metric_phases_df = pd.DataFrame(metric_phases_rows)
        metrics_phases_sequence.update(**metric_phases_df.mean().to_dict())

    df_repository_phases_clustering = pd.DataFrame([metrics_phases_sequence])

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
    if not Path(f"../models/clustering/mts_clustering_classifier.pickle").exists():
        log.warning(
            "The repositories clustering classifier model does not exists in the /models/clustering folder."
            "Run the data_processing/time_series_clustering.py script to create it."
        )
        exit()
    repos_clustering_model = joblib.load(
        "../models/clustering/mts_clustering_classifier.pickle"
    )

    clustered_repository = repos_clustering_model.predict(df_feats)
    cluster_id = clustered_repository[0]
    log.info(f"The repository was assigned to cluster {CLUSTERS_LABELS[cluster_id]}")

    # Get most similar projects
    log.info("Most similar projects:")
    nearest_projects = repos_clustering_model.kneighbors(
        df_feats, return_distance=False
    )
    for point_results in nearest_projects:
        for projects_idx in point_results:
            project_name = repository_names[projects_idx]["full_name"]
            if project_name != REPOSITORY_FULL_NAME:
                log.info(project_name)

    log.info("---------------------------------------------------\n")

    # STEP 5: METRICS FORECASTING
    log.info("STEP 5: METRICS FORECASTING")

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

    # Set the months to forecast
    forecast_horizon = 12
    models_path = f"../models/forecasting/cluster_{cluster_id}"

    for feature_target, dynamic_features in TRAINING_SETTINGS.items():
        metric_model_path = f"{models_path}/mts_forecast_{feature_target}.pickle"

        # Check if model exists
        if not Path(metric_model_path).exists():
            log.warning(
                f"The {feature_target} forecasting model not exists in the /models/forecasting/cluster_{cluster_id} folder."
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

        df_forecast = forecasting_model.predict(
            forecast_horizon, ids=[REPOSITORY_FULL_NAME], X_df=df_time_series
        )

        # Replay history
        backwards_forecast_months = forecast_horizon * 2
        df_replay = forecasting_model.cross_validation(
            df_time_series,
            n_windows=backwards_forecast_months,
            h=1,
            step_size=1,
            refit=True,
            fitted=True,
        )

        # Evaluate forecasted phases
        history_metrics_values = df_time_series.head(-forecast_horizon)["y"].tolist()
        forecasted_metric_values = df_forecast["XGBRegressor"].tolist()
        forecasted_metric_phases = time_series_phases(forecasted_metric_values)
        df_forecasted_metric_phases_features = extrapolate_phases_properties(
            forecasted_metric_phases, forecasted_metric_values
        )

        forecasted_clustered_phases = phases_clustering_model.predict(
            df_forecasted_metric_phases_features.drop(columns=["phase_order"])
        )
        log.info(
            f"The forecasted phases for the metric {feature_target} in the next {forecast_horizon} months are: {[PHASES_LABELS[phase_id] for phase_id in forecasted_clustered_phases]}"
        )

        if SHOW_PLOTS:
            log.info(f"Plotting forecasted curve for metric {feature_target}...\n")
            full_months = list(range(len(metrics_time_series[feature_target]["dates"])))
            full_values = history_metrics_values + forecasted_metric_values
            forecast_metric_plot = create_plot(
                "Forecasted {} {}".format(
                    feature_target, repository_db_record["full_name"]
                ),
                "Total: {} -> {}".format(
                    round(history_metrics_values[-1], 4), round(full_values[-1], 4)
                ),
                "Date",
                "Count",
                full_months,
                [full_values, df_time_series["y"].tolist()],
            )
            forecast_metric_plot.axvline(
                x=len(history_metrics_values),
                color="g",
                label="axvline - full height",
            )
            forecast_metric_plot.show()

    log.info("---------------------------------------------------\n")
