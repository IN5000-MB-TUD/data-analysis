import json
import logging
from math import ceil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from connection import mo
from data_processing.t2f.extraction.extractor import extract_pair_series_features
from data_processing.t2f.model.clustering import ClusterWrapper
from data_processing.t2f.selection.selection import feature_selection
from utils.data import (
    get_stargazers_time_series,
    get_metric_time_series,
    get_metrics_information,
    get_releases_time_series,
    get_size_time_series,
)
from utils.main import normalize
from utils.models import train_knn_classifier
from utils.time_series import group_metric_by_month, group_size_by_month

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


if __name__ == "__main__":
    if not Path("../data/repository_metrics_phases.json").exists():
        log.warning(
            "The repository_metrics_phases.json file is not present in the data folder. Please run the time_series_phases.py script first!"
        )
        exit()

    with open("../data/repository_metrics_phases.json") as json_file:
        repository_metrics_phases = json.load(json_file)

    repositories_names = list(repository_metrics_phases.keys())

    # Retrieve phases from database
    evolution_phases = mo.db["evolution_phases"].find(
        projection={"_id": 0, "phase_name": 0}
    )
    phases_statistical_properties = {}
    for phase in evolution_phases:
        phases_statistical_properties[f"phase_{phase['phase_id']}"] = phase
        del phases_statistical_properties[f"phase_{phase['phase_id']}"]["phase_id"]

    if not Path("../data/time_series_clustering.csv").exists():
        # Get the repositories in the database
        log.info("Start GitHub statistics retrieval from Database")
        repositories = mo.db["repositories_data"].find(
            {"statistics": {"$exists": True}}
        )

        ts_phases_rows = []
        repos_matrix_pairs = []

        for idx, repository in enumerate(repositories):
            log.info("Analyzing repository {}".format(repository["full_name"]))

            # Prepare single metrics phases statistical data
            ts_phases_row_dict = {}
            for metric, metric_phases in repository_metrics_phases[
                repository["full_name"]
            ].items():
                metric_phases_rows = []
                for phase in metric_phases:
                    row_dict = {
                        f"{metric}_{key}": value
                        for key, value in phases_statistical_properties[
                            f"phase_{phase}"
                        ].items()
                    }
                    metric_phases_rows.append(row_dict)
                metric_phases_df = pd.DataFrame(metric_phases_rows)
                ts_phases_row_dict.update(**metric_phases_df.mean().to_dict())

            ts_phases_rows.append(ts_phases_row_dict)

            # Prepare metrics data for pairs distance computation
            repository_age_months = ceil(repository["age"] / 2629746)
            repository_age_start = repository["created_at"].replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )

            metrics_values_pairs = []

            # Gather metrics
            stargazers_dates, _ = get_stargazers_time_series(repository)
            stargazers_by_month = group_metric_by_month(
                stargazers_dates, repository_age_months, repository_age_start
            )
            stargazers_by_month_dates, stargazers_by_month_values = zip(
                *stargazers_by_month
            )
            stargazers_by_month_dates = list(stargazers_by_month_dates)
            stargazers_by_month_values = list(stargazers_by_month_values)
            stargazers_by_month_values = normalize(stargazers_by_month_values, 0, 1)

            metrics_values_pairs.append(
                [
                    (stargazers_by_month_dates[i], stargazers_by_month_values[i])
                    for i in range(len(stargazers_by_month_dates))
                ]
            )

            releases_dates, _ = get_releases_time_series(repository)
            releases_by_month = group_metric_by_month(
                releases_dates, repository_age_months, repository_age_start
            )
            releases_by_month_dates, releases_by_month_values = zip(*releases_by_month)
            releases_by_month_dates = list(releases_by_month_dates)
            releases_by_month_values = list(releases_by_month_values)
            releases_by_month_values = normalize(releases_by_month_values, 0, 1)

            metrics_values_pairs.append(
                [
                    (releases_by_month_dates[i], releases_by_month_values[i])
                    for i in range(len(releases_by_month_dates))
                ]
            )

            (
                repository_actions_dates,
                repository_actions_total,
                _,
            ) = get_size_time_series(repository)
            size_by_month = group_size_by_month(
                repository_actions_dates,
                repository_actions_total,
                repository_age_months,
                repository_age_start,
            )
            size_by_month_dates, size_by_month_values = zip(*size_by_month)
            size_by_month_dates = list(size_by_month_dates)
            size_by_month_values = list(size_by_month_values)
            size_by_month_values = normalize(size_by_month_values, 0, 1)

            metrics_values_pairs.append(
                [
                    (size_by_month_dates[i], size_by_month_values[i])
                    for i in range(len(size_by_month_dates))
                ]
            )

            for metric in get_metrics_information():
                metric_dates, _ = get_metric_time_series(
                    repository,
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

                metric_by_month_values = normalize(metric_by_month_values, 0, 1)
                metrics_values_pairs.append(
                    [
                        (metric_by_month_dates[i], metric_by_month_values[i])
                        for i in range(len(metric_by_month_dates))
                    ]
                )

            # Populate data frame
            repos_matrix_pairs.append(metrics_values_pairs)

        # Create single time series features data frame
        ts_phases_df = pd.DataFrame(ts_phases_rows)

        # Feature extraction
        log.info("Extracting time series pairs features...")
        ts_features_list = []
        for ts_record in repos_matrix_pairs:
            features_pair = extract_pair_series_features(ts_record)
            ts_features_list.append(features_pair)
        ts_features_df = pd.DataFrame(ts_features_list)

        # Store data
        df_feats = pd.concat([ts_phases_df, ts_features_df], axis=1)
        df_feats["id"] = repositories_names
        df_feats.to_csv("../data/time_series_clustering.csv", index=False)
    else:
        # Load data
        df_feats = pd.read_csv("../data/time_series_clustering.csv")

    log.info("Starting the clustering process")

    # Drop not needed columns
    df_feats = df_feats.drop(columns=["id"])

    # Check if model exists
    if not Path(f"../models/clustering/mts_clustering.pickle").exists():
        # Select best features
        top_features = feature_selection(df_feats)
        df_feats_dendrogram = df_feats[top_features]

        # Clustering
        transform_type = "std"  # preprocessing step
        model_type = "Hierarchical"  # clustering model

        linkage_matrix = linkage(df_feats_dendrogram.to_numpy(), method="ward")
        clustering_dendrogram = dendrogram(linkage_matrix)

        # Plot Dendrogram
        plt.title("Hierarchical Clustering Metrics Evolution Patterns Dendrogram")
        plt.xticks([])  # Hide x ticks since the axis is too crowded
        plt.show()

        # Get optimal number of clusters
        clusters = len(set(clustering_dendrogram["leaves_color_list"]))

        log.info(f"Optimal number of clusters is: {clusters}\n")

        # Save model
        model = ClusterWrapper(
            n_clusters=clusters, model_type=model_type, transform_type=transform_type
        )
        model.model.fit(df_feats)
        joblib.dump(
            model,
            f"../models/clustering/mts_clustering.pickle",
        )
    else:
        # Load existing model
        model = joblib.load(f"../models/clustering/mts_clustering.pickle")

    # Cluster repos
    clustered_repositories = model.fit_predict(df_feats)

    # Train and save classifier model
    if not Path("../models/clustering/mts_clustering_classifier.pickle").exists():
        train_knn_classifier(
            df_feats,
            clustered_repositories,
            "../models/clustering/mts_clustering_classifier.pickle",
        )

    # Display clusters sizes
    clusters_dict = {}
    for idx, cluster_id in enumerate(clustered_repositories):
        cluster_list = clusters_dict.get(f"Cluster_{cluster_id}", [])
        cluster_list.append(repositories_names[idx])
        clusters_dict[f"Cluster_{cluster_id}"] = cluster_list

    for cluster, repos in clusters_dict.items():
        log.info(f"{cluster}: {len(repos)} repositories")
        log.info("-----------------")

    # Store repositories clusters
    if not Path("../data/repository_clusters.json").exists():
        with open("../data/repository_clusters.json", "w") as outfile:
            json.dump(clusters_dict, outfile, indent=4)

    # Store clusters metrics curves coefficients
    df_clusters_coefficients = df_feats.filter(regex="_coeff_").copy()
    df_clusters_coefficients["cluster"] = clustered_repositories
    df_clusters_coefficients = df_clusters_coefficients.groupby(["cluster"]).mean()
    if not Path("../data/time_series_clustering_phases_coefficients.csv").exists():
        df_clusters_coefficients.to_csv(
            "../data/time_series_clustering_phases_coefficients.csv", index=True
        )

    # Plot clusters metrics average curves on a 5 years time
    metrics_names = [
        "stargazers",
        "releases",
        "issues",
        "commits",
        "contributors",
        "deployments",
        "forks",
        "pull",
        "workflows",
        "size",
    ]
    for idx, row in df_clusters_coefficients.iterrows():
        x = list(range(0, 13))

        for metric in metrics_names:
            metric_coefficients = row.filter(regex=metric)
            poly_coefficients = [
                metric_coefficients.iloc[0],
                metric_coefficients.iloc[1],
                metric_coefficients.iloc[2],
                metric_coefficients.iloc[3],
            ]
            y = np.polyval(poly_coefficients, x)

            # Plot values
            plt.plot(x, y, label=metric)

        # Display cluster plot
        plt.suptitle(f"Cluster {idx} Metrics Evolution Patterns")
        plt.title(f"{len(clusters_dict[f'Cluster_{idx}'])} repositories")
        plt.xlabel("Time (Months)")
        plt.ylabel("Metrics Value")
        plt.legend()
        plt.show()

    log.info("Successfully clustered repositories time series")
