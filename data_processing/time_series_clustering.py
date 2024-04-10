import json
import logging
from math import ceil
from pathlib import Path

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from connection import mo
from data_processing.t2f.extraction.extractor import extract_pair_series_features
from data_processing.t2f.model.clustering import ClusterWrapper
from utils.data import (
    get_stargazers_time_series,
    get_metric_time_series,
    get_metrics_information, get_releases_time_series,
)
from utils.main import normalize, proper_round
from utils.models import train_knn_classifier
from utils.time_series import group_metric_by_month

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


if __name__ == "__main__":
    if not Path("../data/time_series_clustering_phases.csv").exists():
        log.warning(
            "The phases sequence file is not present in the data folder. Please run the time_series_phases.py script first!"
        )
        exit()

    # Load data
    df_repository_metrics_phases = pd.read_csv(
        "../data/time_series_clustering_phases.csv"
    )
    repositories_names = df_repository_metrics_phases["id"].tolist()

    log.info("Start GitHub statistics retrieval from Database")

    # Get the repositories in the database
    repositories = mo.db["repositories_data"].find({"statistics": {"$exists": True}})

    repos_matrix_pairs = []

    if not Path("../data/time_series_clustering_distances.csv").exists():
        for idx, repository in enumerate(repositories):
            log.info("Analyzing repository {}".format(repository["full_name"]))
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
            releases_by_month_dates, releases_by_month_values = zip(
                *releases_by_month
            )
            releases_by_month_dates = list(releases_by_month_dates)
            releases_by_month_values = list(releases_by_month_values)
            releases_by_month_values = normalize(releases_by_month_values, 0, 1)

            metrics_values_pairs.append(
                [
                    (releases_by_month_dates[i], releases_by_month_values[i])
                    for i in range(len(releases_by_month_dates))
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

        # Feature extraction
        log.info("Extracting time series pairs features")

        ts_features_list = []
        for ts_record in repos_matrix_pairs:
            features_pair = extract_pair_series_features(ts_record)
            ts_features_list.append(features_pair)
        ts_features_df = pd.DataFrame(ts_features_list)

        # Store data
        ts_features_df.to_csv(
            "../data/time_series_clustering_distances.csv", index=False
        )
    else:
        # Load data
        ts_features_df = pd.read_csv("../data/time_series_clustering_distances.csv")

    log.info("Starting the clustering process")

    # Create dataframe for pair features
    df_feats = pd.concat([df_repository_metrics_phases, ts_features_df], axis=1)
    df_feats = df_feats.drop(columns=["id"])

    # Clustering
    transform_type = "std"  # preprocessing step
    model_type = "Hierarchical"  # clustering model

    linkage_matrix = linkage(df_feats.to_numpy(), method="ward")
    clustering_dendrogram = dendrogram(linkage_matrix)

    # Plot Dendrogram
    plt.title("Hierarchical Clustering Dendrogram - Repositories")
    plt.xlabel("Number of points in node.")
    plt.xticks([])  # Hide x ticks since the axis is too crowded
    plt.show()

    # Get optimal number of clusters
    clusters = len(set(clustering_dendrogram["leaves_color_list"]))

    log.info(f"Optimal number of clusters is: {clusters}\n")

    # Check if model exists
    if not Path(f"../models/clustering/mts_clustering.pickle").exists():
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
    # train_knn_classifier(
    #     df_feats,
    #     clustered_repositories,
    #     "../models/clustering/mts_clustering_classifier.pickle",
    # )

    # Print clustered repositories
    df_feats["cluster"] = clustered_repositories
    df_feats = df_feats[df_feats.columns.drop(list(df_feats.filter(regex="pair_")))]
    df_feats = df_feats.groupby(["cluster"]).mean()

    cluster_metrics_phases = {}
    df_feats_columns = list(df_feats.columns)
    df_feats_columns.sort()
    for idx, row in df_feats.iterrows():
        cluster_metrics_phases[f"cluster_{idx}"] = {
            "stargazers": [],
            "releases": [],
            "issues": [],
            "commits": [],
            "contributors": [],
            "deployments": [],
            "forks": [],
            "pull": [],
            "workflows": [],
        }
        for metric_phase in df_feats_columns:
            if metric_phase != "cluster":
                metric_id = metric_phase.split("_")[1]
                cluster_metrics_phases[f"cluster_{idx}"][metric_id].append(
                    int(proper_round(row[metric_phase]))
                )

    with open("../data/cluster_metrics_phases.json", "w") as outfile:
        json.dump(cluster_metrics_phases, outfile, indent=4)

    log.info("-----------------\n")
    log.info(cluster_metrics_phases)
    log.info("-----------------\n")

    clusters = {}
    for idx, cluster_id in enumerate(clustered_repositories):
        cluster_list = clusters.get(f"Cluster_{cluster_id}", [])
        cluster_list.append(repositories_names[idx])
        clusters[f"Cluster_{cluster_id}"] = cluster_list

    for cluster, repos in clusters.items():
        log.info(f"{cluster}: {len(repos)} repositories")
        log.info("-----------------")

    log.info("Successfully clustered repositories time series")
