import json
import logging
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

from connection import mo
from data_processing.t2f.selection.selection import feature_selection

# Setup logging
log = logging.getLogger(__name__)

INPUT_REDUCTIONS = [32, 16, 8, 4, 2]
MIN_CLUSTER_SIZE = 10


if __name__ == "__main__":
    if not Path("../data/time_series_clustering.csv").exists():
        log.warning(
            "The time_series_clustering.csv file is not present in the data folder. Please run the time_series_clustering.py script first!"
        )
        exit()

    log.info(
        "Evaluating the multivariate time series clustering using a different amount of input data"
    )

    # Get clusters patterns distribution
    with open("../data/repository_metrics_phases.json") as json_file:
        repository_metrics_phases = json.load(json_file)
    with open("../data/repository_clusters.json") as json_file:
        repository_clusters = json.load(json_file)

    clusters_patterns_distribution = {
        "Cluster_0": [0, 0, 0],
        "Cluster_1": [0, 0, 0],
        "Cluster_2": [0, 0, 0],
    }

    for cluster, cluster_repos in repository_clusters.items():
        for repository_name in cluster_repos:
            # Count patterns
            for metric, metric_patterns in repository_metrics_phases[
                repository_name
            ].items():
                for pattern in metric_patterns:
                    clusters_patterns_distribution[cluster][pattern] += 1

    log.info(clusters_patterns_distribution)

    clusters_patterns_totals = {
        "Cluster_0": sum(clusters_patterns_distribution["Cluster_0"]),
        "Cluster_1": sum(clusters_patterns_distribution["Cluster_1"]),
        "Cluster_2": sum(clusters_patterns_distribution["Cluster_2"]),
    }

    clusters_patterns_percentages = {
        "Cluster_0": [
            round(i / clusters_patterns_totals["Cluster_0"], 3)
            for i in clusters_patterns_distribution["Cluster_0"]
        ],
        "Cluster_1": [
            round(i / clusters_patterns_totals["Cluster_1"], 3)
            for i in clusters_patterns_distribution["Cluster_1"]
        ],
        "Cluster_2": [
            round(i / clusters_patterns_totals["Cluster_2"], 3)
            for i in clusters_patterns_distribution["Cluster_2"]
        ],
    }

    log.info(clusters_patterns_percentages)

    # Get the repositories in the database
    repositories = list(
        mo.db["repositories_data"].find({"statistics": {"$exists": True}})
    )
    repositories_count = len(repositories)

    # Load data
    df_feats = pd.read_csv("../data/time_series_clustering.csv")

    # Drop not needed columns
    df_feats = df_feats.drop(columns=["id"])

    # Select best features
    top_features = feature_selection(df_feats)
    df_feats_dendrogram = df_feats[top_features]

    # Clustering
    transform_type = "std"  # preprocessing step
    model_type = "Hierarchical"  # clustering model

    results = {}

    for input_reduction in INPUT_REDUCTIONS:
        input_size = int(repositories_count / input_reduction)
        log.info(f"Clustering with input size of {input_size} repositories")

        df_feats_dendrogram_reduced = df_feats_dendrogram.head(input_size).reset_index(
            drop=True
        )
        linkage_matrix = linkage(df_feats_dendrogram_reduced.to_numpy(), method="ward")
        clustering_dendrogram = dendrogram(linkage_matrix)

        # Plot Dendrogram
        plt.title(f"Multivariate Clustering - N/{input_reduction} Input")
        plt.xticks([])  # Hide x ticks since the axis is too crowded
        plt.show()

        # Get optimal number of clusters
        results[f"input_{input_size}"] = {}
        cluster_labels = list(set(clustering_dendrogram["leaves_color_list"]))
        clusters = len(cluster_labels)

        results[f"input_{input_size}"]["clusters"] = clusters
        for cluster_label in cluster_labels:
            results[f"input_{input_size}"][cluster_label] = clustering_dendrogram[
                "leaves_color_list"
            ].count(cluster_label)

        log.info(f"Optimal number of clusters is: {clusters}\n")
        log.info("-----------------------------")

    with open("../data/evaluation_clustering.json", "w") as evaluation_clustering_file:
        json.dump(results, evaluation_clustering_file, indent=4)

    log.info("Done!")
