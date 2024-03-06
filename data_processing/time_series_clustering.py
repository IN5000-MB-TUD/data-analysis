import logging

import joblib
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from connection import mo
from data_processing.t2f.extraction.extractor import feature_extraction
from data_processing.t2f.model.clustering import ClusterWrapper
from data_processing.t2f.selection.selection import feature_selection
from utils.data import (
    get_stargazers_time_series,
    get_metric_time_series,
)
from utils.time_series import (
    build_time_series,
)

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval from Database")

    # Get the repositories in the database
    repositories = mo.db["repositories_data"].find({"statistics": {"$exists": True}})
    repositories_names = []

    repos_matrix_pairs = []
    repos_matrix_single = []

    for idx, repository in enumerate(repositories):
        log.info("Analyzing repository {}".format(repository["full_name"]))

        repositories_names.append(repository["full_name"])
        metrics_values_pairs = []
        metrics_values_single = []

        # Gather metrics
        stargazers_dates, stargazers_cumulative = get_stargazers_time_series(repository)
        metrics_values_pairs.append(zip(stargazers_dates, stargazers_cumulative))
        metrics_values_single.append(zip(stargazers_dates, stargazers_cumulative))

        metrics = [
            ("statistics_commits", "commits", "date", "commits"),
            ("statistics_commits", "contributors", "first_commit", None),
            ("statistics_deployments", "deployments", "created_at", None),
            ("statistics_issues", "issues", "created_at", "open_issues"),
            ("statistics_forks", "forks", "created_at", "forks_count"),
            ("statistics_pull_requests", "pull_requests", "created_at", None),
            ("statistics_workflow_runs", "workflows", "created_at", None),
        ]

        for metric in metrics:
            metric_dates, metric_cumulative = get_metric_time_series(
                repository,
                metric[0],
                metric[1],
                metric[2],
                metric[3],
            )

            metrics_values_pairs.append(zip(metric_dates, metric_cumulative))
            metrics_values_single.append(zip(metric_dates, metric_cumulative))

        # Populate data frame
        repos_matrix_pairs.append(metrics_values_pairs)
        repos_matrix_single.append(metrics_values_single)

    # Feature extraction
    df_feats = feature_extraction(
        np.array(repos_matrix_pairs), np.array(repos_matrix_single), batch_size=100, p=1
    )

    transform_type = "std"  # preprocessing step
    model_type = "Hierarchical"  # clustering model

    # Feature selection
    context = {'model_type': model_type, 'transform_type': transform_type}
    top_feats = feature_selection(df_feats, labels={}, context=context)
    df_feats = df_feats[top_feats]

    # Scale features
    prep = MinMaxScaler()
    df_feats = prep.fit_transform(df_feats)

    # Clustering
    best_fit = -1
    clusters = 2
    for n_cluster in range(2, len(repositories_names)):
        model = ClusterWrapper(
            n_clusters=n_cluster, model_type=model_type, transform_type=transform_type
        )
        model.model.fit(df_feats)
        if silhouette_score(df_feats, model.model.labels_) > best_fit:
            best_fit = silhouette_score(df_feats, model.model.labels_)
            clusters = n_cluster
    log.info(
        f"Optimal number of clusters is: {clusters} with silhouette_score: {best_fit}"
    )

    # Save model
    model = ClusterWrapper(
        n_clusters=clusters, model_type=model_type, transform_type=transform_type
    )
    model.model.fit(df_feats)
    joblib.dump(
        model,
        f"../models/clustering/mts_clustering_{len(repositories_names)}_repos.pickle",
    )

    # Print clustered repos
    clustered_repositories = model.fit_predict(df_feats)
    log.info(clustered_repositories)
    clusters = {}
    for idx, cluster_id in enumerate(clustered_repositories):
        cluster_list = clusters.get(f"Cluster_{cluster_id}", [])
        cluster_list.append(repositories_names[idx])
        clusters[f"Cluster_{cluster_id}"] = cluster_list

    for cluster, repos in clusters.items():
        log.info(cluster)
        for repo in repos:
            log.info(repo)
        log.info("-----------------")

    # Plot Dendogram
    # Create the counts of samples under each node
    plt.title("Hierarchical Clustering Dendrogram")
    counts = np.zeros(model.model.children_.shape[0])
    n_samples = len(model.model.labels_)
    for i, merge in enumerate(model.model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.model.children_, model.model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, truncate_mode="level", p=4)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    log.info("Successfully clustered repositories time series")
