import logging
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

from connection import mo
from data_processing.utils import get_stargazers_time_series

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval from Database")

    # Get the repositories in the database
    repositories = mo.db["repositories_data"].find()

    repository_sequences = {}
    stargazers_features = pd.DataFrame()

    for idx, repository in enumerate(repositories):
        log.info("Analyzing repository {}".format(repository["full_name"]))

        # Process stargazers
        if (
            repository.get("statistics", {}).get("stargazers")
            and repository["stargazers_count"] > 0
        ):
            stargazers, stargazers_cumulative = get_stargazers_time_series(repository)
            stargazers_tuples = [
                (stargazers[i], repository["full_name"], stargazers_cumulative[i])
                for i in range(len(stargazers))
            ]

            stargazers_df = pd.DataFrame(
                stargazers_tuples, columns=["Date", "Repository", "Stargazers"]
            )
            extracted_features = extract_features(
                stargazers_df,
                column_id="Repository",
                column_sort="Date",
                default_fc_parameters=MinimalFCParameters(),
            )
            stargazers_features = pd.concat(
                [stargazers_features, extracted_features], ignore_index=False
            )

    # Cluster the repositories stargazers features data
    prep = StandardScaler()
    scaled_data = prep.fit_transform(stargazers_features)

    # Get optimal clusters number
    clusters = 2
    best_fit = -1
    for n_cluster in range(2, len(stargazers_features.index)):
        kmeans = KMeans(n_clusters=n_cluster, random_state=0)
        kmeans.fit(scaled_data)
        if silhouette_score(scaled_data, kmeans.labels_) > best_fit:
            best_fit = silhouette_score(scaled_data, kmeans.labels_)
            clusters = n_cluster
    log.info(
        f"Optimal number of clusters is: {clusters} with silhouette_score: {best_fit}"
    )

    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(scaled_data)

    for cluster in range(kmeans.n_clusters):
        log.info(f"Cluster {cluster}")
        cluster_data = stargazers_features[kmeans.labels_ == cluster]
        for i in range(min(5, cluster_data.shape[0])):
            log.info(cluster_data.index[i])
        log.info("")

    log.info("Successfully clustered repositories time series")
