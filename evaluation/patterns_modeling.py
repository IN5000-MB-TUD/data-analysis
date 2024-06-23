import json
import logging
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from connection import mo
from data_processing.t2f.model.clustering import ClusterWrapper

# Setup logging
log = logging.getLogger(__name__)

INPUT_REDUCTIONS = [32, 16, 8, 4, 2]
MIN_CLUSTER_SIZE = 10


if __name__ == "__main__":
    if not Path("../data/time_series_phases.csv").exists():
        log.warning(
            "The time_series_phases.csv file is not present in the data folder. Please run the time_series_phases.py script first!"
        )
        exit()

    log.info(
        "Evaluating the patterns generalization using a different amount of input data"
    )

    # Load data
    phases_features = pd.read_csv("../data/time_series_phases.csv")

    # Get the repositories in the database
    repositories = list(
        mo.db["repositories_data"].find({"statistics": {"$exists": True}})
    )
    repositories_count = len(repositories)

    # Input ratio
    input_ratio = ceil(phases_features.shape[0] / repositories_count)

    transform_type = "std"  # preprocessing step
    model_type = "Hierarchical"  # clustering model

    results = {}

    for input_reduction in INPUT_REDUCTIONS:
        input_size = int(repositories_count / input_reduction)
        log.info(f"Clustering with input size of {input_size} repositories")

        # Filter data
        input_count = input_size * input_ratio
        df_phases = (
            phases_features.drop(columns=["phase_order"])
            .head(input_count)
            .reset_index(drop=True)
        )

        # Clustering
        best_fit = float("inf")
        clusters = 3
        for n_cluster in range(3, 5):
            model = ClusterWrapper(
                n_clusters=n_cluster,
                model_type=model_type,
                transform_type=transform_type,
            )
            model.model.fit(df_phases)

            # Check cluster balance
            _, labels_count = np.unique(model.model.labels_, return_counts=True)
            small_cluster_check = [c < MIN_CLUSTER_SIZE for c in labels_count]
            if True in small_cluster_check:
                continue

            labels_variance = np.var(labels_count)
            if labels_variance < best_fit:
                best_fit = labels_variance
                clusters = n_cluster
        log.info(f"Optimal number of clusters is: {clusters}.")

        # Fit model
        model = ClusterWrapper(
            n_clusters=clusters, model_type=model_type, transform_type=transform_type
        )
        model.model.fit(df_phases)

        # Cluster repos
        clustered_phases = model.fit_predict(df_phases)
        df_phases["phase_order"] = clustered_phases

        # Compute average polynomial coefficients
        df_phases = df_phases.groupby(["phase_order"]).mean()

        results[f"input_{input_size}"] = df_phases.to_dict()
        log.info(results[f"input_{input_size}"])

        # Plot phases curves
        x = list(range(0, 13))

        for idx, row in df_phases.iterrows():
            # Plot phase
            poly_coefficients = [
                row[3],
                row[2],
                row[1],
                row[0],
            ]
            y = np.polyval(poly_coefficients, x)
            plt.plot(x, y, label=f"Pattern {idx}")

        plt.title(f"Evolution Patterns - N/{input_reduction} Input")
        plt.xlabel("Time (Months)")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.show()

        log.info("-----------------------------")

    with open(
        "../data/evaluation_patterns_modeling.json", "w"
    ) as evaluation_patterns_modeling_file:
        json.dump(results, evaluation_patterns_modeling_file, indent=4)

    log.info("Done!")
