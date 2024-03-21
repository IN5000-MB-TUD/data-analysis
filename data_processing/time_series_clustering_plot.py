import json
import logging
from itertools import groupby
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from connection import mo

# Setup logging
log = logging.getLogger(__name__)

PHASES = 3

if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval from Database")

    if not Path("../data/cluster_metrics_phases.json").exists():
        log.warning(
            "The cluster_metrics_phases.json file is not present in the data folder. Please run the time_series_clustering.py script first!"
        )
        exit()

    with open("../data/cluster_metrics_phases.json") as json_file:
        cluster_metrics_phases = json.load(json_file)

    # Group phases
    cluster_metrics_phases_grouped = {}
    for cluster, metrics in cluster_metrics_phases.items():
        log.info(f"Grouping phases for cluster {cluster}")
        cluster_metrics_phases_grouped[cluster] = {}
        for metric, phases in metrics.items():
            phases_grouped = []
            for phase_id, phase_sequence in groupby(phases):
                phases_grouped.append((phase_id, len([*phase_sequence])))
            cluster_metrics_phases_grouped[cluster][metric] = phases_grouped

        log.info("---------------------------------------\n")

    # Build and plot time series based on predicted phases
    phases_coefficients = {}
    evolution_phases = mo.db["evolution_phases"].find()
    for phase in evolution_phases:
        phase_id = f"phase_{phase['phase_order']}"
        phases_coefficients[phase_id] = {}
        for key, value in phase.items():
            if "coeff" in key:
                coefficient_id = int(key.split("coeff_")[1][0])
                phases_coefficients[phase_id][f"coefficient_{coefficient_id}"] = value

    for cluster, metrics in cluster_metrics_phases_grouped.items():
        log.info(f"Generating curves for cluster {cluster}")
        # Initialise the subplot function using number of rows and columns
        figure, axis = plt.subplots(4, 2)
        x_idx = 0
        y_idx = 0

        for metric, phases in metrics.items():
            start_value = 0
            metric_time_series = []

            for phase, phase_count in phases:
                poly_coefficients = [
                    phases_coefficients[f"phase_{phase}"][f"coefficient_3"],
                    phases_coefficients[f"phase_{phase}"][f"coefficient_2"],
                    phases_coefficients[f"phase_{phase}"][f"coefficient_1"],
                    phases_coefficients[f"phase_{phase}"][f"coefficient_0"],
                ]
                x = list(range(0, 13 * phase_count))
                y = np.polyval(poly_coefficients, x) + start_value
                start_value = y[-1]
                metric_time_series.extend(y)

            # Plot values
            axis[x_idx, y_idx].plot(
                list(range(len(metric_time_series))), metric_time_series
            )
            axis[x_idx, y_idx].set_title(metric)

            if y_idx == 1:
                y_idx = 0
                x_idx += 1
            else:
                y_idx += 1

        # Display cluster plot
        figure.suptitle(f"Metrics curves for {cluster}")
        figure.tight_layout()  # Improve spacing
        plt.show()

        log.info("---------------------------------------\n")

    log.info("Successfully generated curves for clusters metrics.")
