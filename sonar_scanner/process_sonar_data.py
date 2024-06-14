import json
import logging
from datetime import datetime
from itertools import groupby
from math import ceil
from operator import itemgetter
from pathlib import Path

import joblib
import pandas as pd
from dateutil.relativedelta import relativedelta
from pytz import utc

from connection import mo
from data_processing.time_series_phases import extrapolate_phases_properties
from data_processing.time_series_plot import create_plot
from utils.main import normalize
from utils.time_series import group_util, time_series_phases

# Setup logging
log = logging.getLogger(__name__)

REPOSITORIES = ["cockroachdb/cockroach", "patternfly/patternfly-react", "pypa/pip"]
METRICS = ["cognitive_complexity", "cyclomatic_complexity"]
DATE_FORMAT = "%Y-%m-%d"
SHOW_PLOTS = False
PHASES_LABELS = ["Steep", "Shallow", "Plateau"]


def _group_metric_by_month(
    dates, total_months, min_date, metric_values, monotonic=True
):
    """Group given list of dates by month."""
    if not dates:
        return []

    values_by_date = {dates[i]: metric_values[i] for i in range(len(dates))}

    dates_grouped = []
    dates.sort()

    for key, val in groupby(dates, key=lambda date: group_util(date, min_date)):
        # Keep only months that are >= 0
        if key >= 0:
            dates_grouped.append((key, list(val)))

    time_series_cumulative_by_month = []
    metric_counter = 0
    dates_grouped_idx = 0
    grouped_months_count = len(dates_grouped)
    for month_idx in range(total_months):
        if (
            dates_grouped_idx < grouped_months_count
            and month_idx == dates_grouped[dates_grouped_idx][0]
        ):
            month_last_date = dates_grouped[dates_grouped_idx][1][-1]
            if monotonic:
                metric_counter += values_by_date[month_last_date]
            else:
                metric_counter = values_by_date[month_last_date]

            dates_grouped_idx += 1

        time_series_cumulative_by_month.append(
            (min_date + relativedelta(months=month_idx), metric_counter)
        )

    return time_series_cumulative_by_month


if __name__ == "__main__":
    for repository_full_name in REPOSITORIES:
        log.info(
            f"Evaluate SonarCloud metrics for repository: {repository_full_name}\n"
        )

        repository_db_record = mo.db["repositories_data"].find_one(
            {"full_name": repository_full_name}
        )
        if not repository_db_record:
            log.info(
                f"No record found for repository {repository_full_name}, run the pipeline script to import the data"
            )
            exit()

        repository_age_months = ceil(repository_db_record["age"] / 2629746)
        repository_age_start = repository_db_record["created_at"].replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

        repository_folder = repository_full_name.replace("/", "_")
        metrics_time_series = {}
        for metric in METRICS:
            metrics_time_series[metric] = {}
            with open(f"./{repository_folder}/{metric}.json") as json_file:
                metric_data = json.load(json_file)

            data_pairs = []
            for data_point in metric_data["measures"][0]["history"]:
                data_pairs.append(
                    (
                        datetime.strptime(data_point["date"], DATE_FORMAT).replace(
                            tzinfo=utc
                        ),
                        int(data_point["value"]),
                    )
                )

            data_pairs = sorted(data_pairs, key=itemgetter(0))
            metrics_time_series[metric]["dates"] = [d for d, _ in data_pairs]
            metrics_time_series[metric]["values"] = [v for _, v in data_pairs]

        # Group metrics by month
        for metric in METRICS:
            metric_by_month = _group_metric_by_month(
                metrics_time_series[metric]["dates"],
                repository_age_months,
                repository_age_start,
                metrics_time_series[metric]["values"],
            )
            metric_by_month_dates, metric_by_month_values = zip(*metric_by_month)
            metric_by_month_dates = list(metric_by_month_dates)
            metric_by_month_values = list(metric_by_month_values)
            metrics_time_series[metric]["dates"] = metric_by_month_dates
            metrics_time_series[metric]["values"] = metric_by_month_values

        # Plot metrics curves
        if SHOW_PLOTS:
            for metric, metric_data in metrics_time_series.items():
                log.info(f"Plotting metric {metric} curve")
                metric_plot = create_plot(
                    "{} {}".format(metric, repository_full_name),
                    "Total: {}".format(metric_data["values"][-1]),
                    "Date",
                    "Count",
                    metric_data["dates"],
                    [metric_data["values"]],
                )
                metric_plot.show()

        log.info("Computing metrics phases...")
        metrics_phases = {}
        for metric, metric_data in metrics_time_series.items():
            metric_phases_idxs = time_series_phases(
                metric_data["values"],
                show_plot=SHOW_PLOTS,
                plot_title=f"{repository_full_name} {metric}",
                window_size=6,
            )

            normalized_values = normalize(metric_data["values"], 0, 1) + [1]

            metrics_phases[metric] = {
                "phases": metric_phases_idxs,
                "phases_count": len(metric_phases_idxs),
                "phases_dates": [
                    metric_data["dates"][i - 1] for i in metric_phases_idxs
                ],
                "phases_normalized_value": [
                    round(normalized_values[i], 2) for i in metric_phases_idxs
                ],
            }

            log.info(
                f"{metric} {metrics_phases[metric]['phases_count']} phases breakpoints: {metrics_phases[metric]['phases']}, {metrics_phases[metric]['phases_dates']}"
            )
            log.info(metrics_phases[metric]["phases_normalized_value"])

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

        log.info("--------------------------------------------------\n")

    log.info("Done!")
