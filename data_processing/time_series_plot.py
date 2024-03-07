import logging

from matplotlib import pyplot as plt

from connection import mo
from utils.data import (
    get_stargazers_time_series,
    get_metric_time_series,
    get_metrics_information,
)

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _create_plot(path, title, subtitle, xlabel, ylabel, x, y, labels=None):
    """Create plot and save it as PNG"""
    # Adjusting the figure size
    plt.subplots(figsize=(16, 8))

    # Creating a plot
    for y_idx, y_data in enumerate(y):
        if labels:
            plt.plot(x, y_data, label=labels[y_idx])
        else:
            plt.plot(x, y_data)

    if labels:
        plt.legend(loc="upper left")

    # Adding a plot title and customizing its font size
    plt.suptitle(title, fontsize=20)
    plt.title(subtitle, fontsize=15)

    # Adding axis labels and customizing their font size
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)

    # Rotating axis ticks and customizing their font size
    plt.xticks(rotation=30, fontsize=15)

    # Saving the resulting plot to a file
    plt.savefig(path)


if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval from Database")

    # Get the repositories in the database
    repositories = mo.db["repositories_data"].find({"statistics": {"$exists": True}})

    for idx, repository in enumerate(repositories):
        log.info("Plotting metrics for repository {}".format(repository["full_name"]))

        # Plot metrics
        stargazers, stargazers_cumulative = get_stargazers_time_series(repository)

        _create_plot(
            "../plots/stargazers/{}_{}_{}.png".format(
                idx, repository["owner"], repository["name"]
            ),
            "Stargazers {}".format(repository["full_name"]),
            "Total: {}".format(repository["stargazers_count"]),
            "Date",
            "Count",
            stargazers,
            [stargazers_cumulative],
        )

        for metric in get_metrics_information():
            metric_dates, metric_cumulative = get_metric_time_series(
                repository,
                metric[0],
                metric[1],
                metric[2],
                metric[3],
            )

            if len(metric_cumulative) == 0:
                continue

            _create_plot(
                "../plots/{}/{}_{}_{}.png".format(
                    metric[1], idx, repository["owner"], repository["name"]
                ),
                "{} {}".format(metric[1], repository["full_name"]),
                "Total: {}".format(metric_cumulative[-1]),
                "Date",
                "Count",
                metric_dates,
                [metric_cumulative],
            )

        log.info("------------------------")
