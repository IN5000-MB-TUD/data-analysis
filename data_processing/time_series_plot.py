import logging

from matplotlib import pyplot as plt

from connection import mo
from utils.data import (
    get_stargazers_time_series,
    get_issues_time_series,
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
    repositories = mo.db["repositories_data"].find()

    for idx, repository in enumerate(repositories):
        log.info("Analyzing repository {}".format(repository["full_name"]))

        # Process stargazers
        if (
            repository.get("statistics", {}).get("stargazers")
            and repository["stargazers_count"] > 0
        ):
            stargazers, stargazers_cumulative = get_stargazers_time_series(repository)

            _create_plot(
                "../results/stargazers/{}_{}_{}.png".format(
                    idx, repository["owner"], repository["name"]
                ),
                "Stargazers {}".format(repository["full_name"]),
                "Total: {}".format(repository["stargazers_count"]),
                "Date",
                "Count",
                stargazers,
                [stargazers_cumulative],
            )

        # Process open issues
        if (
            repository.get("statistics", {}).get("issues")
            and repository["open_issues"] > 0
        ):
            issues_dates, issues_cumulative = get_issues_time_series(repository)

            _create_plot(
                "../results/issues/{}_{}_{}.png".format(
                    idx, repository["owner"], repository["name"]
                ),
                "Open Issues {}".format(repository["full_name"]),
                "Total: {}".format(repository["open_issues"]),
                "Date",
                "Count",
                issues_dates,
                [issues_cumulative],
            )

        log.info("------------------------")
