import logging

from connection import mo
from data_processing.utils import (
    create_plot,
    get_stargazers_time_series,
    get_issues_time_series,
)

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


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

            create_plot(
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

            create_plot(
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
