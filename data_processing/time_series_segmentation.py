import logging

from fastpip import pip

from connection import mo
from data_processing.utils import (
    get_stargazers_time_series,
    get_issues_time_series,
    get_additions_deletions_time_series,
    compute_time_series_segments_trends,
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

        # Compute the k amount of perceptually important points (PIP) based on the repository age
        # Get number of months
        k = int(repository["age"] / 2.628e6)

        # If too low, get number of weeks
        if k == 0:
            k = int(repository["age"] / 604800)

        # Process stargazers
        if (
            repository.get("statistics", {}).get("stargazers")
            and repository["stargazers_count"] > 0
        ):
            stargazers, stargazers_cumulative = get_stargazers_time_series(repository)

            # Create list of points (x, y) => (time_ms, stargazers_count)
            stargazers_time_series = [
                (int(stargazers[i].timestamp()), stargazers_cumulative[i])
                for i in range(len(stargazers))
            ]
            stargazers_pip = pip(stargazers_time_series, k)

            stargazers_trends = compute_time_series_segments_trends(stargazers_pip)

        # Process open issues
        if (
            repository.get("statistics", {}).get("issues")
            and repository["open_issues"] > 0
        ):
            issues_dates, issues_cumulative = get_issues_time_series(repository)

            # Create list of points (x, y) => (time_ms, issues_count)
            issues_time_series = [
                (int(issues_dates[i].timestamp()), issues_cumulative[i])
                for i in range(len(issues_dates))
            ]
            issues_pip = pip(issues_time_series, k)

            issues_trends = compute_time_series_segments_trends(issues_pip)

        # Process Weekly Commits Stats
        if repository.get("statistics", {}).get("commits_weekly"):
            (
                commits_dates,
                commits_cumulative,
                additions_cumulative,
                deletions_cumulative,
            ) = get_additions_deletions_time_series(repository)

            # Create list of points (x, y) => (time_ms, changes_count)
            changes_time_series = [
                (int(commits_dates[i].timestamp()), commits_cumulative[i])
                for i in range(len(commits_dates))
            ]
            additions_time_series = [
                (int(commits_dates[i].timestamp()), additions_cumulative[i])
                for i in range(len(commits_dates))
            ]
            deletions_time_series = [
                (int(commits_dates[i].timestamp()), deletions_cumulative[i])
                for i in range(len(commits_dates))
            ]

            changes_pip = pip(changes_time_series, k)
            additions_pip = pip(additions_time_series, k)
            deletions_pip = pip(deletions_time_series, k)

            changes_trends = compute_time_series_segments_trends(changes_pip)
            additions_trends = compute_time_series_segments_trends(additions_pip)
            deletions_trends = compute_time_series_segments_trends(deletions_pip)

        log.info("------------------------")
