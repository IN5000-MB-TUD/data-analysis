import logging

from fastpip import pip

from connection import mo
from data_processing.utils import (
    get_stargazers_time_series,
    get_issues_time_series,
    compute_time_series_segments_trends,
    compute_pattern_distance,
)

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval from Database")

    # Get the repositories in the database
    repositories = mo.db["repositories_data"].find()

    repository_sequences = {}

    for idx, repository in enumerate(repositories):
        log.info("Analyzing repository {}".format(repository["full_name"]))

        repository_sequences[repository["full_name"]] = {}

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
            repository_sequences[repository["full_name"]][
                "stargazers_trends"
            ] = stargazers_trends

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
            repository_sequences[repository["full_name"]][
                "issues_trends"
            ] = issues_trends

        log.info("------------------------")

    log.info("Start computation of pattern distances")
    pattern_distance_matrix = {}

    for repo_1_key, repo_1_values in repository_sequences.items():
        pattern_distance_matrix[repo_1_key] = {}
        for repo_2_key, repo_2_values in repository_sequences.items():
            # Avoid computing for the same repo
            if repo_1_key == repo_2_key:
                pattern_distance_matrix[repo_1_key][repo_2_key] = {
                    "stargazers": 0,
                    "issues": 0,
                }
                continue

            pattern_distance_matrix[repo_1_key][repo_2_key] = {}

            # Loop through repository available values
            for trend_key, trend_sequence in repo_1_values.items():
                if trend_key in repo_2_values:
                    pattern_distance_matrix[repo_1_key][repo_2_key][
                        trend_key
                    ] = compute_pattern_distance(
                        trend_sequence, repo_2_values[trend_key]
                    )
                else:
                    pattern_distance_matrix[repo_1_key][repo_2_key][trend_key] = None

    # Store in DB
    mo.db["repositories_pattern_distance"].insert_one(pattern_distance_matrix)

    log.info("Pattern distance matrix computation completed")
