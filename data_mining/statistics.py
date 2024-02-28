import logging
from datetime import datetime

from pytz import utc

from connection import mo
from connection.github_api import GitHubAPI

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval")
    github_api_client = GitHubAPI()

    # Get the repositories in the database
    repositories = mo.db["repositories_data"].find({"statistics": {"$exists": True}})

    for repository in repositories:
        update_flag = False

        # Gather time series about commits, stargazers and issues
        # The operations are executed if the db entry has been updated more than 1 day ago
        if (datetime.now(tz=utc) - repository["metadata"]["modified"]).days < 1:
            log.info(
                "Skipping repository {} since it was updated less than 1 day ago.".format(
                    repository["full_name"]
                )
            )
            continue

        (
            repository_commits_count,
            repository_commits_dates,
            repository_contributors,
        ) = github_api_client.get_repository_commits(
            repository["owner"], repository["name"]
        )
        if repository_commits_count:
            mo.db["repositories_data"].update_one(
                {"full_name": repository["full_name"]},
                {
                    "$set": {
                        "commits": repository_commits_count,
                        "statistics.commits": repository_commits_dates,
                        "statistics.contributors": repository_contributors,
                    }
                },
            )
            log.info(
                "Successfully updated commits data for {}".format(
                    repository["full_name"]
                )
            )
            update_flag = True

        repository_stargazers = github_api_client.get_repository_stargazers_time(
            repository["owner"], repository["name"]
        )
        if repository_stargazers:
            mo.db["repositories_data"].update_one(
                {"full_name": repository["full_name"]},
                {
                    "$set": {
                        "statistics.stargazers": repository_stargazers,
                    }
                },
            )
            log.info(
                "Successfully updated stargazers time series for {}".format(
                    repository["full_name"]
                )
            )
            update_flag = True

        repository_issues = github_api_client.get_repository_issues_time(
            repository["owner"], repository["name"]
        )
        if repository_issues:
            mo.db["repositories_data"].update_one(
                {"full_name": repository["full_name"]},
                {
                    "$set": {
                        "statistics.issues": repository_issues,
                    }
                },
            )
            log.info(
                "Successfully updated issues time series for {}".format(
                    repository["full_name"]
                )
            )
            update_flag = True

        # Update the metadata
        if update_flag:
            mo.db["repositories_data"].update_one(
                {"full_name": repository["full_name"]},
                {
                    "$set": {
                        "metadata.modified": datetime.now(tz=utc),
                    }
                },
            )
            log.info(
                "Successfully updated statistics data for {} -------------".format(
                    repository["full_name"]
                )
            )
