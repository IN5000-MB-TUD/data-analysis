import logging
from datetime import datetime

import pymongo.errors
from pytz import utc

from connection import mo
from connection.github_api import GitHubAPI

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
STATISTICS_EXIST = True


def _update_statistics(collection, repository_full_name, update_query, message):
    """
    Update statistics in the given collection.

    :param collection: The collection name.
    :param repository_full_name: The repository full name.
    :param update_query: The update query.
    :param message: The message to log.
    :return: True if the update was successful. False otherwise.
    """
    try:
        mo.db[collection].update_one(
            {"full_name": repository_full_name},
            {"$set": update_query},
        )
        log.info(
            "Successfully updated {} for {}".format(
                message,
                repository_full_name,
            )
        )
    except pymongo.errors.PyMongoError as err:
        log.error(
            "Error while updating {} for {}: {}".format(
                message,
                repository_full_name,
                err,
            )
        )
        return False

    return True


if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval")
    github_api_client = GitHubAPI()

    # Get the repositories in the database
    repositories = mo.db["repositories_data"].find(
        {"statistics": {"$exists": STATISTICS_EXIST}}
    )

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
        if repository_commits_count is not None:
            update_flag = _update_statistics(
                "repositories_data",
                repository["full_name"],
                {
                    "commits": repository_commits_count,
                    "statistics.commits": repository_commits_dates,
                    "statistics.contributors": repository_contributors,
                },
                "commits data",
            )

        repository_stargazers = github_api_client.get_repository_stargazers_time(
            repository["owner"], repository["name"]
        )
        if repository_stargazers is not None:
            update_flag = _update_statistics(
                "repositories_data",
                repository["full_name"],
                {
                    "statistics.stargazers": repository_stargazers,
                },
                "stargazers time series",
            )

        repository_issues = github_api_client.get_repository_issues_time(
            repository["owner"], repository["name"]
        )
        if repository_issues is not None:
            update_flag = _update_statistics(
                "repositories_data",
                repository["full_name"],
                {
                    "statistics.issues": repository_issues,
                },
                "issues time series",
            )

        repository_workflows = github_api_client.get_repository_workflows(
            repository["owner"], repository["name"]
        )
        if repository_workflows is not None:
            update_flag = _update_statistics(
                "repositories_data",
                repository["full_name"],
                {
                    "workflows": repository_workflows,
                },
                "workflows data",
            )

        repository_workflow_runs = github_api_client.get_repository_workflows_time(
            repository["owner"], repository["name"]
        )
        if repository_workflow_runs is not None:
            update_flag = _update_statistics(
                "repositories_data",
                repository["full_name"],
                {
                    "statistics.workflows": repository_workflow_runs,
                },
                "workflows time series",
            )

        repository_environments = github_api_client.get_repository_environments(
            repository["owner"], repository["name"]
        )
        if repository_environments is not None:
            update_flag = _update_statistics(
                "repositories_data",
                repository["full_name"],
                {
                    "environments": repository_environments,
                },
                "environments data",
            )

        repository_deployments = github_api_client.get_repository_deployments(
            repository["owner"], repository["name"]
        )
        if repository_deployments is not None:
            update_flag = _update_statistics(
                "repositories_data",
                repository["full_name"],
                {
                    "statistics.deployments": repository_deployments,
                },
                "deployments time series",
            )

        repository_pull_requests = github_api_client.get_repository_pull_requests_time(
            repository["owner"], repository["name"]
        )
        if repository_pull_requests is not None:
            update_flag = _update_statistics(
                "repositories_data",
                repository["full_name"],
                {
                    "statistics.pull_requests": repository_pull_requests,
                },
                "pull requests time series",
            )

        repository_forks = github_api_client.get_repository_forks(
            repository["owner"], repository["name"]
        )
        if repository_forks is not None:
            update_flag = _update_statistics(
                "repositories_data",
                repository["full_name"],
                {
                    "statistics.forks": repository_forks,
                },
                "forks time series",
            )

        # Update the metadata
        if update_flag:
            _update_statistics(
                "repositories_data",
                repository["full_name"],
                {
                    "metadata.modified": datetime.now(tz=utc),
                },
                "statistics data",
            )
            log.info("---------------------------------------------------\n")
