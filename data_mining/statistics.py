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


def _update_statistics(
    collection, repository_id, repository_full_name, update_query, message
):
    """
    Update statistics in the given collection.

    :param collection: The collection name.
    :param repository_id: The repository unique id in the database.
    :param repository_full_name: The repository full name.
    :param update_query: The update query.
    :param message: The message to log.
    :return: True if the update was successful. False otherwise.
    """

    # Setup query
    search_query = {"full_name": repository_full_name}
    if repository_id:
        search_query["repository_id"] = repository_id

    try:
        mo.db[collection].update_one(
            search_query,
            {"$set": update_query},
            upsert=True,
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
    except pymongo.errors.DocumentTooLarge as err:
        log.warning(
            "DocumentTooLarge while updating {} for {}: {}. Needs down-sampling.".format(
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
                "Skipping repository {} since it was updated less than 1 day ago.\n".format(
                    repository["full_name"]
                )
            )
            continue

        repository_commits_count = github_api_client.get_repository_commits_count(
            repository["owner"], repository["name"]
        )
        if repository_commits_count is not None:
            _update_statistics(
                "repositories_data",
                None,
                repository["full_name"],
                {
                    "commits": repository_commits_count,
                },
                "commits count",
            )
            repository["commits"] = repository_commits_count

        (
            repository_commits_dates,
            repository_contributors,
        ) = github_api_client.get_repository_commits(repository)
        if repository_commits_dates is not None:
            update_flag = _update_statistics(
                "statistics_commits",
                repository["_id"],
                repository["full_name"],
                {
                    "commits": repository_commits_dates,
                    "contributors": repository_contributors,
                },
                "commits",
            )

        repository_stargazers = github_api_client.get_repository_stargazers_time(
            repository
        )
        if repository_stargazers is not None:
            update_flag = _update_statistics(
                "statistics_stargazers",
                repository["_id"],
                repository["full_name"],
                {
                    "stargazers": repository_stargazers,
                },
                "stargazers",
            )

        repository_issues = github_api_client.get_repository_issues_time(repository)
        if repository_issues is not None:
            update_flag = _update_statistics(
                "statistics_issues",
                repository["_id"],
                repository["full_name"],
                {
                    "issues": repository_issues,
                },
                "issues",
            )

        repository_workflows = github_api_client.get_repository_workflows(
            repository["owner"], repository["name"]
        )
        if repository_workflows is not None:
            update_flag = _update_statistics(
                "repositories_data",
                None,
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
                "statistics_workflow_runs",
                repository["_id"],
                repository["full_name"],
                {
                    "workflows": repository_workflow_runs,
                },
                "workflows",
            )

        repository_environments = github_api_client.get_repository_environments(
            repository["owner"], repository["name"]
        )
        if repository_environments is not None:
            update_flag = _update_statistics(
                "repositories_data",
                None,
                repository["full_name"],
                {
                    "environments": repository_environments,
                },
                "environments data",
            )

        repository_deployments = github_api_client.get_repository_deployments(
            repository["owner"],
            repository["name"],
            repository_environments if repository_environments else {},
        )
        if repository_deployments is not None:
            update_flag = _update_statistics(
                "statistics_deployments",
                repository["_id"],
                repository["full_name"],
                {
                    "deployments": repository_deployments,
                },
                "deployments",
            )

        repository_pull_requests = github_api_client.get_repository_pull_requests_time(
            repository["owner"], repository["name"]
        )
        if repository_pull_requests is not None:
            update_flag = _update_statistics(
                "statistics_pull_requests",
                repository["_id"],
                repository["full_name"],
                {
                    "pull_requests": repository_pull_requests,
                },
                "pull_requests",
            )

        repository_forks = github_api_client.get_repository_forks(repository)
        if repository_forks is not None:
            update_flag = _update_statistics(
                "statistics_forks",
                repository["_id"],
                repository["full_name"],
                {
                    "forks": repository_forks,
                },
                "forks",
            )

        # Update the metadata
        if update_flag:
            _update_statistics(
                "repositories_data",
                None,
                repository["full_name"],
                {
                    "statistics": True,
                    "metadata.modified": datetime.now(tz=utc),
                },
                "statistics data",
            )
            log.info("---------------------------------------------------\n")
