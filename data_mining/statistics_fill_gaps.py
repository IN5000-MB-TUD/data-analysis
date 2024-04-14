import logging

from connection import mo

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
METRICS_QUERY = {
    "commits": {
        "commits": {},
        "contributors": {},
    },
    "deployments": {
        "deployments": {},
    },
    "forks": {
        "forks": {},
    },
    "issues": {
        "issues": {},
    },
    "pull_requests": {
        "pull_requests": {},
    },
    "size": {
        "size": {},
    },
    "stargazers": {
        "stargazers": [],
    },
    "workflow_runs": {
        "workflows": {},
    },
}


def fill_repository_statistics_gaps(repository):
    """
    Fill the missing statistics data for a repository.

    :param repository: The repository data.
    """
    log.info(f"Checking repository {repository['full_name']}")

    # Check if a statistics is missing and fill it
    for metric, update_query in METRICS_QUERY.items():
        metric_statistic = mo.db[f"statistics_{metric}"].find_one(
            {
                "full_name": repository["full_name"],
                "repository_id": repository["_id"],
            }
        )
        if not metric_statistic:
            log.info(f"Filling statistics for metric: {metric}")
            item_to_insert = {
                "full_name": repository["full_name"],
                "repository_id": repository["_id"],
            }
            item_to_insert.update(update_query)
            mo.db[f"statistics_{metric}"].insert_one(item_to_insert)

    log.info("---------------------------------------------------\n")


if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval")

    # Get the repositories in the database
    repositories_data = mo.db["repositories_data"].find(
        {"statistics": {"$exists": True}}
    )

    for repository_data in repositories_data:
        fill_repository_statistics_gaps(repository_data)
