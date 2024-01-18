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
    repositories = mo.db["repositories_data"].find()

    for repository in repositories:
        repository_statistics = github_api_client.get_weekly_commits_statistics(repository["owner"], repository["name"])

        # If new statistics are available, update the object
        # The api does not process repos with more than 10000 commits, so those are skipped
        if repository_statistics and repository["commits"] <= 10000:
            mo.db["repositories_data"].update_one(
                {"full_name": repository["full_name"]},
                {"$set": {
                    "statistics": {
                        "commits_weekly": repository_statistics,
                    },
                    "metadata": {
                        "modified": datetime.now(tz=utc),
                    }
                }}
            )
            log.info("Successfully updated statistics for {}".format(repository["full_name"]))
