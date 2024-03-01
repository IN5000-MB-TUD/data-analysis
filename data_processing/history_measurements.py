import logging
from datetime import datetime

from pytz import utc

from connection import mo
from data_processing.utils import (
    sorted_once_nested_dict,
    create_releases_statistics_dictionary,
    compute_releases_history_metrics,
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

        # Process versions/releases
        if repository.get("releases") and len(repository.get("releases", {})) > 1:
            log.info("Start historical analysis")

            # Get releases dates
            sorted_releases = sorted_once_nested_dict(
                repository["releases"], "created_at"
            )

            # Build releases stats dictionary
            releases_statistics = create_releases_statistics_dictionary(
                sorted_releases, repository
            )

            # Compute historical measurements
            releases_history_metrics = compute_releases_history_metrics(
                releases_statistics,
                ["stargazers", "issues"],
            )

            log.info(releases_history_metrics)

            # Store in DB
            mo.db["repositories_analysis"].update_one(
                {"full_name": repository["full_name"]},
                {
                    "$set": {
                        "releases_history_metrics": releases_history_metrics,
                        "metadata": {
                            "created": datetime.now(tz=utc),
                            "modified": datetime.now(tz=utc),
                        },
                    }
                },
                upsert=True,
            )
        else:
            log.info("Not enough releases to carry out historical analysis")

        log.info("------------------------")
