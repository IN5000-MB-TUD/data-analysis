import csv
import logging
from datetime import datetime

from pytz import utc

from connection import mo
from data_mining.repositories import GitHubAPI

# Setup logging
log = logging.getLogger(__name__)

DATA_PATH = "../data"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

if __name__ == "__main__":
    log.info("Start GitHub data mining")
    github_api_client = GitHubAPI()

    # Open repositories list
    with open(f"{DATA_PATH}/Candidates.csv", newline="") as csv_file:
        repositories_list_reader = csv.reader(csv_file, delimiter=",")
        for row in repositories_list_reader:
            repository_url_split = row[1].split("/")
            repository_owner = repository_url_split[-2]
            repository_name = repository_url_split[-1]

            github_api_data = github_api_client.get_repository_data(
                repository_owner, repository_name
            )

            if not github_api_data:
                continue

            repository_data = {
                "owner": repository_owner,
                "name": repository_name,
                "full_name": github_api_data["full_name"],
                "private": github_api_data["private"],
                "description": github_api_data["description"],
                "fork": github_api_data["fork"],
                "forks_count": github_api_data["forks_count"],
                "watchers": github_api_data["watchers"],
                "stargazers_count": github_api_data["stargazers_count"],
                "size": github_api_data["size"],
                "default_branch": github_api_data["default_branch"],
                "open_issues": github_api_data["open_issues"],
                "is_template": github_api_data["is_template"],
                "topics": github_api_data["topics"],
                "has_issues": github_api_data["has_issues"],
                "has_projects": github_api_data["has_projects"],
                "has_wiki": github_api_data["has_wiki"],
                "has_pages": github_api_data["has_pages"],
                "has_downloads": github_api_data["has_downloads"],
                "has_discussions": github_api_data["has_discussions"],
                "archived": github_api_data["archived"],
                "disabled": github_api_data["disabled"],
                "subscribers_count": github_api_data["subscribers_count"],
                "network_count": github_api_data["network_count"],
                "license": github_api_data["license"]["key"]
                if github_api_data["license"]
                else None,
                "pushed_at": datetime.strptime(
                    github_api_data["pushed_at"], DATE_FORMAT
                ).replace(tzinfo=utc),
                "created_at": datetime.strptime(
                    github_api_data["created_at"], DATE_FORMAT
                ).replace(tzinfo=utc),
                "updated_at": datetime.strptime(
                    github_api_data["updated_at"], DATE_FORMAT
                ).replace(tzinfo=utc),
                "metadata": {
                    "created": datetime.now(tz=utc),
                    "modified": datetime.now(tz=utc),
                },
            }

            # Store in DB
            mo.db["repositories_data"].update_one(
                {"full_name": github_api_data["full_name"]},
                {"$set": repository_data},
                upsert=True,
            )
            log.info(f"Successfully updated {repository_owner}/{repository_name}")
