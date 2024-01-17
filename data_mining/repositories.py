import logging
import os

import requests

# Setup logging
log = logging.getLogger(__name__)


class GitHubAPI:
    """GitHubAPI repositories data collector"""

    def __init__(self):
        """Initialize GitHubAPI class"""
        self.github_api_url = "https://api.github.com/repos/"
        self.github_auth_token = os.getenv("GITHUB_AUTH_TOKEN")

    def get_repository_data(self, repository_owner, repository_name):
        """
        Get data from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The data from the GitHubAPI as a dictionary. None if an error occurs.
        """
        if not self.github_auth_token:
            log.warning(
                "Please provide a valid GITHUB_AUTH_TOKEN in your environment variables!"
            )
            return None

        headers = {"Authorization": "Bearer " + self.github_auth_token}
        request_url = self.github_api_url + f"{repository_owner}/{repository_name}"

        github_api_response = requests.get(request_url, headers=headers)
        if github_api_response.status_code != 200:
            log.error(
                f"The request for repository {repository_owner}/{repository_name} returned a status code {github_api_response.status_code}: {github_api_response.reason}"
            )
            return None

        return github_api_response.json()
