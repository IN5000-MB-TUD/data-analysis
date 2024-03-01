import logging
import os
from datetime import datetime

import requests
from pytz import utc

# Setup logging
log = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class GitHubAPI:
    """GitHubAPI repositories data collector"""

    def __init__(self):
        """Initialize GitHubAPI class"""
        self.github_api_url = "https://api.github.com/repos/"
        self.github_auth_token = os.getenv("GITHUB_AUTH_TOKEN")
        self.github_api_version = "2022-11-28"
        self.headers = {
            "Authorization": "Bearer " + self.github_auth_token,
            "X-GitHub-Api-Version": self.github_api_version,
            "Accept": "application/vnd.github+json",
        }

    def _is_authenticated(self):
        """
        Check the GitHub API authentication.

        :return: True if the token exists. False otherwise.
        """
        if not self.github_auth_token:
            log.warning(
                "Please provide a valid GITHUB_AUTH_TOKEN in your environment variables!"
            )
            return False

        return True

    def _make_request(self, request_url, headers, repository_owner, repository_name):
        """
        Make request to the GitHub API.

        :param request_url: Request URL.
        :param headers: The request headers.
        :param repository_owner: The repository owner.
        :param repository_name: The repository name.
        :return: The response body. None if an error occurs.
        """
        if not self._is_authenticated():
            return None

        github_api_response = requests.get(request_url, headers=headers)
        if github_api_response.status_code != 200:
            log.error(
                f"The request for repository {repository_owner}/{repository_name} returned a status code "
                f"{github_api_response.status_code}: {github_api_response.reason}"
            )
            return None

        return github_api_response.json()

    def get_repository_data(self, repository_owner, repository_name):
        """
        Get data from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The data from the GitHubAPI as a dictionary. None if an error occurs.
        """
        request_url = self.github_api_url + f"{repository_owner}/{repository_name}"

        return self._make_request(
            request_url, self.headers, repository_owner, repository_name
        )

    def get_repository_contributors(self, repository_owner, repository_name):
        """
        Get contributors data from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The contributors data from the GitHubAPI as a dictionary. None if an error occurs.
        """
        request_url = (
            self.github_api_url + f"{repository_owner}/{repository_name}/contributors"
        )

        github_repository_contributors = self._make_request(
            request_url, self.headers, repository_owner, repository_name
        )
        repository_contributors = {}

        for contributor in github_repository_contributors:
            repository_contributors[contributor["login"]] = {
                "id": contributor["id"],
                "type": contributor["type"],
                "admin": contributor["site_admin"],
                "contributions": contributor["contributions"],
            }

        return repository_contributors

    def get_repository_languages(self, repository_owner, repository_name):
        """
        Get programming languages data from GitHub.
        The value shown for each language is the number of bytes of code written in that language.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The programming languages data from the GitHubAPI as a dictionary. None if an error occurs.
        """
        request_url = (
            self.github_api_url + f"{repository_owner}/{repository_name}/languages"
        )

        return self._make_request(
            request_url, self.headers, repository_owner, repository_name
        )

    def get_repository_branches(self, repository_owner, repository_name):
        """
        Get branches data from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The active branches data from the GitHubAPI as a list. None if an error occurs.
        """
        request_url = (
            self.github_api_url + f"{repository_owner}/{repository_name}/branches"
        )

        github_api_response = self._make_request(
            request_url, self.headers, repository_owner, repository_name
        )

        if not github_api_response:
            return None

        return {
            branch["name"]: {"protected": branch["protected"]}
            for branch in github_api_response
        }

    def get_repository_commits(self, repository_owner, repository_name):
        """
        Get commits from GitHub.
        The commits are taken from the main branch.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The commits from the GitHubAPI. None if an error occurs.
        """
        response_page = 1
        commits_count = 0
        commits_to_add = 100
        commits_dates = {}
        commits_contributors = {}
        request_url = (
            self.github_api_url
            + f"{repository_owner}/{repository_name}/commits?per_page=100"
        )

        while commits_to_add > 0:
            github_api_response = self._make_request(
                request_url + f"&page={response_page}",
                self.headers,
                repository_owner,
                repository_name,
            )
            if not github_api_response:
                return None, None, None
            else:
                commits_to_add = len(github_api_response)

                # Retrieve data
                for commit in github_api_response:
                    if (
                        commit.get("author")
                        and commit.get("commit", {}).get("author", {}).get("date", None)
                        and commit.get("author", {}).get("login", None)
                    ):
                        commit_date = datetime.strptime(
                            commit["commit"]["author"]["date"], DATE_FORMAT
                        ).replace(tzinfo=utc)
                        commit_author = commit["author"]["login"]

                        commits_dates[commit["sha"]] = {
                            "author": commit_author,
                            "date": commit_date,
                        }

                        if commit_author not in commits_contributors:
                            commits_contributors[commit_author] = {
                                "first_commit": commit_date,
                                "commits": 1,
                            }
                        else:
                            commits_contributors[commit_author]["commits"] += 1

                        # Increase counter
                        commits_count += 1

            response_page += 1

        return commits_count, commits_dates, commits_contributors

    def get_repository_releases(self, repository_owner, repository_name):
        """
        Get releases data from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The releases data from the GitHubAPI as a dictionary. None if an error occurs.
        """
        response_page = 1
        releases_to_add = 100
        repository_releases = {}
        request_url = (
            self.github_api_url
            + f"{repository_owner}/{repository_name}/releases?per_page=100"
        )

        while releases_to_add > 0:
            github_api_response = self._make_request(
                request_url + f"&page={response_page}",
                self.headers,
                repository_owner,
                repository_name,
            )
            if not github_api_response:
                return None
            else:
                releases_to_add = len(github_api_response)
                for release in github_api_response:
                    release_name = (
                        release["name"] if release["name"] else release["tag_name"]
                    )
                    repository_releases.update(
                        {
                            release_name: {
                                "tag_name": release["tag_name"],
                                "target": release["target_commitish"],
                                "body": release["body"],
                                "draft": release["draft"],
                                "prerelease": release["prerelease"],
                                "created_at": datetime.strptime(
                                    release["created_at"], DATE_FORMAT
                                ).replace(tzinfo=utc),
                                "published_at": datetime.strptime(
                                    release["published_at"], DATE_FORMAT
                                ).replace(tzinfo=utc),
                            }
                        }
                    )
            response_page += 1

        return repository_releases

    def get_repository_dependencies_count(self, repository_owner, repository_name):
        """
        Get repository dependencies count from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The repository dependencies count from the GitHubAPI. None if an error occurs.
        """
        request_url = (
            self.github_api_url
            + f"{repository_owner}/{repository_name}/dependency-graph/sbom"
        )

        github_api_response = self._make_request(
            request_url, self.headers, repository_owner, repository_name
        )
        if not github_api_response:
            return None

        return len(github_api_response.get("sbom", {}).get("packages", []))

    def get_repository_stargazers_time(self, repository_owner, repository_name):
        """
        Get stargazers time data from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The stargazers time data from the GitHubAPI as a list. None if an error occurs.
        """
        headers = self.headers
        headers["Accept"] = "application/vnd.github.v3.star+json"
        response_page = 1
        stargazers_to_add = 100
        repository_stargazers = []
        request_url = (
            self.github_api_url
            + f"{repository_owner}/{repository_name}/stargazers?per_page=100"
        )

        while stargazers_to_add > 0:
            github_api_response = self._make_request(
                request_url + f"&page={response_page}",
                headers,
                repository_owner,
                repository_name,
            )
            if not github_api_response:
                return None
            else:
                stargazers_to_add = len(github_api_response)
                for stargaze in github_api_response:
                    if stargaze.get("starred_at"):
                        repository_stargazers.append(
                            datetime.strptime(
                                stargaze["starred_at"], DATE_FORMAT
                            ).replace(tzinfo=utc),
                        )
            response_page += 1

        return repository_stargazers

    def get_repository_issues_time(self, repository_owner, repository_name):
        """
        Get issues time data from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The issues time data from the GitHubAPI as a dictionary. None if an error occurs.
        """
        response_page = 1
        issues_to_add = 100
        repository_issues = {}
        request_url = (
            self.github_api_url
            + f"{repository_owner}/{repository_name}/issues?per_page=100"
        )

        while issues_to_add > 0:
            github_api_response = self._make_request(
                request_url + f"&page={response_page}",
                self.headers,
                repository_owner,
                repository_name,
            )
            if not github_api_response:
                return None
            else:
                issues_to_add = len(github_api_response)
                for issue in github_api_response:
                    repository_issues.update(
                        {
                            "issue_{}".format(issue["number"]): {
                                "id": issue["id"],
                                "number": issue["number"],
                                "state": issue["state"],
                                "title": issue["title"],
                                "body": issue["body"],
                                "user": issue["user"]["login"],
                                "labels": {
                                    label["name"]: label["description"]
                                    for label in issue["labels"]
                                },
                                "comments": issue["comments"],
                                "created_at": datetime.strptime(
                                    issue["created_at"], DATE_FORMAT
                                ).replace(tzinfo=utc),
                                "updated_at": datetime.strptime(
                                    issue["updated_at"], DATE_FORMAT
                                ).replace(tzinfo=utc),
                                "closed_at": datetime.strptime(
                                    issue["closed_at"], DATE_FORMAT
                                ).replace(tzinfo=utc)
                                if issue["closed_at"]
                                else None,
                                "author_association": issue["author_association"],
                                "state_reason": issue["state_reason"],
                            }
                        }
                    )
            response_page += 1

        return repository_issues
