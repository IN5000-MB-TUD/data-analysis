import logging
import os
from datetime import datetime
from math import ceil

import requests
from pytz import utc

# Setup logging
log = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DATE_FORMAT_MS = "%Y-%m-%dT%H:%M:%S.000%z"


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
            return None if github_api_response.status_code != 422 else {}

        return github_api_response.json()

    def _request_page_increment(self, repository_age, item_count):
        """
        Compute the page increment for an API request to sample the data by month.

        :param repository_age: The age of the repository, in seconds.
        :param item_count: The total count of the item to sample.
        :return: The API request page increment.
        """
        repository_age_months = ceil(repository_age / 2629746)
        requests_count = ceil(item_count / 100)
        time_series_stargazers_increment = max(
            1, int(requests_count / repository_age_months)
        )
        return time_series_stargazers_increment

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

        if github_api_response is None:
            return None

        return {
            branch["name"]: {"protected": branch["protected"]}
            for branch in github_api_response
        }

    def get_repository_commits_count(self, repository_owner, repository_name):
        """
        Get commits count from GitHub.
        The commits are taken from the main branch.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The commits count from the GitHubAPI. None if an error occurs.
        """
        request_url = (
            self.github_api_url
            + f"{repository_owner}/{repository_name}/commits?per_page=1&page=1"
        )

        if not self._is_authenticated():
            return None

        github_api_response = requests.get(request_url, headers=self.headers)
        if github_api_response.status_code != 200:
            log.error(
                f"The request for repository {repository_owner}/{repository_name} returned a status code "
                f"{github_api_response.status_code}: {github_api_response.reason}"
            )
            return None

        github_api_response_headers = github_api_response.headers
        commits_count = int(
            github_api_response_headers["Link"]
            .split('>; rel="last"')[0]
            .split("&page=")[-1]
        )

        return commits_count

    def get_repository_commits(self, repository):
        """
        Get commits from GitHub.
        The commits are taken from the main branch.

        :param repository: The repository data.
        :return: The commits from the GitHubAPI. None if an error occurs.
        """
        response_page = 1
        response_page_increment = self._request_page_increment(
            repository["age"], repository["commits"]
        )
        commits_to_add = 100
        commits_dates = {}
        commits_contributors = {}

        repository_owner = repository["owner"]
        repository_name = repository["name"]
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
            if github_api_response is None:
                return None, None
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

            response_page += response_page_increment

        return commits_dates, commits_contributors

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
            if github_api_response is None:
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
        if github_api_response is None:
            return None

        return len(github_api_response.get("sbom", {}).get("packages", []))

    def get_repository_stargazers_time(self, repository):
        """
        Get stargazers time data from GitHub.

        :param repository: the repository data.
        :return: The stargazers time data from the GitHubAPI as a list. None if an error occurs.
        """
        headers = self.headers
        headers["Accept"] = "application/vnd.github.v3.star+json"
        response_page = 1
        response_page_increment = self._request_page_increment(
            repository["age"], repository["stargazers_count"]
        )
        stargazers_to_add = 100
        repository_stargazers = []

        repository_owner = repository["owner"]
        repository_name = repository["name"]
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
            if github_api_response is None:
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
            response_page += response_page_increment

        return repository_stargazers

    def get_repository_issues_time(self, repository):
        """
        Get issues time data from GitHub.

        :param repository: the repository data.
        :return: The issues time data from the GitHubAPI as a dictionary. None if an error occurs.
        """
        response_page = 1
        response_page_increment = self._request_page_increment(
            repository["age"], repository["open_issues"]
        )
        issues_to_add = 100
        repository_issues = {}

        repository_owner = repository["owner"]
        repository_name = repository["name"]
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
            if github_api_response is None:
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
            response_page += response_page_increment

        return repository_issues

    def get_repository_workflows(self, repository_owner, repository_name):
        """
        Get workflows data from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The repository workflow data. None if an error occurs.
        """
        response_page = 1
        workflows_to_add = 100
        repository_workflows = {}
        request_url = (
            self.github_api_url
            + f"{repository_owner}/{repository_name}/actions/workflows?per_page=100"
        )

        while workflows_to_add > 0:
            github_api_response = self._make_request(
                request_url + f"&page={response_page}",
                self.headers,
                repository_owner,
                repository_name,
            )
            if github_api_response is None:
                return None
            else:
                workflows_to_add = len(github_api_response.get("workflows", []))
                for workflow in github_api_response.get("workflows", []):
                    repository_workflows.update(
                        {
                            f"workflow_{workflow['id']}": {
                                "id": workflow["id"],
                                "name": workflow["name"],
                                "created_at": datetime.strptime(
                                    workflow["created_at"], DATE_FORMAT_MS
                                ).replace(tzinfo=utc),
                            }
                        }
                    )
            response_page += 1

        return repository_workflows

    def get_repository_workflows_time(self, repository_owner, repository_name):
        """
        Get workflows time series data from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The repository workflow time series data. None if an error occurs.
        """
        response_page = 1
        workflow_runs_to_add = 100
        repository_workflow_runs = {}
        request_url = (
            self.github_api_url
            + f"{repository_owner}/{repository_name}/actions/runs?status=success&per_page=100"
        )

        while workflow_runs_to_add > 0:
            github_api_response = self._make_request(
                request_url + f"&page={response_page}",
                self.headers,
                repository_owner,
                repository_name,
            )
            if github_api_response is None:
                return None
            else:
                workflow_runs_to_add = len(github_api_response.get("workflow_runs", []))
                for workflow_run in github_api_response.get("workflow_runs", []):
                    repository_workflow_runs.update(
                        {
                            f"run_{workflow_run['id']}": {
                                "id": workflow_run["id"],
                                "workflow_id": workflow_run["workflow_id"],
                                "created_at": datetime.strptime(
                                    workflow_run["created_at"], DATE_FORMAT
                                ).replace(tzinfo=utc),
                            }
                        }
                    )
            response_page += 1

        return repository_workflow_runs

    def get_repository_environments(self, repository_owner, repository_name):
        """
        Get environments data from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The repository environments data. None if an error occurs.
        """
        response_page = 1
        environments_to_add = 100
        repository_environments = {}
        request_url = (
            self.github_api_url
            + f"{repository_owner}/{repository_name}/environments?per_page=100"
        )

        while environments_to_add > 0:
            github_api_response = self._make_request(
                request_url + f"&page={response_page}",
                self.headers,
                repository_owner,
                repository_name,
            )
            if github_api_response is None:
                return None
            else:
                environments_to_add = len(github_api_response.get("environments", []))
                for environment in github_api_response.get("environments", []):
                    repository_environments.update(
                        {
                            f"env_{environment['id']}": {
                                "id": environment["id"],
                                "name": environment["name"],
                                "created_at": datetime.strptime(
                                    environment["created_at"], DATE_FORMAT
                                ).replace(tzinfo=utc),
                            }
                        }
                    )
            response_page += 1

        return repository_environments

    def get_repository_deployments(
        self, repository_owner, repository_name, environments
    ):
        """
        Get deployments time series data from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :param environments: The repository environments.
        :return: The repository deployments time series data. None if an error occurs.
        """
        response_page = 1
        deployments_to_add = 100
        repository_deployments = {}

        production_environment = "production"
        for environment in environments:
            if environment["name"].lower() in {"production", "release"}:
                production_environment = environment["name"]
                break

        request_url = (
            self.github_api_url
            + f"{repository_owner}/{repository_name}/deployments?environment={production_environment}per_page=100"
        )

        while deployments_to_add > 0:
            github_api_response = self._make_request(
                request_url + f"&page={response_page}",
                self.headers,
                repository_owner,
                repository_name,
            )
            if github_api_response is None:
                return None
            else:
                deployments_to_add = len(github_api_response)
                for deployment in github_api_response:
                    repository_deployments.update(
                        {
                            f"deployment_{deployment['id']}": {
                                "id": deployment["id"],
                                "environment": deployment["environment"],
                                "transient_environment": deployment[
                                    "transient_environment"
                                ],
                                "production_environment": deployment[
                                    "production_environment"
                                ],
                                "created_at": datetime.strptime(
                                    deployment["created_at"], DATE_FORMAT
                                ).replace(tzinfo=utc),
                            }
                        }
                    )
            response_page += 1

        return repository_deployments

    def get_repository_pull_requests_time(self, repository_owner, repository_name):
        """
        Get pull request time series data from GitHub.

        :param repository_owner: The owner of the repository.
        :param repository_name: The name of the repository.
        :return: The repository pull request time series data. None if an error occurs.
        """
        response_page = 1
        pull_request_to_add = 100
        repository_pull_request = {}
        request_url = (
            self.github_api_url
            + f"{repository_owner}/{repository_name}/pulls?per_page=100"
        )

        while pull_request_to_add > 0:
            github_api_response = self._make_request(
                request_url + f"&page={response_page}",
                self.headers,
                repository_owner,
                repository_name,
            )
            if github_api_response is None:
                return None
            else:
                pull_request_to_add = len(github_api_response)
                for pull_request in github_api_response:
                    repository_pull_request.update(
                        {
                            f"pull_{pull_request['number']}": {
                                "id": pull_request["id"],
                                "number": pull_request["number"],
                                "state": pull_request["state"],
                                "created_at": datetime.strptime(
                                    pull_request["created_at"], DATE_FORMAT
                                ).replace(tzinfo=utc),
                                "closed_at": datetime.strptime(
                                    pull_request["closed_at"], DATE_FORMAT
                                ).replace(tzinfo=utc)
                                if pull_request["closed_at"]
                                else None,
                                "merged_at": datetime.strptime(
                                    pull_request["merged_at"], DATE_FORMAT
                                ).replace(tzinfo=utc)
                                if pull_request["merged_at"]
                                else None,
                            }
                        }
                    )
            response_page += 1

        return repository_pull_request

    def get_repository_forks(self, repository):
        """
        Get forks time series data from GitHub.

        :param repository: the repository data.
        :return: The repository forks time series data. None if an error occurs.
        """
        response_page = 1
        response_page_increment = self._request_page_increment(
            repository["age"], repository["forks_count"]
        )
        forks_to_add = 100
        repository_forks = {}

        repository_owner = repository["owner"]
        repository_name = repository["name"]
        request_url = (
            self.github_api_url
            + f"{repository_owner}/{repository_name}/forks?per_page=100"
        )

        while forks_to_add > 0:
            github_api_response = self._make_request(
                request_url + f"&page={response_page}",
                self.headers,
                repository_owner,
                repository_name,
            )
            if github_api_response is None:
                return None
            else:
                forks_to_add = len(github_api_response)
                for fork in github_api_response:
                    repository_forks.update(
                        {
                            f"fork_{fork['id']}": {
                                "id": fork["id"],
                                "full_name": fork["full_name"],
                                "created_at": datetime.strptime(
                                    fork["created_at"], DATE_FORMAT
                                ).replace(tzinfo=utc),
                            }
                        }
                    )
            response_page += response_page_increment

        return repository_forks
