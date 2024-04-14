from connection import mo


def get_metrics_information():
    """Return a list of the metrics information."""
    return [
        ("statistics_commits", "commits", "date", "commits"),
        ("statistics_commits", "contributors", "first_commit", None),
        ("statistics_deployments", "deployments", "created_at", None),
        ("statistics_issues", "issues", "created_at", "open_issues"),
        ("statistics_forks", "forks", "created_at", "forks_count"),
        ("statistics_pull_requests", "pull_requests", "created_at", None),
        ("statistics_workflow_runs", "workflows", "created_at", None),
    ]


def get_stargazers_time_series(repository):
    """Get repository stargazers time series"""
    repository_stargazers = mo.db["statistics_stargazers"].find_one(
        {"repository_id": repository["_id"]}
    )
    if not repository_stargazers:
        return [], []

    stargazers = [repository["created_at"]] + repository_stargazers["stargazers"]
    stargazers_cumulative = [0]
    stargazers_counter = 0
    for i in range(1, len(stargazers)):
        stargazers_counter += 1
        stargazers_cumulative.append(stargazers_counter)

    stargazers_cumulative.append(repository["stargazers_count"])
    stargazers.append(repository["metadata"]["modified"])

    return stargazers, stargazers_cumulative


def get_releases_time_series(repository):
    """Get repository releases time series"""
    repository_releases = repository["releases"]
    if repository_releases is None:
        repository_releases = {}

    releases_dates = [repository["created_at"]] + [
        release["created_at"] for release in repository_releases.values()
    ]
    releases_dates.sort()
    releases_cumulative = [0]
    releases_counter = 0
    for i in range(1, len(releases_dates)):
        releases_counter += 1
        releases_cumulative.append(releases_counter)

    releases_cumulative.append(releases_counter)
    releases_dates.append(repository["metadata"]["modified"])

    return releases_dates, releases_cumulative


def get_size_time_series(repository):
    """Get repository size time series"""
    repository_size = mo.db["statistics_size"].find_one(
        {"repository_id": repository["_id"]}
    )
    if not repository_size:
        return [], [], []

    repository_size_dates = [repository["created_at"]]
    repository_size_cumulative_total = [0]
    repository_size_cumulative_difference = [0]
    size_total_counter = 0
    size_difference_counter = 0
    for size_action in repository_size["size"].values():
        size_total_counter += size_action["total"]
        repository_size_cumulative_total.append(size_total_counter)

        size_difference_counter += size_action["size"]
        repository_size_cumulative_difference.append(size_difference_counter)

        repository_size_dates.append(size_action["date"])

    repository_size_cumulative_total.append(size_total_counter)
    repository_size_cumulative_difference.append(size_difference_counter)
    repository_size_dates.append(repository["metadata"]["modified"])

    return (
        repository_size_dates,
        repository_size_cumulative_total,
        repository_size_cumulative_difference,
    )


def get_metric_time_series(
    repository, metric_collection, metric_name, date_field, total_value=None
):
    """Get repository metric time series"""
    repository_metric = mo.db[metric_collection].find_one(
        {"repository_id": repository["_id"]}
    )
    if not repository_metric:
        return [], []

    metric_dates = [repository["created_at"]] + [
        metric[date_field] for metric in repository_metric[metric_name].values()
    ]
    metric_dates.sort()
    metric_cumulative = [0]
    metric_counter = 0
    for i in range(1, len(metric_dates)):
        metric_counter += 1
        metric_cumulative.append(metric_counter)

    if total_value is not None:
        metric_cumulative.append(repository[total_value])
    else:
        metric_cumulative.append(metric_counter)
    metric_dates.append(repository["metadata"]["modified"])

    return metric_dates, metric_cumulative
