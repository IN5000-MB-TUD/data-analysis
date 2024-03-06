from connection import mo


def get_stargazers_time_series(repository):
    """Get repository stargazers time series"""
    repository_stargazers = mo.db["statistics_stargazers"].find_one({"repository_id": repository["_id"]})
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


def get_issues_time_series(repository):
    """Get repository issues time series"""
    issues_dates = [repository["created_at"]] + [
        issue["created_at"] for issue in repository["statistics"]["issues"].values()
    ]
    issues_dates.sort()
    issues_cumulative = [0]
    issues_counter = 0
    for i in range(1, len(issues_dates)):
        issues_counter += 1
        issues_cumulative.append(issues_counter)

    issues_cumulative.append(repository["open_issues"])
    issues_dates.append(repository["metadata"]["modified"])

    return issues_dates, issues_cumulative
