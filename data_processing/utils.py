import matplotlib.pyplot as plt


def create_plot(path, title, subtitle, xlabel, ylabel, x, y, labels=None):
    """Create plot and save it as PNG"""
    # Adjusting the figure size
    plt.subplots(figsize=(16, 8))

    # Creating a plot
    for y_idx, y_data in enumerate(y):
        if labels:
            plt.plot(x, y_data, label=labels[y_idx])
        else:
            plt.plot(x, y_data)

    if labels:
        plt.legend(loc="upper left")

    # Adding a plot title and customizing its font size
    plt.suptitle(title, fontsize=20)
    plt.title(subtitle, fontsize=15)

    # Adding axis labels and customizing their font size
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)

    # Rotating axis ticks and customizing their font size
    plt.xticks(rotation=30, fontsize=15)

    # Saving the resulting plot to a file
    plt.savefig(path)


def sorted_simple_dict(d):
    """Sort simple dictionary by keys"""
    return {k: v for k, v in sorted(d.items())}


def sorted_once_nested_dict(d, key):
    """Sort nested dictionary by inner key and remove empty keys"""
    return {
        k: sorted_simple_dict(v)
        for k, v in sorted(d.items(), key=lambda x: x[1][key])
        if k
    }


def nearest(items, pivot):
    """Find nearest index and item in a list to the given item"""
    return min(enumerate(items), key=lambda x: abs(x[1] - pivot))


def get_stargazers_time_series(repository):
    """Get repository stargazers time series"""
    stargazers = [repository["created_at"]] + repository["statistics"]["stargazers"]
    stargazers_cumulative = [0]
    stargazers_counter = 0
    for i in range(1, len(stargazers)):
        stargazers_counter += 1
        stargazers_cumulative.append(stargazers_counter)

    stargazers_cumulative.append(repository["stargazers_count"])
    stargazers.append(repository["updated_at"])

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


def get_additions_deletions_time_series(repository):
    """Get repository additions and deletions time series"""
    commits_dates = [
        commit["timestamp"]
        for commit in repository["statistics"]["commits_weekly"].values()
    ]

    commits_cumulative = []
    additions_cumulative = []
    deletions_cumulative = []
    commits_counter = 0
    additions_counter = 0
    deletions_counter = 0
    for commit in repository["statistics"]["commits_weekly"].values():
        commits_counter += commit["total"]
        commits_cumulative.append(commits_counter)

        additions_counter += commit["additions"]
        additions_cumulative.append(additions_counter)

        deletions_counter += commit["deletions"]
        deletions_cumulative.append(deletions_counter)

    return commits_dates, commits_cumulative, additions_cumulative, deletions_cumulative


def create_releases_statistics_dictionary(releases, repository):
    """Create release statistics dictionary"""
    release_statistics = {}
    for release_key, release_values in releases.items():
        release_statistics[release_key] = {
            "timestamp": release_values["created_at"],
        }

        # Process stargazers
        if (
            repository.get("statistics", {}).get("stargazers")
            and repository["stargazers_count"] > 0
        ):
            stargazers, stargazers_cumulative = get_stargazers_time_series(repository)
            # Find nearest date to release
            nearest_stargazer_date_idx, nearest_stargazer_date = nearest(
                stargazers, release_values["created_at"]
            )
            release_statistics[release_key][
                "stargazers_timestamp"
            ] = nearest_stargazer_date
            release_statistics[release_key]["stargazers"] = stargazers_cumulative[
                nearest_stargazer_date_idx
            ]

        # Process open issues
        if (
            repository.get("statistics", {}).get("issues")
            and repository["open_issues"] > 0
        ):
            issues_dates, issues_cumulative = get_issues_time_series(repository)
            # Find nearest date to release
            nearest_issue_date_idx, nearest_issue_date = nearest(
                issues_dates, release_values["created_at"]
            )
            release_statistics[release_key]["issues_timestamp"] = nearest_issue_date
            release_statistics[release_key]["issues"] = issues_cumulative[
                nearest_issue_date_idx
            ]

        # Process Weekly Commits Stats
        if repository.get("statistics", {}).get("commits_weekly"):
            (
                commits_dates,
                commits_cumulative,
                additions_cumulative,
                deletions_cumulative,
            ) = get_additions_deletions_time_series(repository)
            # Find nearest date to weekly commit
            nearest_commit_date_idx, nearest_commit_date = nearest(
                commits_dates, release_values["created_at"]
            )
            release_statistics[release_key][
                "commits_weekly_timestamp"
            ] = nearest_commit_date
            release_statistics[release_key]["total_changes"] = commits_cumulative[
                nearest_commit_date_idx
            ]
            release_statistics[release_key]["additions"] = additions_cumulative[
                nearest_commit_date_idx
            ]
            release_statistics[release_key]["deletions"] = deletions_cumulative[
                nearest_commit_date_idx
            ]

    return release_statistics


def compute_releases_history_metrics(releases_statistics, properties=None):
    """Compute release history metrics from the provided statistics"""
    if not properties:
        return None

    # Ranks
    min_rank = 0
    max_rank = len(releases_statistics) - 1

    properties_metrics = {k: {"EP": 0, "LEP": 0, "EEP": 0} for k in properties}

    releases_statistics_list = [
        (release_key, statistics)
        for release_key, statistics in releases_statistics.items()
    ]

    for idx in range(1, len(releases_statistics_list) - 1):
        _, statistics = releases_statistics_list[idx]
        _, next_statistics = releases_statistics_list[idx + 1]
        # Compute metrics for properties
        for property_key in properties:
            if property_key in statistics:
                properties_metrics[property_key]["EP"] += abs(
                    statistics[property_key] - next_statistics[property_key]
                )
                properties_metrics[property_key]["LEP"] += abs(
                    statistics[property_key] - next_statistics[property_key]
                ) * pow(2, idx - max_rank)
                properties_metrics[property_key]["EEP"] += abs(
                    statistics[property_key] - next_statistics[property_key]
                ) * pow(2, min_rank - idx + 1)
            else:
                properties_metrics[property_key]["EP"] = None
                properties_metrics[property_key]["LEP"] = None
                properties_metrics[property_key]["EEP"] = None

    return properties_metrics
