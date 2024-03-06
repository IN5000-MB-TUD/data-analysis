from connection import mo


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
