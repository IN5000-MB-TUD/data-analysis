from datetime import timedelta, datetime
from itertools import groupby

import numpy as np
import ruptures as rpt
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from pytz import utc
from ruptures.exceptions import BadSegmentationParameters


def group_util(date, min_date):
    return (date - min_date).days // 31


def group_metric_by_month(dates, total_months, min_date, monotonic=True):
    """Group given list of dates by month."""
    if not dates:
        return []

    dates_grouped = []
    dates.sort()

    for key, val in groupby(dates, key=lambda date: group_util(date, min_date)):
        # Keep only months that are >= 0
        if key >= 0:
            dates_grouped.append((key, list(val)))

    time_series_cumulative_by_month = []
    metric_counter = -1
    dates_grouped_idx = 0
    grouped_months_count = len(dates_grouped)
    for month_idx in range(total_months + 1):
        if (
            dates_grouped_idx < grouped_months_count
            and month_idx == dates_grouped[dates_grouped_idx][0]
        ):
            if monotonic:
                metric_counter += len(dates_grouped[dates_grouped_idx][1])
            else:
                metric_counter = len(dates_grouped[dates_grouped_idx][1])
            dates_grouped_idx += 1

        time_series_cumulative_by_month.append(
            (min_date + relativedelta(months=month_idx), metric_counter)
        )

    return time_series_cumulative_by_month


def time_series_phases(time_series, show_plot=False, n_phases=None, window_size=12):
    if not time_series:
        return []

    # Check if list is made of tuples or integers
    if isinstance(time_series[0], tuple):
        time_series_np = np.array([value for _, value in time_series], dtype="int")
    else:
        time_series_np = np.array(time_series, dtype="int")

    # Setup Window model with L2 cost function
    model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
    algo = rpt.Window(width=min(window_size, time_series_np.shape[0] - 1), model=model).fit(
        time_series_np
    )

    pen = np.log(time_series_np.shape[0]) * 1 * time_series_np.std() ** 2

    # Predict break points based on the given n_phases.
    if n_phases is None:
        phases_break_points = algo.predict(pen=pen)
    else:
        try:
            phases_break_points = algo.predict(n_bkps=n_phases - 1)
        except BadSegmentationParameters:
            phases_break_points = algo.predict(pen=pen)

    if show_plot:
        rpt.show.display(
            time_series_np, phases_break_points, phases_break_points, figsize=(10, 6)
        )
        plt.show()

    return phases_break_points


def build_time_series(repository, max_count_key):
    """Build time series based on repository age and variable max value"""
    if repository[max_count_key] == 0:
        return [], []

    variable_timestamps = []
    variable_cumulative = []
    variable_counter = 0
    stargazers_time_delta = timedelta(
        seconds=int(repository["age"] / repository[max_count_key])
    )
    time_counter = repository["created_at"]
    while time_counter < repository["metadata"]["modified"]:
        variable_timestamps.append(time_counter)
        variable_cumulative.append(variable_counter)

        variable_counter += 1
        time_counter += stargazers_time_delta

    return variable_timestamps, variable_cumulative


def merge_time_series(time_series_1, time_series_2):
    """Merge two time series to align the time sequence"""
    if not time_series_1 and not time_series_2:
        current_time = datetime.now(tz=utc)
        return [(current_time, 0)], [(current_time, 0)]

    time_series_1_times = [t for (t, _) in time_series_1]
    time_series_2_times = [t for (t, _) in time_series_2]

    time_series_times = time_series_1_times + list(
        set(time_series_2_times) - set(time_series_1_times)
    )
    time_series_times.sort()

    time_series_1_adjusted = []
    time_series_2_adjusted = []

    time_series_1_idx = 0
    time_series_2_idx = 0

    # Make sure that the lists are populated
    if not time_series_1:
        time_series_1 = [(time_series_times[0], 0)]

    if not time_series_2:
        time_series_2 = [(time_series_times[0], 0)]

    for time_series_timestamp in time_series_times:
        # Time Series 1
        if time_series_1_idx < len(time_series_1):
            if time_series_timestamp == time_series_1[time_series_1_idx][0]:
                time_series_1_adjusted.append(
                    (time_series_timestamp, time_series_1[time_series_1_idx][1])
                )
                time_series_1_idx += 1
            else:
                time_series_1_adjusted.append(
                    (time_series_timestamp, time_series_1[time_series_1_idx][1])
                )
        else:
            time_series_1_adjusted.append((time_series_timestamp, time_series_1[-1][1]))

        # Time Series 2
        if time_series_2_idx < len(time_series_2):
            if time_series_timestamp == time_series_2[time_series_2_idx][0]:
                time_series_2_adjusted.append(
                    (time_series_timestamp, time_series_2[time_series_2_idx][1])
                )
                time_series_2_idx += 1
            else:
                time_series_2_adjusted.append(
                    (time_series_timestamp, time_series_2[time_series_2_idx][1])
                )
        else:
            time_series_2_adjusted.append((time_series_timestamp, time_series_2[-1][1]))

    return time_series_1_adjusted, time_series_2_adjusted


def compute_time_series_segments_trends(time_series):
    """Compute the segments trends in a time series"""
    segments_trends = []
    for i in range(1, len(time_series)):
        previous_idx = i - 1
        current_idx = i

        trend_timestamp = time_series[current_idx][0]
        if time_series[previous_idx][1] < time_series[current_idx][1]:
            trend_status = 1
        elif time_series[previous_idx][1] == time_series[current_idx][1]:
            trend_status = 0
        else:
            trend_status = -1

        segments_trends.append((trend_timestamp, trend_status))

    return segments_trends


def merge_segments_trends(trend_1, trend_2):
    """Merge two segment trends to align the time sequence"""
    if not trend_1 and not trend_2:
        current_time = int(datetime.now(tz=utc).timestamp())
        return [(current_time, 0)], [(current_time, 0)]

    trend_1_times = [t for (t, _) in trend_1]
    trend_2_times = [t for (t, _) in trend_2]

    trends_times = trend_1_times + list(set(trend_2_times) - set(trend_1_times))
    trends_times.sort()

    trends_1_adjusted = []
    trends_2_adjusted = []

    trends_1_idx = 0
    trends_2_idx = 0

    # Make sure that the lists are populated
    if not trend_1:
        trend_1 = [(trends_times[0], 0)]

    if not trend_2:
        trend_2 = [(trends_times[0], 0)]

    for trend_timestamp in trends_times:
        # Trends 1
        if trends_1_idx < len(trend_1):
            if trend_timestamp == trend_1[trends_1_idx][0]:
                trends_1_adjusted.append((trend_timestamp, trend_1[trends_1_idx][1]))
                trends_1_idx += 1
            else:
                trends_1_adjusted.append((trend_timestamp, 0))
        else:
            trends_1_adjusted.append((trend_timestamp, 0))

        # Trends 2
        if trends_2_idx < len(trend_2):
            if trend_timestamp == trend_2[trends_2_idx][0]:
                trends_2_adjusted.append((trend_timestamp, trend_2[trends_2_idx][1]))
                trends_2_idx += 1
            else:
                trends_2_adjusted.append((trend_timestamp, 0))
        else:
            trends_2_adjusted.append((trend_timestamp, 0))

    return trends_1_adjusted, trends_2_adjusted


def compute_pattern_distance(trend_1, trend_2):
    """Compute pattern distance between two trends"""
    # Align trends times
    trends_1_adjusted, trends_2_adjusted = merge_segments_trends(trend_1, trend_2)

    pattern_distance = 0
    for i in range(1, len(trends_1_adjusted)):
        pattern_distance += (
            trends_1_adjusted[i][0] - trends_1_adjusted[i - 1][0]
        ) * abs(trends_1_adjusted[i][1] - trends_2_adjusted[i][1])

    # Divide by latest timestamp
    pattern_distance /= trends_1_adjusted[-1][0]

    return pattern_distance
