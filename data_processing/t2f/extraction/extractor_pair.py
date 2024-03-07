import bz2
from math import ceil

import numpy as np
import scipy.spatial.distance as dist

from dtaidistance import dtw
from utils.time_series import (
    compute_time_series_segments_trends,
    compute_pattern_distance,
)


def sax_bins(all_values, ts_months):
    bins = np.percentile(all_values[all_values > 0], np.linspace(0, 100, ts_months + 1))
    bins[0] = 0
    bins[-1] = 1e1000
    return bins


def sax_transform(all_values, bins):
    indices = np.digitize(all_values, bins) - 1
    alphabet = np.array([str(i) for i in range(0, len(bins))])
    text = "".join(alphabet[indices])
    return str.encode(text)


def compression_based_dissimilarity(ts1, ts2, ts_months):
    m_bins = sax_bins(ts1, ts_months)
    n_bins = sax_bins(ts2, ts_months)
    m = sax_transform(ts1, m_bins)
    n = sax_transform(ts2, n_bins)
    len_m = len(bz2.compress(m))
    len_n = len(bz2.compress(n))
    len_combined = len(bz2.compress(m + n))
    return len_combined / (len_m + len_n)


def distance(
    ts1: np.array,
    ts2: np.array,
    ts1_segments: list,
    ts2_segments: list,
    ts_age_months: int,
):
    metrics = {
        # "correlation": dist.correlation,
        # "euclidean": dist.euclidean,
        "pattern": compute_pattern_distance,
        # "dtw": dtw.distance_fast,
        # "cbd": compression_based_dissimilarity,
    }

    distances = {}
    for k, f in metrics.items():
        if k == "pattern":
            distances[k] = f(ts1_segments, ts2_segments)
        elif k == "cbd":
            distances[k] = f(ts1, ts2, ts_age_months)
        else:
            distances[k] = f(ts1, ts2)

    return distances


def extract_pair_features(time_series_1, time_series_2):
    # Create numpy arrays
    ts1 = np.array([time_series_1_value for _, time_series_1_value in time_series_1])
    ts2 = np.array([time_series_2_value for _, time_series_2_value in time_series_2])

    # Compute segments
    ts1_segments = compute_time_series_segments_trends(time_series_1)
    ts2_segments = compute_time_series_segments_trends(time_series_2)

    # Compute months
    ts_age_months = ceil((time_series_1[-1][0] - time_series_1[0][0]) / 2629746)

    features = {}
    distances = distance(ts1, ts2, ts1_segments, ts2_segments, ts_age_months)
    features.update(distances)

    return features
