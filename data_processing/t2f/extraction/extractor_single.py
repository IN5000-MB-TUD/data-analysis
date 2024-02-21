from datetime import datetime

import numpy as np
import pandas as pd
from pytz import utc
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters


# Adapt Multivariate time series for extracting features.
def adapt_time_series(ts, sensors_name):
    list_multi_ts = {}
    for i, k in enumerate(sensors_name):
        list_multi_ts[k] = list(ts[i])

    list_id = [1 for _ in ts]
    list_time = [i for i in range(len(ts))]

    dict_df = {"id": list_id, "time": list_time}
    for sensor in sensors_name:
        dict_df[sensor] = list_multi_ts[sensor]

    df_time_series = pd.DataFrame(dict_df)
    return df_time_series


def extract_sensors_features(
    ts: np.array, sensors_name: list, feats_select: dict = None
):
    """Extract features from each signal in the time series as univariate type"""
    signal_features = pd.DataFrame()

    for i, k in enumerate(sensors_name):
        signal_ts = list(ts[i])
        signal_ts_tuples = [
            (k, signal_ts[i][0].timestamp() * 1000, signal_ts[i][1])
            for i in range(len(signal_ts))
        ]

        if not signal_ts_tuples:
            signal_ts_tuples = [(k, datetime.now(tz=utc).timestamp() * 1000, 0)]

        signal_ts_df = pd.DataFrame(signal_ts_tuples, columns=["id", "time", "value"])

        extracted_features = extract_features(
            signal_ts_df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=MinimalFCParameters(),
        )

        signal_features = pd.concat(
            [signal_features, extracted_features], ignore_index=False
        )

    return signal_features


def extract_univariate_features(
    ts: np.array, sensors_name: list, feats_select: dict = None
):
    features_extracted = extract_sensors_features(ts, sensors_name, feats_select)
    features = features_extracted.T.astype(float).to_dict()
    return features
