import logging

import joblib
import numpy as np
import pandas as pd

from data_processing.time_series_phases import extrapolate_phases_properties
from utils.main import normalize
from utils.time_series import time_series_phases


# Setup logging
log = logging.getLogger(__name__)


def forecast_scenario_patterns(
    scenario_metric_target,
    scenario_months,
    repository_full_name,
    df_multi_time_series,
    metrics_pattern_hypothesis,
    phases_statistical_properties,
    forecasting_model,
    phases_clustering_model,
):
    """Forecast the pattern for the given metric in the provided scenario"""

    df_time_series = df_multi_time_series.rename(
        columns={scenario_metric_target: "y"}
    ).reset_index(drop=True)

    scenario_ds = (
        list(range(1, scenario_months + 1)) + df_multi_time_series["ds"].iloc[-1]
    )
    normalization_ratio = df_multi_time_series["ds"].iloc[-1] / scenario_ds[-1]
    df_time_series["y"] = df_time_series["y"] * normalization_ratio

    metrics_values_hypothesis = {}
    for metric, metric_pattern in metrics_pattern_hypothesis.items():
        poly_coefficients = [
            phases_statistical_properties[f"phase_{metric_pattern}"]["coeff_3"],
            phases_statistical_properties[f"phase_{metric_pattern}"]["coeff_2"],
            phases_statistical_properties[f"phase_{metric_pattern}"]["coeff_1"],
            phases_statistical_properties[f"phase_{metric_pattern}"]["coeff_0"],
        ]
        y = np.polyval(poly_coefficients, list(range(0, scenario_months))).tolist()

        # Re-balance column
        df_time_series[metric] = df_time_series[metric] * normalization_ratio
        equation_constant = df_time_series[metric].iloc[-1]

        metrics_values_hypothesis[metric] = [
            x + equation_constant for x in normalize(y, t_min=0, t_max=1)
        ]

    df_hypothesis = {
        "ds": scenario_ds,
        "unique_id": [repository_full_name] * scenario_months,
    }
    df_hypothesis.update(metrics_values_hypothesis)
    df_hypothesis = pd.DataFrame(df_hypothesis)

    # Load model and predict pattern
    df_forecast = (
        forecasting_model.predict(
            scenario_months + 12,
            ids=[repository_full_name],
            X_df=df_time_series._append(df_hypothesis, ignore_index=True).drop(
                columns=["y"]
            ),
        )
        .tail(scenario_months)
        .reset_index(drop=True)
    )

    # Evaluate forecasted patterns
    forecasted_metric_values = df_forecast["XGBRegressor"].tolist()
    forecasted_metric_phases = time_series_phases(forecasted_metric_values)
    df_forecasted_metric_phases_features = extrapolate_phases_properties(
        forecasted_metric_phases, forecasted_metric_values
    )

    forecasted_clustered_phases = phases_clustering_model.predict(
        df_forecasted_metric_phases_features.drop(columns=["phase_order"])
    )

    return forecasted_metric_phases, forecasted_clustered_phases


def forecast_scenario_values(
    latest_metric_value,
    forecasted_metric_phases,
    forecasted_clustered_phases,
    phases_statistical_properties,
):
    """Compute the forecast scenario values"""
    forecast_values = []
    for i, metric_phase_idx in enumerate(forecasted_metric_phases):
        metric_pattern = forecasted_clustered_phases[i]
        poly_coefficients = [
            phases_statistical_properties[f"phase_{metric_pattern}"]["coeff_3"],
            phases_statistical_properties[f"phase_{metric_pattern}"]["coeff_2"],
            phases_statistical_properties[f"phase_{metric_pattern}"]["coeff_1"],
            phases_statistical_properties[f"phase_{metric_pattern}"]["coeff_0"],
        ]
        y = np.polyval(poly_coefficients, list(range(0, metric_phase_idx))).tolist()

        forecast_values.extend([int(y_val + latest_metric_value) for y_val in y])
        latest_metric_value = forecast_values[-1]

    return forecast_values


def log_forecast_values(
    scenario_metric_target,
    scenario_months,
    repository_full_name,
    metrics_time_series,
    df_multi_time_series,
    metrics_pattern_hypothesis,
    phases_statistical_properties,
    models_path,
    phases_labels,
    phases_clustering_model,
):
    """Log the forecasted values"""
    for metric, metric_pattern in metrics_pattern_hypothesis.items():
        metric_forecast_values = forecast_scenario_values(
            metrics_time_series[metric]["values"][-1],
            [scenario_months],
            [metric_pattern],
            phases_statistical_properties,
        )

        log.info(
            f"Metric {metric} is assumed to grow following a {phases_labels[metric_pattern]} pattern, going from {metric_forecast_values[0]} to {metric_forecast_values[-1]}, with an overall growth of {metric_forecast_values[-1] - metric_forecast_values[0]}"
        )

    forecasting_model = joblib.load(
        f"{models_path}/mts_forecast_{scenario_metric_target}.pickle"
    )

    forecasted_metric_phases, forecasted_clustered_phases = forecast_scenario_patterns(
        scenario_metric_target,
        scenario_months,
        repository_full_name,
        df_multi_time_series,
        metrics_pattern_hypothesis,
        phases_statistical_properties,
        forecasting_model,
        phases_clustering_model,
    )

    log.info(
        f"The forecasted patterns for the metric {scenario_metric_target} in the next {scenario_months} months are: "
        f"{[phases_labels[phase_id] for phase_id in forecasted_clustered_phases]}"
    )

    latest_metric_value = metrics_time_series[scenario_metric_target]["values"][-1]
    forecast_values = forecast_scenario_values(
        latest_metric_value,
        forecasted_metric_phases,
        forecasted_clustered_phases,
        phases_statistical_properties,
    )

    log.info(
        f"The {scenario_metric_target} value is predicted to go from {forecast_values[0]} to {forecast_values[-1]}, with an overall growth of {forecast_values[-1] - forecast_values[0]}"
    )
