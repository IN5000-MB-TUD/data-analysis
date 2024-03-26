import json
import logging
import random
from copy import copy
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from connection import mo

# Setup logging
log = logging.getLogger(__name__)

PHASES = 3

if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval from Database")

    if not Path("../data/repository_metrics_phases.json").exists():
        log.warning(
            "The repository_metrics_phases.json file is not present in the data folder. Please run the time_series_phases.py script first!"
        )
        exit()

    with open("../data/repository_metrics_phases.json") as json_file:
        repository_metrics_phases = json.load(json_file)

    # Initialize phases object
    phases_combinations = {}
    for i in range(0, PHASES):
        phases_combinations[f"prev_{i}"] = 0
        phases_combinations[f"next_{i}"] = 0
        for k in range(0, PHASES):
            phases_combinations[f"prev_{i}_next_{k}"] = 0

    phases_occurrences = {}
    for i in range(0, PHASES):
        phases_occurrences[f"phase_{i}"] = copy(phases_combinations)

    # Add occurrences
    for _, metrics_phases in repository_metrics_phases.items():
        for phases in metrics_phases.values():
            phases_count = len(phases)
            # Single phases do not count
            if phases_count <= 1:
                continue
            # Count phases occurrence
            for i in range(0, phases_count):
                phase = phases[i]
                if i == 0:
                    phases_occurrences[f"phase_{phase}"][f"next_{phases[i + 1]}"] += 1
                elif i == phases_count - 1:
                    phases_occurrences[f"phase_{phase}"][f"prev_{phases[i - 1]}"] += 1
                else:
                    phases_occurrences[f"phase_{phase}"][f"next_{phases[i + 1]}"] += 1
                    phases_occurrences[f"phase_{phase}"][f"prev_{phases[i - 1]}"] += 1
                    phases_occurrences[f"phase_{phase}"][
                        f"prev_{phases[i - 1]}_next_{phases[i + 1]}"
                    ] += 1

    # Compute probabilities
    phases_probabilities = {}
    for i in range(PHASES):
        phases_probabilities[f"phase_{i}"] = copy(phases_combinations)

    for phase, _ in phases_probabilities.items():
        # Compute totals
        prev_total = 0
        next_total = 0
        prev_next_totals = [0] * PHASES
        for i in range(PHASES):
            prev_total += phases_occurrences[phase][f"prev_{i}"]
            next_total += phases_occurrences[phase][f"next_{i}"]
            for k in range(PHASES):
                prev_next_totals[i] += phases_occurrences[phase][f"prev_{i}_next_{k}"]

        # Compute probabilities
        for i in range(PHASES):
            phases_probabilities[phase][f"prev_{i}"] = round(
                phases_occurrences[phase][f"prev_{i}"] / prev_total, 2
            )
            phases_probabilities[phase][f"next_{i}"] = round(
                phases_occurrences[phase][f"next_{i}"] / next_total, 2
            )
            for k in range(PHASES):
                phases_probabilities[phase][f"prev_{i}_next_{k}"] = round(
                    phases_occurrences[phase][f"prev_{i}_next_{k}"]
                    / prev_next_totals[i],
                    2,
                )

    # Store probabilities in DB
    for phase, probabilities in phases_probabilities.items():
        mo.db["evolution_phases_probabilities"].update_one(
            {"phase": phase}, {"$set": probabilities}, upsert=True
        )

    # Generate sequences based on probabilities
    sequence_length = 10
    phases = [0, 1, 2]
    phases_probability_lists = {}
    for phase in phases:
        phase_id = f"phase_{phase}"
        phases_probability_lists[phase_id] = {
            "next_probabilities": [],
            "prev_next_probabilities": [],
        }
        # Get probabilities
        for i in range(PHASES):
            phases_probability_lists[phase_id]["next_probabilities"].append(
                phases_probabilities[phase_id][f"next_{i}"]
            )
            phases_probability_lists[phase_id]["prev_next_probabilities"].append([])
            for k in range(PHASES):
                phases_probability_lists[phase_id]["prev_next_probabilities"][i].append(
                    phases_probabilities[phase_id][f"prev_{i}_next_{k}"]
                )

    phases_sequences = {}
    for phase in phases:
        phase_id = f"phase_{phase}"
        phases_sequences[phase_id] = [phase]
        for i in range(0, sequence_length - 1):
            next_phase = random.choices(
                phases,
                phases_probability_lists[phase_id]["prev_next_probabilities"][
                    phases_sequences[phase_id][i]
                ],
                k=1,
            )[0]
            phases_sequences[phase_id].append(next_phase)

    # Build and plot time series based on predicted phases
    phases_coefficients = {}
    evolution_phases = mo.db["evolution_phases"].find()
    for phase in evolution_phases:
        phase_id = f"phase_{phase['phase_order']}"
        phases_coefficients[phase_id] = {}
        for key, value in phase.items():
            if "coeff" in key:
                coefficient_id = int(key.split("coeff_")[1][0])
                phases_coefficients[phase_id][f"coefficient_{coefficient_id}"] = value

    phases_time_series = {}
    x = list(range(0, 13))
    for phase_id, sequence in phases_sequences.items():
        phases_time_series[phase_id] = []
        start_value = 0

        for phase in sequence:
            poly_coefficients = [
                phases_coefficients[f"phase_{phase}"][f"coefficient_3"],
                phases_coefficients[f"phase_{phase}"][f"coefficient_2"],
                phases_coefficients[f"phase_{phase}"][f"coefficient_1"],
                phases_coefficients[f"phase_{phase}"][f"coefficient_0"],
            ]
            y = np.polyval(poly_coefficients, x) + start_value
            start_value = y[-1]
            phases_time_series[phase_id].extend(y)

        plt.plot(
            list(range(len(phases_time_series[phase_id]))),
            phases_time_series[phase_id],
            label=f"TS {phase_id}",
        )

    plt.title(f"Phases sequence based on relational probabilities")
    plt.xlabel("Time (Months)")
    plt.ylabel("TS Value")
    plt.legend()
    plt.show()

    log.info("Successfully computed phases probabilities")
