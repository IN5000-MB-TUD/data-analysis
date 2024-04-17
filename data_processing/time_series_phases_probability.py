import json
import logging
from copy import copy
from pathlib import Path

import pandas as pd

from connection import mo

# Setup logging
log = logging.getLogger(__name__)


if __name__ == "__main__":
    log.info("Start computing phases probabilities")

    # Retrieve phases from database
    evolution_phases = mo.db["evolution_phases"].find()
    phases_names = {}
    for phase in evolution_phases:
        phases_names[f"phase_{phase['phase_id']}"] = {
            "phase_id": phase["phase_id"],
            "phase_name": phase["phase_name"],
        }

    PHASES = len(phases_names)

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
        phases_combinations["first"] = 0
        phases_combinations["last"] = 0
        phases_combinations["middle"] = 0
        phases_combinations[f"prev_{i}"] = 0
        phases_combinations[f"next_{i}"] = 0
        for k in range(0, PHASES):
            phases_combinations[f"prev_{i}_next_{k}"] = 0

    phases_occurrences = {}
    for i in range(0, PHASES):
        phases_occurrences[f"phase_{i}"] = copy(phases_combinations)

    # Add occurrences
    first_total = 0
    last_total = 0
    middle_total = 0
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
                    phases_occurrences[f"phase_{phase}"]["first"] += 1
                    first_total += 1
                    phases_occurrences[f"phase_{phase}"][f"next_{phases[i + 1]}"] += 1
                elif i == phases_count - 1:
                    phases_occurrences[f"phase_{phase}"]["last"] += 1
                    last_total += 1
                    phases_occurrences[f"phase_{phase}"][f"prev_{phases[i - 1]}"] += 1
                else:
                    phases_occurrences[f"phase_{phase}"]["middle"] += 1
                    middle_total += 1
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
        phases_probabilities[phase]["first"] = round(
            phases_occurrences[phase]["first"] / first_total, 2
        )
        phases_probabilities[phase]["last"] = round(
            phases_occurrences[phase]["last"] / last_total, 2
        )
        phases_probabilities[phase]["middle"] = round(
            phases_occurrences[phase]["middle"] / middle_total, 2
        )

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
    log.info("Probabilities computed, storing them in the database...")
    df_rows = []
    for phase, probabilities in phases_probabilities.items():
        mo.db["evolution_phases_probabilities"].update_one(
            phases_names[phase], {"$set": probabilities}, upsert=True
        )
        df_rows.append({**phases_names[phase], **probabilities})

    # Save csv file
    df_time_series_phases_probabilities = pd.DataFrame(df_rows)
    df_time_series_phases_probabilities.to_csv(
        "../data/time_series_phases_probabilities.csv", index=False
    )

    log.info("Done!")
