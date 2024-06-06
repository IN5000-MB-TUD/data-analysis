import json
import logging
from itertools import product
from pathlib import Path
from connection import mo

import pandas as pd

from data_processing.time_series_n_grams import generate_ngrams

# Setup logging
log = logging.getLogger(__name__)


# Declare constants
PATTERNS_LABELS = ["Steep", "Shallow", "Plateau"]
PATTERNS_MAP = {
    "steep": 0,
    "shallow": 1,
    "plateau": 2,
}
N_TESTS = [2, 3, 4, 5, 6]

if __name__ == "__main__":
    log.info("Evaluate N-grams predictions to find the most suitable N")

    if not Path("../data/bi_grams_probability_table.csv").exists():
        log.warning(
            "The bi_grams_probability_table.cvs file is not present in the data folder. Please run the time_series_n_grams.py script first!"
        )
        exit()

    bi_grams_probability_table = pd.read_csv("../data/bi_grams_probability_table.csv")
    bi_grams_probability = bi_grams_probability_table.to_dict()

    with open("../data/repository_metrics_phases.json") as json_file:
        repository_metrics_phases = json.load(json_file)
    with open("../data/repository_metrics_phases_count.json") as json_file:
        repository_metrics_phases_count = json.load(json_file)

    # Compute n-gram probabilities for all Ns
    n_grams_probability = {}
    for n in N_TESTS:
        sequences = []
        for i in product([0, 1, 2], repeat=n):
            sequences.append(list(i))

        n_grams_probability[f"{n}_grams"] = {}
        for sequence in sequences:
            sequence_tokens = [PATTERNS_LABELS[i].lower() for i in sequence]
            sequence_bi_grams = generate_ngrams(["eos"] + sequence_tokens + ["eos"], 2)
            probability = 1
            for bi_gram in sequence_bi_grams:
                probability *= bi_grams_probability_table.loc[
                    bi_grams_probability_table["Token"] == bi_gram[0]
                ][bi_gram[1]].to_numpy()[0]

            n_grams_probability[f"{n}_grams"][
                " - ".join([PATTERNS_LABELS[i] for i in sequence])
            ] = round(probability, 4)

    # Evaluate the best N performance
    n_grams_accuracy = {}
    for n in N_TESTS:
        total_predictions = 0
        correct_predictions = 0
        deviation_predictions = 0

        for repository_name, repository_metrics in repository_metrics_phases.items():
            for metric, metric_patterns in repository_metrics.items():
                if repository_metrics_phases_count[repository_name][metric] >= n:
                    # Predict next patterns based on probabilities
                    backwards_n = n - 1
                    predicted = [i for i in metric_patterns[:backwards_n]]
                    moving_idx = backwards_n
                    for i in metric_patterns[backwards_n:]:
                        base_patterns = [
                            PATTERNS_LABELS[i].lower()
                            for i in predicted[moving_idx - backwards_n : moving_idx]
                        ]
                        base_patterns = " - ".join(base_patterns)

                        # Look for highest probability
                        highest_probability = -1
                        best_next_pattern = ""
                        for sequence, probability in n_grams_probability[
                            f"{n}_grams"
                        ].items():
                            lower_sequence = sequence.lower()
                            if (
                                lower_sequence.startswith(base_patterns)
                                and probability > highest_probability
                            ):
                                highest_probability = probability
                                best_next_pattern = lower_sequence.split(" - ")[-1]

                        predicted.append(PATTERNS_MAP[best_next_pattern])
                        moving_idx += 1

                    # Count correct predictions
                    for idx, pred_values in enumerate(predicted[backwards_n:]):
                        total_predictions += 1
                        if pred_values == metric_patterns[idx]:
                            correct_predictions += 1
                        deviation_predictions += abs(pred_values - metric_patterns[idx])

        if total_predictions > 0:
            n_grams_accuracy[f"{n}_grams"] = {
                "correct_predictions": correct_predictions,
                "total_predictions": total_predictions,
                "performance": correct_predictions / total_predictions,
                "deviation": deviation_predictions / total_predictions,
            }

    log.info(n_grams_accuracy)

    best_accuracy = -1
    best_n = ""
    for n_key, n_items in n_grams_accuracy.items():
        if n_items["performance"] > best_accuracy:
            best_accuracy = n_items["performance"]
            best_n = n_key
    log.info(f"Best N is {best_n} with accuracy {round(best_accuracy, 3)}")

    log.info("Done!")
