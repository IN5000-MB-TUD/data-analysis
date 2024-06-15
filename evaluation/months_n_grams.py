import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from connection import mo
from data_processing.time_series_n_grams import (
    generate_ngrams,
    generate_ngram_freq,
    generate_probability_table,
)

# Setup logging
log = logging.getLogger(__name__)

# Declare constants
PATTERNS_LABELS = ["steep", "shallow", "plateau"]
PATTERNS_MAP = {
    "steep": 0,
    "shallow": 1,
    "plateau": 2,
}
N_TESTS = [2, 3, 4, 7, 13, 25]


if __name__ == "__main__":
    log.info("Evaluate N-grams predictions to find the most suitable N")

    # Open files
    with open("../data/repository_metrics_phases.json") as json_file:
        repository_metrics_phases = json.load(json_file)
    with open("../data/repository_metrics_phases_count.json") as json_file:
        repository_metrics_phases_count = json.load(json_file)
    with open("../data/repository_metrics_phases_idxs.json") as json_file:
        repository_metrics_phases_idxs = json.load(json_file)

    # Split train-test data
    data = np.array(list(repository_metrics_phases.keys()))
    x_train, x_test = train_test_split(data, test_size=0.2)

    x_train = {str(x) for x in x_train}
    x_test = {str(x) for x in x_test}

    # Build the patterns sequence by month for each repository
    corpus = ""
    repository_metrics_patterns_sequence = {}
    for repository_name, repository_metrics in repository_metrics_phases.items():
        repository_metrics_patterns_sequence[repository_name] = {}
        for metric, metric_patterns in repository_metrics.items():
            metric_patterns_sequence = []
            metric_patterns_idxs = [0] + repository_metrics_phases_idxs[
                repository_name
            ][metric]
            patterns_boundaries = list(
                zip(metric_patterns_idxs[:-1], metric_patterns_idxs[1:])
            )
            for pattern_idx, pattern in enumerate(metric_patterns):
                metric_patterns_sequence += [PATTERNS_LABELS[pattern]] * (
                    patterns_boundaries[pattern_idx][1]
                    - patterns_boundaries[pattern_idx][0]
                )
            repository_metrics_patterns_sequence[repository_name][
                metric
            ] = metric_patterns_sequence

            # Add to corpus if repository is in train split and avoid single patterns metrics
            if (
                repository_name in x_train
                and repository_metrics_phases_count[repository_name][metric] > 1
            ):
                corpus += " ".join(metric_patterns_sequence) + ". "

    # Compute bigrams probabilities
    corpus = corpus.lower()
    corpus = "eos " + corpus
    corpus = corpus.replace(".", " eos")
    corpus = corpus[:-1]
    tokens = corpus.split(" ")
    distinct_tokens = list(set(sorted(tokens)))
    log.info(f"Distinct tokens: {distinct_tokens}")

    # Generate tokens frequency
    tokens_frequency = {}
    for token in tokens:
        if token not in tokens_frequency:
            tokens_frequency[token] = 0
        tokens_frequency[token] += 1

    log.info(f"Tokens frequency: {tokens_frequency}")

    # Generate bi-grams probabilities
    bi_grams = generate_ngrams(tokens, 2)
    bi_grams_frequency = generate_ngram_freq(bi_grams)
    bi_grams_probability_table = generate_probability_table(
        distinct_tokens, tokens_frequency, bi_grams_frequency, precision=5
    )

    log.info("Probability Table: \n")
    log.info(bi_grams_probability_table)

    for full_matrix in [True, False]:
        # Build baseline by only repeating the previous pattern for the next months
        baseline_scores = {}

        for n in N_TESTS:
            total_predictions = 0
            correct_predictions = 0
            deviation_predictions = 0

            for (
                repository_name,
                repository_metrics,
            ) in repository_metrics_patterns_sequence.items():
                if repository_name not in x_test:
                    continue

                repository_age_months = len(repository_metrics["commits"])

                for metric, metric_patterns in repository_metrics.items():
                    # Skip if the full evaluation is False and the metric only has one pattern
                    if (
                        not full_matrix
                        and repository_metrics_phases_count[repository_name][metric]
                        <= 1
                    ):
                        continue

                    latest_value = metric_patterns[:n][-1]
                    predicted = [latest_value] * repository_age_months

                    # Count correct predictions
                    for idx, pred_values in enumerate(predicted):
                        total_predictions += 1
                        if pred_values == metric_patterns[idx]:
                            correct_predictions += 1
                        deviation_predictions += abs(
                            PATTERNS_MAP[pred_values]
                            - PATTERNS_MAP[metric_patterns[idx]]
                        )

            if total_predictions > 0:
                baseline_scores[f"{n}_grams"] = {
                    "correct_predictions": correct_predictions,
                    "total_predictions": total_predictions,
                    "performance": correct_predictions / total_predictions,
                    "deviation": deviation_predictions / total_predictions,
                }

        # Evaluate the best N performance on test data
        if full_matrix:
            n_grams_accuracy_output_file = "n_grams_accuracy.json"
        else:
            n_grams_accuracy_output_file = "n_grams_accuracy_not_full_matrix.json"
        if not Path(f"../data/{n_grams_accuracy_output_file}").exists():
            n_grams_accuracy = {}
        else:
            with open(f"../data/{n_grams_accuracy_output_file}") as json_file:
                n_grams_accuracy = json.load(json_file)

        for n in N_TESTS:
            log.info(f"Evaluating {n}-grams...")
            total_predictions = 0
            correct_predictions = 0
            deviation_predictions = 0

            for (
                repository_name,
                repository_metrics,
            ) in repository_metrics_patterns_sequence.items():
                if repository_name not in x_test:
                    continue

                repository_age_months = len(repository_metrics["commits"])
                if repository_age_months < n:
                    continue

                for metric, metric_patterns in repository_metrics.items():
                    # Skip if the full evaluation is False and the metric only has one pattern
                    if (
                        not full_matrix
                        and repository_metrics_phases_count[repository_name][metric]
                        <= 1
                    ):
                        continue

                    # Predict next patterns based on probabilities
                    backwards_n = n - 1
                    predicted = [i for i in metric_patterns[:backwards_n]]
                    moving_idx = backwards_n

                    for i in metric_patterns[backwards_n:]:
                        base_patterns = [
                            i for i in predicted[moving_idx - backwards_n : moving_idx]
                        ]
                        if len(base_patterns) > 1:
                            # Compute probability of n-gram using chain rule and use it to retrieve the best next pattern
                            if moving_idx - backwards_n == 0:
                                bi_grams_corpus = ["eos"] + base_patterns
                            elif moving_idx == repository_age_months:
                                bi_grams_corpus = base_patterns + ["eos"]
                            else:
                                bi_grams_corpus = base_patterns
                            sequence_bi_grams = generate_ngrams(bi_grams_corpus, 2)
                            probability = 1
                            for bi_gram in sequence_bi_grams:
                                probability *= bi_grams_probability_table.loc[
                                    bi_grams_probability_table["Token"] == bi_gram[0]
                                ][bi_gram[1]].to_numpy()[0]

                            # Look for highest probability
                            highest_probability = -1
                            best_next_pattern = ""
                            for pattern in PATTERNS_LABELS:
                                pattern_probability = (
                                    probability
                                    * bi_grams_probability_table.loc[
                                        bi_grams_probability_table["Token"]
                                        == base_patterns[-1]
                                    ][pattern].to_numpy()[0]
                                )
                                if pattern_probability > highest_probability:
                                    highest_probability = pattern_probability
                                    best_next_pattern = pattern
                        else:
                            # Only looking 1 month back, just look directly at the bigrams probability table
                            highest_probability = -1
                            best_next_pattern = ""
                            for pattern in PATTERNS_LABELS:
                                pattern_probability = bi_grams_probability_table.loc[
                                    bi_grams_probability_table["Token"]
                                    == base_patterns[-1]
                                ][pattern].to_numpy()[0]
                                if pattern_probability > highest_probability:
                                    highest_probability = pattern_probability
                                    best_next_pattern = pattern

                        predicted.append(best_next_pattern)
                        moving_idx += 1

                    # Count correct predictions
                    for idx, pred_values in enumerate(predicted[backwards_n:]):
                        total_predictions += 1
                        if pred_values == metric_patterns[idx]:
                            correct_predictions += 1
                        deviation_predictions += abs(
                            PATTERNS_MAP[pred_values]
                            - PATTERNS_MAP[metric_patterns[idx]]
                        )

            if total_predictions > 0:
                n_grams_accuracy[f"{n}_grams"] = {
                    "correct_predictions": correct_predictions,
                    "total_predictions": total_predictions,
                    "performance": correct_predictions / total_predictions,
                    "deviation": deviation_predictions / total_predictions,
                }

        n_grams_accuracy["baseline"] = baseline_scores
        log.info(n_grams_accuracy)
        with open(f"../data/{n_grams_accuracy_output_file}", "w") as outfile:
            json.dump(n_grams_accuracy, outfile, indent=4)

        best_accuracy = -1
        best_n = ""
        for n_key, n_items in n_grams_accuracy.items():
            if n_key != "baseline":
                if n_items["performance"] > best_accuracy:
                    best_accuracy = n_items["performance"]
                    best_n = n_key
        log.info(f"Best N is {best_n} with accuracy {round(best_accuracy, 3)}")

    log.info("Done!")
