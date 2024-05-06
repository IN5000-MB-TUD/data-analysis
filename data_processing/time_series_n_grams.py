import json
import logging
from pathlib import Path

import pandas as pd

from connection import mo

# Setup logging
log = logging.getLogger(__name__)


# Declare constants
PATTERNS_LABELS = ["Steep", "Shallow", "Plateau"]


# Helper functions
def generate_ngrams(corpus_tokens, k):
    """Generate n-grams for the given tokens"""
    n_grams = []
    i = 0
    while i < len(corpus_tokens):
        n_grams.append(corpus_tokens[i : i + k])
        i = i + 1
    return n_grams[:-1]


def generate_ngram_freq(n_gram):
    """Generate n-gram frequencies"""
    ngram_freq = {}
    for i in n_gram:
        st = " ".join(i)
        if st not in ngram_freq:
            ngram_freq[st] = 0
        ngram_freq[st] += 1
    return ngram_freq


def generate_probability_table(
    distinct_tokens_list, distinct_tokens_frequency, ngram_freq
):
    """Generate n-grams probability table"""
    n_tokens = len(distinct_tokens_list)

    probability_table_rows = []
    for i in range(n_tokens):
        row_dict = {
            "Token": distinct_tokens_list[i],
        }
        denominator = distinct_tokens_frequency[distinct_tokens_list[i]]
        for j in range(n_tokens):
            sequence = distinct_tokens_list[i] + " " + distinct_tokens_list[j]
            numerator = ngram_freq[sequence] if sequence in ngram_freq else 0

            row_dict[distinct_tokens_list[j]] = round(numerator / denominator, 3)

        probability_table_rows.append(row_dict)

    probability_table = pd.DataFrame(probability_table_rows)
    return probability_table


# Start execution
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
    with open("../data/repository_metrics_phases_count.json") as json_file:
        repository_metrics_phases_count = json.load(json_file)

    corpus = ""
    max_segments = -1
    for repository_name, metrics_phases in repository_metrics_phases.items():
        for metric, phases in metrics_phases.items():
            max_segments = max(
                max_segments, repository_metrics_phases_count[repository_name][metric]
            )
            patterns_sequence = [PATTERNS_LABELS[x] for x in phases]
            corpus += " ".join(patterns_sequence) + ". "

    log.info(f"Max segments: {max_segments}")

    # Preprocess corpus
    corpus = corpus.lower()
    corpus = "eos " + corpus
    corpus = corpus.replace(".", " eos")
    corpus = corpus[:-1]

    # Generate tokens
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
        distinct_tokens, tokens_frequency, bi_grams_frequency
    )

    # Save csv file
    bi_grams_probability_table.to_csv(
        "../data/bi_grams_probability_table.csv", index=False
    )

    log.info(f"Probability Table: \n")
    log.info(bi_grams_probability_table)

    sequences = [
        # Steep first
        [0, 0, 0],
        [0, 1, 0],
        [0, 2, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 2, 1],
        [0, 0, 2],
        [0, 1, 2],
        [0, 2, 2],
        # Shallow first
        [1, 0, 0],
        [1, 1, 0],
        [1, 2, 0],
        [1, 0, 1],
        [1, 1, 1],
        [1, 2, 1],
        [1, 0, 2],
        [1, 1, 2],
        [1, 2, 2],
        # Plateau first
        [2, 0, 0],
        [2, 1, 0],
        [2, 2, 0],
        [2, 0, 1],
        [2, 1, 1],
        [2, 2, 1],
        [2, 0, 2],
        [2, 1, 2],
        [2, 2, 2],
    ]
    sequences_probability_rows = []
    for sequence in sequences:
        sequence_tokens = [PATTERNS_LABELS[i].lower() for i in sequence]
        sequence_bi_grams = generate_ngrams(["eos"] + sequence_tokens, 2)
        probability = 1
        for bi_gram in sequence_bi_grams:
            probability *= bi_grams_probability_table.loc[
                bi_grams_probability_table["Token"] == bi_gram[0]
            ][bi_gram[1]].to_numpy()[0]
        sequences_probability_rows.append(
            {
                "sequence": " - ".join([PATTERNS_LABELS[i] for i in sequence]),
                "probability": round(probability, 4),
            }
        )

    sequences_probability = pd.DataFrame(sequences_probability_rows)
    log.info(f"Sequences Probability Table: \n")
    log.info(sequences_probability)

    # Save csv file
    sequences_probability.to_csv(
        "../data/time_series_phases_probabilities.csv", index=False
    )

    log.info("Done!")
