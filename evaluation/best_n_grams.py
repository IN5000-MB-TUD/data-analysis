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

    log.info("Done!")
