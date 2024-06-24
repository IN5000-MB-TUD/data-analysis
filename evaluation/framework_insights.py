import json
import logging

from ruptures.metrics import precision_recall

from connection import mo


# Setup logging
log = logging.getLogger(__name__)

CLUSTERS = [0, 1, 2]
PATTERNS = [0, 1, 2]
PATTERNS_LABELS = ["Steep", "Shallow", "Plateau"]
METRICS = [
    "stargazers",
    "issues",
    "commits",
    "contributors",
    "deployments",
    "forks",
    "pull_requests",
    "workflows",
    "releases",
    "size",
]
METRICS_BLACKLIST = {
    "deployments",
    "workflows",
    "releases",
}


if __name__ == "__main__":
    log.info("Find plateauing projects")

    # Open files
    with open("../data/repository_clusters.json") as json_file:
        repository_clusters = json.load(json_file)
    with open("../data/repository_metrics_phases.json") as json_file:
        repository_metrics_phases = json.load(json_file)
    with open("../data/repository_metrics_phases_count.json") as json_file:
        repository_metrics_phases_count = json.load(json_file)
    with open("../data/repository_metrics_phases_idxs.json") as json_file:
        repository_metrics_phases_idxs = json.load(json_file)

    # Find plateauing projects
    plateauing_projects = {}
    plateauing_metrics = {}
    for cluster in CLUSTERS:
        cluster_repositories = repository_clusters[f"Cluster_{cluster}"]
        for repository_name in cluster_repositories:
            # Check for plateauing metrics
            for metric in METRICS:
                # Avoid unique patterns
                if repository_metrics_phases_count[repository_name][metric] <= 1:
                    continue
                if metric in METRICS_BLACKLIST:
                    continue
                if (
                    repository_metrics_phases[repository_name][metric][0] == 1
                    and repository_metrics_phases[repository_name][metric][-1] == 2
                ):
                    if repository_name not in plateauing_projects:
                        plateauing_projects[repository_name] = 0
                        plateauing_metrics[repository_name] = []
                    plateauing_projects[repository_name] += 1
                    plateauing_metrics[repository_name].append(metric)

    # Get projects with most plateauing metrics
    plateauing_projects_names = sorted(
        plateauing_projects, key=plateauing_projects.get, reverse=True
    )
    log.info(
        [
            (n, plateauing_projects[n])
            for n in plateauing_projects_names
            if plateauing_projects[n] > 1
        ]
    )

    # Find non-plateauing similar projects
    repository_full_name = "SPANDigital/presidium"
    repository_patterns_idxs = repository_metrics_phases_idxs[repository_full_name]
    repository_patterns = repository_metrics_phases[repository_full_name]
    repository_cluster = 0

    log.info(repository_full_name)
    log.info(repository_patterns_idxs)
    log.info(repository_patterns)
    log.info(plateauing_metrics[repository_full_name])

    cluster_repositories = repository_clusters[f"Cluster_{repository_cluster}"]
    similar_repositories = {}
    for repository_name in cluster_repositories:
        if repository_name == repository_full_name:
            continue

        similar_repositories[repository_name] = 0

        # Check for plateauing metrics
        for metric in plateauing_metrics[repository_full_name]:
            if (
                repository_metrics_phases[repository_name][metric][0] == 1
                and repository_metrics_phases[repository_name][metric][-1] != 2
            ):
                max_length = max(
                    repository_patterns_idxs[metric][-1],
                    repository_metrics_phases_idxs[repository_name][metric][-1],
                )
                repository_patterns_idxs[metric][-1] = max_length
                repository_metrics_phases_idxs[repository_name][metric][-1] = max_length
                _, recall = precision_recall(
                    repository_patterns_idxs[metric],
                    repository_metrics_phases_idxs[repository_name][metric],
                )
                similar_repositories[repository_name] += recall

        similar_repositories[repository_name] /= len(
            plateauing_metrics[repository_full_name]
        )

    similar_projects_names = sorted(
        similar_repositories, key=similar_repositories.get, reverse=True
    )
    most_similar_repository_name = similar_projects_names[0]
    most_similar_repository_patterns_idxs = repository_metrics_phases_idxs[
        most_similar_repository_name
    ]
    most_similar_repository_patterns = repository_metrics_phases[
        most_similar_repository_name
    ]

    log.info(most_similar_repository_name)
    log.info(most_similar_repository_patterns_idxs)
    log.info(most_similar_repository_patterns)

    log.info("Done!")
