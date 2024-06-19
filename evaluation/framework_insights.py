import json
import logging

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
    for cluster in CLUSTERS:
        cluster_repositories = repository_clusters[f"Cluster_{cluster}"]
        for repository_name in cluster_repositories:
            # Check for plateauing metrics
            for metric in METRICS:
                # Avoid unique patterns
                if repository_metrics_phases_count[repository_name][metric] <= 1:
                    continue
                if metric == "deployments" or metric == "workflows":
                    continue
                if (
                    repository_metrics_phases[repository_name][metric][0] == 1
                    and repository_metrics_phases[repository_name][metric][-1] == 2
                ):
                    if repository_name not in plateauing_projects:
                        plateauing_projects[repository_name] = 0
                    plateauing_projects[repository_name] += 1

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

    log.info("Done!")
