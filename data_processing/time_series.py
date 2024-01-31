import logging
import matplotlib.pyplot as plt

from connection import mo

# Setup logging
log = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def create_plot(path, title, subtitle, xlabel, ylabel, x, y, labels=None):
    """Create plot and save it as PNG"""
    # Adjusting the figure size
    plt.subplots(figsize=(16, 8))

    # Creating a plot
    for y_idx, y_data in enumerate(y):
        if labels:
            plt.plot(x, y_data, label=labels[y_idx])
        else:
            plt.plot(x, y_data)

    if labels:
        plt.legend(loc="upper left")

    # Adding a plot title and customizing its font size
    plt.suptitle(title, fontsize=20)
    plt.title(subtitle, fontsize=15)

    # Adding axis labels and customizing their font size
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)

    # Rotating axis ticks and customizing their font size
    plt.xticks(rotation=30, fontsize=15)

    # Saving the resulting plot to a file
    plt.savefig(path)


if __name__ == "__main__":
    log.info("Start GitHub statistics retrieval from Database")

    # Get the repositories in the database
    repositories = mo.db["repositories_data"].find()

    for idx, repository in enumerate(repositories):
        log.info("Analyzing repository {}".format(repository["full_name"]))

        # Process stargazers
        if (
            repository.get("statistics", {}).get("stargazers")
            and repository["stargazers_count"] > 0
        ):
            stargazers = [repository["created_at"]] + repository["statistics"][
                "stargazers"
            ]
            stargazers_cumulative = [0]
            stargazers_counter = 0
            for i in range(1, len(stargazers)):
                stargazers_counter += 1
                stargazers_cumulative.append(stargazers_counter)

            stargazers_cumulative.append(repository["stargazers_count"])
            stargazers.append(repository["updated_at"])

            create_plot(
                "../results/stargazers/{}_{}_{}.png".format(
                    idx, repository["owner"], repository["name"]
                ),
                "Stargazers {}".format(repository["full_name"]),
                "Total: {}".format(repository["stargazers_count"]),
                "Date",
                "Count",
                stargazers,
                [stargazers_cumulative],
            )

        # Process open issues
        if (
            repository.get("statistics", {}).get("issues")
            and repository["open_issues"] > 0
        ):
            issues_dates = [repository["created_at"]] + [
                issue["created_at"]
                for issue in repository["statistics"]["issues"].values()
            ]
            issues_dates.sort()
            issues_cumulative = [0]
            issues_counter = 0
            for i in range(1, len(issues_dates)):
                issues_counter += 1
                issues_cumulative.append(issues_counter)

            issues_cumulative.append(repository["open_issues"])
            issues_dates.append(repository["metadata"]["modified"])

            create_plot(
                "../results/issues/{}_{}_{}.png".format(
                    idx, repository["owner"], repository["name"]
                ),
                "Open Issues {}".format(repository["full_name"]),
                "Total: {}".format(repository["open_issues"]),
                "Date",
                "Count",
                issues_dates,
                [issues_cumulative],
            )

        # Process Weekly Commits Stats
        if repository.get("statistics", {}).get("commits_weekly"):
            commits_dates = [
                commit["timestamp"]
                for commit in repository["statistics"]["commits_weekly"].values()
            ]

            commits_cumulative = []
            additions_cumulative = []
            deletions_cumulative = []
            commits_counter = 0
            additions_counter = 0
            deletions_counter = 0
            for commit in repository["statistics"]["commits_weekly"].values():
                commits_counter += commit["total"]
                commits_cumulative.append(commits_counter)

                additions_counter += commit["additions"]
                additions_cumulative.append(additions_counter)

                deletions_counter += commit["deletions"]
                deletions_cumulative.append(deletions_counter)

            create_plot(
                "../results/commits/{}_{}_{}.png".format(
                    idx, repository["owner"], repository["name"]
                ),
                "Additions and Deletions {}".format(repository["full_name"]),
                "Total Commits: {}".format(repository["commits"]),
                "Date",
                "Count",
                commits_dates,
                [commits_cumulative, additions_cumulative, deletions_cumulative],
                ["Total", "Additions", "Deletions"],
            )

        log.info("------------------------")
