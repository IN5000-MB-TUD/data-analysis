# SonarCloud Scanner

This folder contains the scripts used to gather additional metrics for the following projects through SonarCloud:

- cockroachdb/cockroach
- patternfly/patternfly-react
- pypa/pip

To successfully add projects and evaluate them in SonarCloud, the following steps are required:

1. Create a new project in SonarCloud
2. Create a project folder in the current directory
3. Create the `pull_releases.sh` and `sonar.sh` scripts following the existing ones in the current folder and adjust the required variables for the specific project
4. Add the tags that you want to analyze in the `pull_releases.sh` and run it to pull the code from GitHub
5. Run the `sonar.sh` script to execute Sonar evaluations on the pulled tags code and report the results in SonarCloud
6. Extract the metrics from SonarCloud

