# data-analysis
IN5000 TU Delft - MSc Computer Science - Data Analysis

## Requirements

Run in the terminal:

```shell
# Scripts requirements
pip install -r requirements.txt

# Scripts + code formatting requirements
pip install -r requirements-ci.txt
```

## Environment Variables

The following environment variables must be set in order to run the scripts:

```shell
GITHUB_AUTH_TOKEN
MONGODB_HOST
MONGODB_PORT
MONGODB_DATABASE
MONGODB_USER
MONGODB_PASSWORD
MONGODB_QPARAMS
```

## Run Scripts

### Data Mining

The data mining uses the GitHub API to gather repositories data. The version used for this project is: "2022-11-28".

https://docs.github.com/en/rest/repos/repos?apiVersion=2022-11-28

Release

https://docs.github.com/en/rest/commits/commits?apiVersion=2022-11-28
https://docs.github.com/en/rest/releases/releases?apiVersion=2022-11-28
https://docs.github.com/en/rest/branches/branches?apiVersion=2022-11-28
https://docs.github.com/en/rest/projects/cards?apiVersion=2022-11-28

Code & Community

https://docs.github.com/en/rest/dependency-graph/sboms?apiVersion=2022-11-28
https://docs.github.com/en/rest/metrics/statistics?apiVersion=2022-11-28
https://docs.github.com/en/rest/metrics/traffic?apiVersion=2022-11-28

Move to the `data_mining` folder and run:

```shell
# Move to the data_mining folder
cd data_mining

# Run GitHub repositories mining
python repositories.py

# Run GitHub statistics gathering
# Might need to be re-run since the GitHub API needs time to gather the values
python statistics.py
```

## Code Formatting

To ensure high code quality, the Black formatted can be run as follows:

```shell
# Check formatting
black --check ./

# Format files
black ./
```
