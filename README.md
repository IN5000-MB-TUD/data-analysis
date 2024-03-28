# Repositories Data Analysis

IN5000 TU Delft - MSc Computer Science - Data Analysis

## Requirements

Python Version must be >= 3.10.

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

Move to the `data_mining` folder and run:

```shell
# Move to the data_mining folder
cd data_mining

# Run GitHub repositories mining
python repositories.py

# Run GitHub statistics gathering
python statistics.py

# In case of missing statistics, fill the gaps with zeros
python statistics_fill_gaps.py
```

### Data Processing

Move to the `data_processing` folder and run the following scripts:

#### Phases Clustering

```shell
# Move to the data_processing folder
cd data_processing

# Create the phases clustering model (saved in the models/phases folder)
python time_series_phases.py

# Gather the occurrence probabilities of each phase (stored in the database)
python time_series_phase_probability.py
```

#### Repositories Clustering

```shell
# Move to the data_processing folder
cd data_processing

# Cluster repositories based on their metrics phases and metrics patterns similarity
# The clustering model is saved in the models/clustering folder
python time_series_clustering.py

# Plot the average metrics curves for each cluster
python time_series_clustering_plot.py
```

#### Multi-Variate Time Series Forecasting

```shell
# Move to the data_processing folder
cd data_processing

# Create the forecasting models for each metric
# The forecasting models are saved in the models/forecasting folder
python time_series_forecasting.py
```

#### Plot Curves

```shell
# Move to the data_processing folder
cd data_processing

# Plot the repositories metrics time series curves
python time_series_plot.py
```

## Code Formatting

To ensure high code quality, the Black formatted can be run as follows:

```shell
# Check formatting
black --check ./

# Format files
black ./
```

## Contributors

Project developed for the course IN5000 - Master's thesis of the 2023/2024 academic year at TU Delft.

Author:
- Mattia Bonfanti
- [m.bonfanti@student.tudelft.nl](mailto:m.bonfanti@student.tudelft.nl)
- Master's in Computer Science - Software Technology Track
