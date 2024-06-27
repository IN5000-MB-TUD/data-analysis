# A Framework for Identifying Evolution Patterns of Open-Source Software Projects

IN5000 TU Delft - MSc Computer Science

This repository contains the scripts to reproduce and expand the work for the MSc Thesis in Computer Science related to developing a framework for identifying the evolution patterns of open-source software projects. 

## Paper

The MSc Thesis paper is available [here](paper/A_Framework_for_Identifying_Evolution_Patterns_of_Open-Source_Software_Projects.pdf).

## Requirements

- Python >= 3.10
- MongoDB (can be changed if needed but the scripts must be adapted)

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

#### Evolution Patterns Modeling

```shell
# Move to the data_processing folder
cd data_processing

# Create the patterns clustering model (saved in the models/phases folder)
python time_series_phases.py
```

#### Multivariate Time Series Clustering

```shell
# Move to the data_processing folder
cd data_processing

# Cluster repositories based on their metrics phases and metrics patterns similarity
# The clustering model is saved in the models/clustering folder
python time_series_clustering.py
```

#### Multivariate Time Series Forecasting

```shell
# Move to the data_processing folder
cd data_processing

# Create the forecasting models for each metric
# The forecasting models are saved in the models/forecasting folder
# One subfolder will be present for each cluster
python time_series_forecasting.py
```

### Framework Evaluation

Move to the `evaluation` folder and run the following scripts:

```shell
# Move to the evaluation folder
cd evaluation

# Framework insights - Patterns Modeling
python patterns_modeling.py

# Framework insights - Clustering
python multivariate_clustering.py

# Framework insights - Forecasting
python patterns_forecasting_models.py

# Framework insights - Features importance
python forecasting_features_importance_ablation.py

# N-grams 
python months_n_grams.py
```

### Framework Pipeline

Move to the `data_pipeline` folder and run the following scripts:

```shell
# Move to the data_pipeline folder
cd data_pipeline

# Run the pipeline
python repository_pipeline.py
````

## Code Formatting

To ensure high code quality, the Black formatted can be run as follows:

```shell
# Check formatting
black --check ./

# Format files
black ./
```

## SonarCloud Scripts

To reproduce and/or process more data using [SonarCloud](https://www.sonarsource.com/products/sonarcloud/), follow these steps:

1. Create a SonarCloud free account
2. Create a project and get the related `SONAR_TOKEN`
3. Create the project folder in the `sonar_scanner` folder
4. Create the `pull_releases.sh` and `sonar.sh` scripts following the existing examples
5. Run in the terminal:

```shell
# Move to the project folder
cd project_folder

# Pull repository code releases from GitHub
./pull_releases.sh

# Process code with SonarCloud
./sonar.sh
```

6. Retireve the metrics results from the [SonarCloud API](https://sonarcloud.io/web_api) and store them in JSON files following the existing examples
7. Run in the terminal

```shell
# Process the metrics time series to obtain the break points and patterns sequence
python process_nar_data.py
```

## Models and Datasets

The models are available in the `model` folder in this repository as well as at the following [HuggingFace Collection](https://huggingface.co/collections/MattiaBonfanti-CS/in5000-mb-tud-65e8756337418d7dbd383f66)

The datasets are available at the same collection and can be placed in the `data` folder. Otherwise, they can be created by running the scripts.

## User Interface

A simple UI to visualize the time series data is available in the following repository: https://github.com/IN5000-MB-TUD/data-app

## Contributors

Project developed for the course IN5000 - Master's thesis of the 2023/2024 academic year at TU Delft.

Author:
- Mattia Bonfanti
- [m.bonfanti@student.tudelft.nl](mailto:m.bonfanti@student.tudelft.nl)
- Master's in Computer Science - Software Technology Track
