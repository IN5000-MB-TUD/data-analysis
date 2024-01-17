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

Move to the `data_mining` fodler and run:

```shell
# Move to the data_mining folder
cd data_mining

# Run GitHub repositories mining
python main.py
```

## Code Formatting

To ensure high code quality, the Black formatted can be run as follows:

```shell
# Check formatting
black --check ./

# Format files
black ./
```
