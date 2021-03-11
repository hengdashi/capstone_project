# Capstone Project

This is the repo for the capstone project `Transformer in Time Series Data`.
The implementation is modified based on [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://github.com/zhouhaoyi/Informer2020).

## Setup

```
conda create -y -n ts_transformer python=3.7.5
conda activate ts_transformer
pip install -r requirements.txt
```

## Data

The data can be downloaded from [here](https://www.kaggle.com/c/jane-street-market-prediction/data).

If `kaggle` module in installed locally, data can also be downloaded using `kaggle competitions download -c jane-street-market-prediction`.

The data should be unzipped and place into `data/jane-street-market-prediction`.

## Run the experiment

`run.sh` contains the hyperparameters used to generate the best result.
Other hyperparameters can also be experimented by tuning the hyperparameters.

```
./run.sh
```
