# Capstone Project

This is the repo for the capstone project `Transformer in Time Series Data`.
The implementation is modified based on [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://github.com/zhouhaoyi/Informer2020)

## Setup

```
conda create -y -n ts_transformer python=3.7.5
pip install -r requirements.txt
```

## Run the experiment

`run.sh` contains the hyperparameters used to generate the best result.
Other hyperparameters can also be experimented by tuning the hyperparameters.

```
./run.sh
```
