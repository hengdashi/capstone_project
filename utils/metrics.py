import numpy as np
import pandas as pd

def RSE(pred, true):
  return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
  u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
  d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
  return (u/d).mean(-1)

def MAE(pred, true):
  return np.mean(np.abs(pred-true))

def MSE(pred, true):
  return np.mean((pred-true)**2)

def RMSE(pred, true):
  return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
  return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
  return np.mean(np.square((pred - true) / true))

def metric(pred, true):
  mae = MAE(pred, true)
  mse = MSE(pred, true)
  rmse = RMSE(pred, true)
  mape = MAPE(pred, true)
  mspe = MSPE(pred, true)

  return mae,mse,rmse,mape,mspe

def utility_score_bincount(date, weight, resp, action):
  #  print('date', date.shape)
  #  print('weight: ', weight.shape)
  #  print('resps: ', resp.shape)
  #  print('action: ', action.shape)

  #  print(np.unique(date))
  print(pd.DataFrame(action).describe())

  count_i = len(np.unique(date))

  Pi = np.bincount(date, weight * resp * action)
  #  print('Pi: ', Pi.shape)
  #  print(Pi)
  print('action length: ', action.shape)
  print('sell action: ', action[action == 0].shape)
  print('buy action: ', action[action == 1].shape)
  summed = np.sum(Pi)
  squared = np.sqrt(np.sum(Pi ** 2))
  t = np.divide(summed, squared, out=np.zeros_like(summed), where=(squared!=0)) * np.sqrt(250 / count_i)
  print('t: ', t)
  u = np.clip(t, 0, 6) * summed
  print('u: ', u)
  return u
