#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Hengda Shi <hengda.shi@cs.ucla.edu>
#
# Distributed under terms of the MIT license.

"""

"""

import gc
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import RobustScaler


def preprocess(df, scale):
  print('Adjust Data Type...')
  df = adjust_type(df)

  print('Fill Missing Values...')
  df = fill_missing(df)

  #  print('Generate Features...')
  #  df = feature_generate(df)

  print('Transform Features...')
  df = feature_transform(df, scale)

  #  print('Reduce Dimension...')
  #  df = feature_reduce(df)
  #  df = adjust_type(df)
  return df


def adjust_type(df):
  for c in df.columns:
    min_val, max_val = df[c].min(), df[c].max()
    if df[c].dtype == 'float64':
      # note: this might loss precision!! might change this latter.
      #  if min_val>np.finfo(np.float16).min and max_val<np.finfo(np.float16).max:
      #    df[c] = df[c].astype(np.float16)
      if min_val>np.finfo(np.float32).min and max_val<np.finfo(np.float32).max:
        df[c] = df[c].astype(np.float32)
    elif df[c].dtype == 'int64' or df[c].dtype == 'int32':
      if min_val>np.iinfo(np.int8).min and max_val<np.iinfo(np.int8).max:
        df[c] = df[c].astype(np.int8)
      elif min_val>np.iinfo(np.int16).min and max_val<np.iinfo(np.int16).max:
        df[c] = df[c].astype(np.int16)

  return df

def fill_missing(df):
  #  # rough value range
  #  IQR = df[features].quantile(0.25) - df[features].quantile(0.75)
  #  # here we fill missing with negative outlier to denote it's truely missing
  #  filler = pd.Series(df[features].min()-0.1*(1.5*IQR), index=features)
  #  # filling data
  #  df[features] = df[features].fillna(filler)

  #  df = df.interpolate(method='index')
  df = df.fillna(method='ffill').fillna(0)

  return df

def feature_reduce(df):
  n_components = 24
  features = [c for c in df.columns if 'feature' in c and c != 'feature_0']
  pca = IncrementalPCA(n_components=n_components, batch_size=len(df)//4)
  pcs = pca.fit_transform(df[features])
  pcs_df = pd.DataFrame(pcs, columns=[f'feature_pc_{i}'
                                      for i in range(n_components)])
  print(pcs_df)
  df = df.drop(columns=features)
  df = df.join(pcs_df)
  return df

def feature_transform(df, scale):
  # scale input
  scaler = RobustScaler(copy=False)
  features = [c for c in df.columns if 'feature' in c and c != 'feature_0']
  if scale:
    n = 10
    for feature in [features[i*n:(i+1)*n] for i in range((len(features)+n-1)//n)]:
      df[feature] = scaler.fit_transform(df[feature].values)
  # feature_0 is the only categorical data
  df['feature_0'] = np.where(df['feature_0'].values == 1, 1, 0)
  return df

def feature_generate(df):
  df['feature_cross_41_42_43'] = df['feature_41'] + df['feature_42'] + df['feature_43']
  df['feature_cross_1_2'] = df['feature_1'] / (df['feature_2'] + 1e-5)
  return df
