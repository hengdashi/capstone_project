import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=5000):
    super(PositionalEmbedding, self).__init__()
    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
  def __init__(self, c_in, channel, d_model, dropout=0.1, data='ETT'):
    super(TokenEmbedding, self).__init__()
    self.data = data
    if not data.startswith('janestreet'):
      padding = 1 if torch.__version__>='1.5.0' else 2
      self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                  kernel_size=3, padding=padding, padding_mode='circular')
      for m in self.modules():
        if isinstance(m, nn.Conv1d):
          nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
    else:
      self.dense1 = nn.Linear(c_in, d_model)
      self.bn1 = nn.BatchNorm1d(d_model)
      self.dense2 = nn.Linear(d_model, d_model)
      self.bn2 = nn.BatchNorm1d(d_model)
      self.act = nn.SiLU()


  def forward(self, x):
    if not self.data.startswith('janestreet'):
      x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
    else:
      x = self.dense1(x)
      x = x.permute(0, 2, 1)
      x = self.bn1(x)
      x = x.permute(0, 2, 1)
      x = self.dense2(x)
      x = x.permute(0, 2, 1)
      x = self.bn2(x)
      x = x.permute(0, 2, 1)
      x = self.act(x)
    return x

class FixedEmbedding(nn.Module):
  def __init__(self, c_in, d_model):
    super(FixedEmbedding, self).__init__()

    w = torch.zeros(c_in, d_model).float()
    # 24 * 512
    w.require_grad = False

    position = torch.arange(0, c_in).float().unsqueeze(1)
    # [1, 24]
    # [[0], [1], [2], [3], ..., [23]]
    div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

    w[:, 0::2] = torch.sin(position * div_term)
    w[:, 1::2] = torch.cos(position * div_term)

    self.emb = nn.Embedding(c_in, d_model)
    self.emb.weight = nn.Parameter(w, requires_grad=False)

  def forward(self, x):
    return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
  def __init__(self, d_model, embed_type='fixed', freq='h', data='ETT'):
    super(TemporalEmbedding, self).__init__()

    self.data = data

    # 15 mins -> 60/15 = 4
    minute_size = 4; hour_size = 24
    weekday_size = 7; day_size = 32; month_size = 13

    Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
    if freq == 't':
      self.minute_embed = Embed(minute_size, d_model)
    if not data.startswith('janestreet'):
      self.hour_embed = Embed(hour_size, d_model)
      self.weekday_embed = Embed(weekday_size, d_model)
      self.month_embed = Embed(month_size, d_model)

    self.day_embed = Embed(500, d_model)

  def forward(self, x):
    x = x.long()

    if not self.data.startswith('janestreet'):
      minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
      hour_x = self.hour_embed(x[:,:,3])
      weekday_x = self.weekday_embed(x[:,:,2])
      day_x = self.day_embed(x[:,:,1])
      month_x = self.month_embed(x[:,:,0])
      return hour_x + weekday_x + day_x + month_x + minute_x
    else:
      return self.day_embed(x)

class TimeFeatureEmbedding(nn.Module):
  def __init__(self, d_model, embed_type='timeF', freq='h'):
    super(TimeFeatureEmbedding, self).__init__()

    freq_map = {'h':4, 't':5, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
    d_inp = freq_map[freq]
    self.embed = nn.Linear(d_inp, d_model)

  def forward(self, x):
    return self.embed(x)

class DataEmbedding(nn.Module):
  def __init__(self, c_in, channel, d_model, embed_type='fixed', freq='h', dropout=0.1, data='ETT'):
    super(DataEmbedding, self).__init__()

    self.data = data

    self.value_embedding = TokenEmbedding(c_in=c_in, channel=channel, d_model=d_model, dropout=dropout, data=data)
    self.position_embedding = PositionalEmbedding(d_model=d_model)
    if not self.data.startswith('janestreet'):
      self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq, data=data) \
                                if embed_type!='timeF' \
                                else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x, x_mark):
    if not self.data.startswith('janestreet'):
      x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
    else:
      x = self.value_embedding(x) + self.position_embedding(x)
    return self.dropout(x)
    #  return x
