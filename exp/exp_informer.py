import os
import gc
import time
import warnings


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
#  from torch.cuda.amp import autocast, GradScaler


from sklearn.metrics import roc_auc_score


# import data from data_loader
from data.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_JaneStreet
)
# import basic
from exp.exp_basic import Exp_Basic
# import model
from models.model import Informer, InformerStack

# tools from utils
from utils.tools import EarlyStopping, adjust_learning_rate, remove_overlap
from utils.metrics import metric, utility_score_bincount

# loss
from optimizer.loss import SmoothBCEwLogits


warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
  def __init__(self, args):
    super(Exp_Informer, self).__init__(args)

  def _build_model(self):
    model_dict = {
      'informer':Informer,
    }
    if self.args.model in ['informer', 'informerstack']:
        model = model_dict[self.args.model](
            # encoder input size
            self.args.enc_in,
            # decoder input size
            self.args.dec_in, 
            # output size
            self.args.c_out, 
            # input series length
            self.args.seq_len, 
            # help series length
            self.args.label_len,
            # predict series length
            self.args.pred_len, 
            # prob sparse factor
            self.args.factor,
            # dimension of model
            self.args.d_model,
            # number of multiheads
            self.args.n_heads, 
            # number of encoder layer
            self.args.e_layers,
            # number of decoder layer
            self.args.d_layers,
            # dimension of fcn
            self.args.d_ff,
            # dropout rate
            self.args.dropout, 
            # attention type
            self.args.attn,
            # embedding type
            self.args.embed,
            # dataset
            self.args.data,
            # frequency
            self.args.freq,
            # activation layer type
            self.args.activation,
            # output attention
            self.args.output_attention,
            # distil
            self.args.distil,
            # device
            self.device
        )

    return model.float()

  def _get_data(self, flag):
    args = self.args

    data_dict = {
        'ETTh1':Dataset_ETT_hour,
        'ETTh2':Dataset_ETT_hour,
        'ETTm1':Dataset_ETT_minute,
        'janestreet': Dataset_JaneStreet
    }
    Data = data_dict[self.args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if not flag == 'train':
      shuffle_flag = False; drop_last = True; batch_size = args.batch_size
    else:
      shuffle_flag = True; drop_last = True; batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        data_name=args.data_name,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq
    )

    print(f'length of {flag} dataset: ', len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    return data_set, data_loader

  def _select_optimizer(self):
    model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
    return model_optim

  def _select_criterion(self, data='ETT'):
    if not data.startswith('janestreet'):
      print('using MSE')
      criterion = nn.MSELoss()
    else:
      print('using BCE')
      #  criterion = SmoothBCEwLogits(smoothing=0)
      #  criterion = nn.BCEWithLogitsLoss()
      #  criterion = nn.MSELoss()
      criterion = nn.SmoothL1Loss()
    return criterion

  def evaluate(self, valid_data, valid_loader, criterion):
    self.model.eval()
    valid_loss = []
    valid_auc = []
    y_preds = []
    y_trues = []

    iter_count = 0
    start = None
    end = None
    for i, (s_begin, s_end, r_begin, r_end, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
    #  for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):

      if i == 0:
        start = s_end[0]

      #  print(f'{i} s_begin: ', s_begin)
           #  print(f'{i} s_end: ', s_end)
      #  print(f'{i} r_begin: ', r_begin)
      #  print(f'{i} r_end: ',r_end)

      iter_count += 1

      batch_x = batch_x.float().to(self.device)
      batch_y = batch_y.float()

      batch_x_mark = batch_x_mark.float().to(self.device)
      batch_y_mark = batch_y_mark.float().to(self.device)

      # decoder input
      dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
      dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
      # encoder - decoder
      if self.args.output_attention:
        y_pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
      else:
        y_pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
      f_dim = -1 if self.args.features=='MS' else 0

      y_true = batch_y[:,-self.args.pred_len:,-self.args.c_out:].float().to(self.device)
      #  y_true = batch_y[:,-self.args.pred_len:,f_dim:].float().to(self.device)

      #  print(f'y_pred: ', y_pred)
      #  print(f'y_true: ', y_true)
      #  print(f's_end: ', s_end)
      #  print(f'r_end: ', r_end)

      loss = criterion(y_pred, y_true)

      loss = loss.item()
      valid_loss.append(loss)
      #  print(y_pred.shape)
      y_preds.append(y_pred.sigmoid().detach().cpu().numpy())
      y_trues.append(y_true.sigmoid().detach().cpu().numpy())
      #  y_trues.append(y_true.detach().cpu().numpy().astype(int))

      y_pred = np.where(y_preds[-1] >= 0.5, 1, 0).astype(int)
      y_true = np.where(y_trues[-1] >= 0.5, 1, 0).astype(int)
      #  y_true = y_trues[-1].astype(int)

      #  print(y_true)
      #  print(y_pred)
      valid_auc.append(roc_auc_score(np.median(y_true, axis=1), np.median(y_pred, axis=1)))

      if (i+1) % 100 == 0:
        print(f"\tvalid_iters={i+1} | loss={loss:.4f} | running_loss={np.mean(valid_loss):.4f} | running_auc={np.mean(valid_auc):.4f}")

      del batch_x, batch_x_mark, batch_y_mark, dec_inp, y_true, y_pred

      torch.cuda.empty_cache()

    end = r_end[-1]
    #  print(f'{i} r_end: ',r_end)

    valid_loss = np.mean(valid_loss)

    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)

    #  print(y_preds[0])

    y_final_preds = None
    y_final_trues = None
    if self.args.pred_len != 1:
      y_final_preds = remove_overlap(y_preds, self.args.c_out, self.args.pred_len)
      y_final_trues = remove_overlap(y_trues, self.args.c_out, self.args.pred_len)
    else:
      y_final_preds = y_preds.reshape(-1, self.args.c_out)
      y_final_trues = y_trues.reshape(-1, self.args.c_out)

      #  print(y_final_preds.shape)
      #  print(y_final_preds.min(), y_final_preds.max())

    #  print(y_final_preds.shape)

    assert (end - start == len(y_final_preds))
    assert (end - start == len(y_final_trues))

    self.model.train()
    return valid_loss, y_final_preds, y_final_trues, start, end


  def train(self, setting):
    train_data, train_loader = self._get_data(flag='train')
    valid_data, valid_loader = self._get_data(flag='val')

    print(f'number of batches in train data={len(train_loader)}')
    print(f'number of batches in valid data={len(valid_loader)}')

    path = './checkpoints/'+setting
    if not os.path.exists(path):
      os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    model_optim = self._select_optimizer()
    criterion =  self._select_criterion(self.args.data)

    #  print(self.model)
    best_utility = 0
    for epoch in range(self.args.train_epochs):
      iter_count = 0
      train_loss = []
      train_auc  = []

      self.model.train()

      # batch_x: (batch_size, seq_len, n_features)
      # batch_y: (batch_size, label_len + pred_len, n_features)
      # batch_x_mark: (batch_size, seq_len)
      # batch_y_mark: (batch_size, label_len + pred_len)


      for i, (s_begin, s_end, r_begin, r_end, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
      #  for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

        #  print(f'{i} s_begin: ', s_begin)
        #  print(f'{i} s_end: ', s_end)
        #  print(f'{i} r_begin: ', r_begin)
        #  print(f'{i} r_end: ',r_end)

        iter_count += 1

        #  print(f'x : {batch_x}')
        #  print(f'y : {batch_y}')
        #  print(f'x_mark : {batch_x_mark}')
        #  print(f'y_mark : {batch_y_mark}')

        model_optim.zero_grad()

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.output_attention:
          y_pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
          y_pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features=='MS' else 0

        y_true = batch_y[:,-self.args.pred_len:,-self.args.c_out:].to(self.device)
        #  y_true = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        loss = criterion(y_pred, y_true)

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        model_optim.step()

        #  print(y_pred)
        y_pred = np.where(y_pred.sigmoid().detach().cpu().numpy() >= 0.5, 1, 0).astype(int)
        y_true = np.where(y_true.sigmoid().detach().cpu().numpy() >= 0.5, 1, 0).astype(int)
        #  y_true = y_true.detach().cpu().numpy().astype(int)
        #  print('y_true: ', np.median(y_true, axis=1))
        #  print('y_pred: ', np.median(y_pred, axis=1))

        train_auc.append(roc_auc_score(np.median(y_true, axis=1), np.median(y_pred, axis=1)))

        loss = loss.item()
        train_loss.append(loss)


        if (i+1) % 100 == 0:
          print(f'\ttrain_iters={i+1} | epoch={epoch+1} | ' \
                f'batch_loss={loss:.4f} | running_loss={np.mean(train_loss):.4f} | running_auc={np.mean(train_auc):.4f}')
          speed = (time.time()-time_now)/iter_count
          left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
          print(f'\tspeed={speed:.4f}s/batch | left_time={left_time:.4f}s')
          #  print(torch.cuda.memory_summary(abbreviated=True))

          iter_count = 0
          time_now = time.time()

        del batch_x, batch_x_mark, batch_y, batch_y_mark, dec_inp, y_true, y_pred


      gc.collect()

      torch.cuda.empty_cache()
      torch.cuda.reset_max_memory_allocated(self.device)

      returns = self.evaluate(valid_data, valid_loader, criterion)
      valid_loss, valid_preds, valid_trues, v_start, v_end = returns

      #  print(valid_data.data_x[b_start:b_end, -self.args.c_out:].shape)
      #  print('before where: ', valid_preds)
      #  print(valid_data.data_x[b_start:b_end, -self.args.c_out:])
      #  valid_trues = valid_data.data_x[b_start:b_end, -self.args.c_out:]
      #  print('valid_preds.shape: ', valid_preds.shape)
      valid_preds = np.median(valid_preds, axis=1)
      print(pd.DataFrame(valid_preds).describe())
      valid_preds = np.where(valid_preds >= 0.5, 1, 0).astype(int)

      #  print('after where: ', valid_preds)
      #  valid_trues = 1/(1+np.exp(-valid_trues))
      valid_trues = np.median(valid_trues, axis=1)
      valid_trues = np.where(valid_trues >= 0.5, 1, 0).astype(int)
      #  print('valid_trues shape: ', valid_trues.shape)
      #  print('valid_trues: ', valid_trues)
      valid_auc = roc_auc_score(valid_trues, valid_preds)
      valid_u_score = utility_score_bincount(date=valid_data.data_stamp[v_start:v_end],
                                              weight=valid_data.weight[v_start:v_end],
                                              resp=valid_data.resp[v_start:v_end],
                                              action=valid_preds)
      max_u_score = utility_score_bincount(date=valid_data.data_stamp[v_start:v_end],
                                            weight=valid_data.weight[v_start:v_end],
                                            resp=valid_data.resp[v_start:v_end],
                                            action=valid_trues)

      best_utility = max(best_utility, valid_u_score)

      print(f'epoch={epoch+1} | ' \
            f'average_train_loss={np.mean(train_loss):.4f} | average_valid_loss={valid_loss:.4f} | '
            f'valid_utility={valid_u_score:.4f}/{max_u_score:.4f} | valid_auc={valid_auc:.4f}')


      early_stopping(valid_auc, self.model, path)
      if early_stopping.early_stop:
        print("Early stopping")
        print(f"Best utility score is {best_utility:.4f}")
        break

      adjust_learning_rate(model_optim, epoch+1, self.args)

    best_model_path = path+'/'+'checkpoint.pth'
    self.model.load_state_dict(torch.load(best_model_path))

    return self.model


  def test(self, setting):
    test_data, test_loader = self._get_data(flag='test')

    self.model.eval()

    preds = []
    trues = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
      batch_x = batch_x.float().to(self.device)
      #  batch_y = batch_y.double()
      batch_x_mark = batch_x_mark.float().to(self.device)
      batch_y_mark = batch_y_mark.float().to(self.device)

      # decoder input
      dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:])
      dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
      # encoder - decoder
      if self.args.output_attention:
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
      else:
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

      f_dim = -1 if self.args.features=='MS' else 0

      batch_y = batch_y[:,-self.args.pred_len:,f_dim:].float().to(self.device)

      pred = outputs.detach().cpu().numpy()#.squeeze()
      true = batch_y.detach().cpu().numpy()#.squeeze()

      preds.append(pred)
      trues.append(true)

    preds = np.array(preds)
    trues = np.array(trues)
    #  print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #  print('test shape:', preds.shape, trues.shape)

    # result save
    folder_path = './results/' + setting +'/'
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))

    np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path+'pred.npy', preds)
    np.save(folder_path+'true.npy', trues)

    return
