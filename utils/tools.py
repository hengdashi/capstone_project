import gc

import numpy as np
import torch


def remove_overlap(inputs, c_out, pred_len):
  results = None
  for batch in inputs:
    if results is None:
      results = batch
      continue
    #  print(true_preds[:i].shape)
    #  print(true_preds[i:].shape)
    #  print(np.vstack((true_preds[i:],np.zeros(self.args.c_out))))
    results = np.vstack((results, np.zeros((1, c_out))))
    batch = np.vstack((np.zeros((results.shape[0]-batch.shape[0],
                                 c_out)), batch))
    results += batch

  n_data = results.shape[0]
  n_stack = results.shape[0] - results.shape[1] + 1
  for i in range(n_data):
    max_divisor = min(n_stack, pred_len)
    divisor = max_divisor if max_divisor <= i+1 <= n_data-max_divisor+1 \
                          else i + 1 if i+1 < max_divisor \
                                     else n_data - (i + 1) + 1
    results[i] /= divisor

  return results



def print_tensor_in_gc():
  count = 0
  for obj in gc.get_objects():
    try:
      if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        count += 1
        #  print(type(obj), obj.size())
    except:
      pass

  print(f'num of tensors: {count}')



def adjust_learning_rate(optimizer, epoch, args):
  # lr = args.learning_rate * (0.2 ** (epoch // 2))
  if args.lradj=='type1':
    lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
  elif args.lradj=='type2':
    lr_adjust = {
        2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
        10: 5e-7, 15: 1e-7, 20: 5e-8
    }
  if epoch in lr_adjust.keys():
    lr = lr_adjust[epoch]
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
    print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
  def __init__(self, patience=7, verbose=False, delta=0):
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf
    self.delta = delta

  def __call__(self, val_loss, model, path):
    score = -val_loss
    if self.best_score is None:
      self.best_score = score
      self.save_checkpoint(val_loss, model, path)
    elif score < self.best_score + self.delta:
      self.counter += 1
      print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_score = score
      self.save_checkpoint(val_loss, model, path)
      self.counter = 0

  def save_checkpoint(self, val_loss, model, path):
    if self.verbose:
      print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
    self.val_loss_min = val_loss

class dotdict(dict):
  """dot.notation access to dictionary attributes"""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

class StandardScaler():
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def transform(self, data):
    return (data - self.mean) / self.std

  def inverse_transform(self, data):
    return (data * self.std) + self.mean
