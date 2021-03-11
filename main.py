import os
import argparse

import torch

# import exp informer
from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=True, default='janestreet', help='data')
parser.add_argument('--root_path', type=str, default='./data/jane-street-market-prediction/', help='root path of the data file')
parser.add_argument('--data_name', type=str, default='train.csv', help='data file name')    
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS(TBD)]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')

parser.add_argument('--seq_len', type=int, default=4*24, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=2*24, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=1*24, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')


if __name__ == "__main__":
  args = parser.parse_args()

  data_parser = {
      #  'janestreet': {'data': 'train.csv','T':'action', 'M':[135,135,5],'S':[1,1,5]}
      #  'janestreet': {'data': 'train.csv','T':'action', 'M':[134,134,5],'S':[1,1,5]}
    'janestreet': {'data': 'train.csv','T':'action', 'M':[131,131,1],'S':[1,1,5]}
  }
  if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_name = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

  Exp = Exp_Informer

  # model: informer
  # data: janestreet
  # features: M
  # seq_len (input series length): 96
  # label_len (help series length): 48
  # pred_len (predict series length): 24
  # d_model (dimension of model): 512
  # n_heads (num of heads): 8
  # e_layers (num of encode layers): 3
  # d_layers (num of decode layers): 2
  # d_ff (dimension of fcn): 1024
  # attn (attention): prob
  # embed (embedding type): fixed
  # des (exp description): test
  for ii in range(args.itr):
    setting = f'{args.model}_' \
              f'{args.data}_' \
              f'ft{args.features}_' \
              f'sl{args.seq_len}_' \
              f'll{args.label_len}_' \
              f'pl{args.pred_len}_' \
              f'dm{args.d_model}_' \
              f'nh{args.n_heads}_' \
              f'el{args.e_layers}_' \
              f'dl{args.d_layers}_' \
              f'df{args.d_ff}_' \
              f'at{args.attn}_' \
              f'eb{args.embed}_' \
              f'dt{args.distil}_' \
              f'{args.des}_' \
              f'{ii}'

    exp = Exp(args)
    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(setting)

    #  print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    #  exp.test(setting)

    torch.cuda.empty_cache()

