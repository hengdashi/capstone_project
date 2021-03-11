# SmoothL1Loss with resp as the action
python -u main.py --model informer --data janestreet --root_path ./data/jane-street-market-prediction --learning_rate 0.001 --pred_len 1 --d_model 128 --d_ff 128 --attn full --itr 1 --dropout 0.1 --e_layers 0 --d_layers 3 --n_heads 8 --seq_len 96 --label_len 96 --embed fixed --train_epochs 20 --patience 7 --batch_size 512
