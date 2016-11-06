bashsize=64

## MLP
# python mlp.py --max_len 10 --num_hid 50 --batch_size $bashsize --dataset ../data/smaller \
#     --epochs 15 --init_lr 0.05 --output_file ../output/predict_smaller.npy

## LSTM
python lstm.py --max_len 10 --num_hid 50 --batch_size $bashsize --dataset ../data/smaller \
       --epochs 15 --init_lr 0.5 --output_file ../lstm_out/predict_smaller.npy \