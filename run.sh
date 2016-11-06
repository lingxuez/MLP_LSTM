bashsize=64

## MLP
# python mlp.py --max_len 10 --num_hid 50 --batch_size $bashsize --dataset ../data/smaller \
#     --epochs 15 --init_lr 0.05 --output_file ../mlp_out/predict_smaller.npy

## LSTM
python lstm.py --max_len 10 --num_hid 50 --batch_size $bashsize --dataset ../data/tiny \
       --epochs 15 --init_lr 0.5 --output_file ../lstm_out/predict_tiny.npy \
## ideal_lstm_loss: 0.9 student_lstm_loss: 1.60943791243

## tar -cvf hw5.tar lstm.py mlp.py functions.py autograd.py