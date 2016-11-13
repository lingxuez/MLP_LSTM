bashsize=64

## MLP
# time python mlp.py --dataset ../data/smaller --output_file ../mlp_out/predict_smaller.npy

## LSTM
time python lstm.py --dataset ../data/smaller --output_file ../lstm_out/predict_smaller.npy
## ideal_lstm_loss: 0.9 student_lstm_loss: 1.60943791243

## tar -cvf hw5.tar lstm.py mlp.py functions.py autograd.py