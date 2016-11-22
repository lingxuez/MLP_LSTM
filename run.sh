## Predict the category label of the DBPedia entity
## using Multilayer Perceptron and LSTM.
## 
## Lingxue Zhu (lzhu@cmu.edu)
##
## The data set released in this repository is a tiny example. 
## Please replace them with your favorate data sets.
## Data should be stored using the same prefix for train, valid and test sets:
##
##  mydata.train
##  mydata.valid
##  mydata.test
##
## Each data file contains two columns, separated by tab.
## The first column contains the title, a string without spaces
## The second column contains the label name, another string without spaces
## For example:
## ----------
## Lloyd_Stinson   Person
## Lobogenesis_centrota    Species
## Loch_of_Craiglush   Place
## ----------
##
## The predicted probability on test data set is stored in a .npy file,
## where the output filename can be specified by the `--output_file` option.
##

dataset="data/tiny"
batchsize=64

##############
## MLP
##############
time python mlp.py --dataset $dataset --output_file mlp_predict.npy --batch_size $batchsize

#################
## LSTM
#################
time python lstm.py --dataset $dataset --output_file lstm_predict.npy --batch_size $batchsize

