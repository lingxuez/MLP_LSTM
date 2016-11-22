# MLP_LSTM

My python implementation of Multilayer Perceptron (MLP) and Long Short Term Memory (LSTM) followed by a feedforward layer. 
This is a homework for 
[CMU 10-605](http://curtis.ml.cmu.edu/w/courses/index.php/Machine_Learning_with_Large_Datasets_10-605_in_Fall_2016).

## Code

The goal is character level entity classification. Specifically, we build a classifier for 
article titles in the [DBPedia data set](http://wiki.dbpedia.org/services-resources/datasets/dbpedia-datasets). 
We use the following 5 DBPedia categories:

> Person, Place, Organisation, Work, Species.

Please see `run.sh` for how to run the code.

## Data

The data set released in this repository is a tiny example. 
Please replace them with your favorate data sets.
For the code to work, data must be stored using the same prefix for train, valid and test sets.
For example:
> data/mydata.train
>
> data/mydata.valid
>
> data/mydata.test

Each data file contains two columns, separated by tab.
The first column contains the title, a string without spaces
The second column contains the label name, another string without spaces
For example:

> Lloyd_Stinson   Person
>
> Lobogenesis_centrota    Species
>
> Loch_of_Craiglush   Place

The predicted probability on test data set is stored in a .npy file,
where the output filename can be specified by the `--output_file` option.



