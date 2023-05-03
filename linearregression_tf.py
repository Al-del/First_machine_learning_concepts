from __future__ import absolute_import,division,print_function,unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  IPython.display import clear_output
from six.moves import urllib
import tensorflow._api.v2.feature_column as fc
import tensorflow as tf
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train=dftrain.pop("survived")
y_eval=dfeval.pop("survived")
CATEGORICAL_COLUM=['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
NUMERIC_COLUMS=['age','fare']
feature_colums=[]
for feature_name in CATEGORICAL_COLUM:
    vocabulary=dftrain[feature_name].unique()
    feature_colums.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))
for feature_name in NUMERIC_COLUMS:
    feature_colums.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))
def make_input_fn(data_df,label_df,num_epochs=10,shuffle=True,batch_size=32):
    def input_functionn():
        ds=tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
        if shuffle:
            ds=ds.shuffle(1000)
        ds=ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_functionn
train_input_fn=make_input_fn(dftrain,y_train)
eval_input=make_input_fn(dfeval,y_eval,num_epochs=1,shuffle=False)
linear_est=tf.estimator.LinearClassifier(feature_columns=feature_colums)
linear_est.train(train_input_fn)
result=linear_est.evaluate(eval_input)
clear_output()
result=list(linear_est.predict(eval_input))
print(dfeval.loc[55])
print(y_eval.loc[55])
print(result[55]['probabilities'][0])