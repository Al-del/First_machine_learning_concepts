# Introduction to regresion
# Regresion=predicting numerical variable based on some pther combinations of variables
# Import tenserflow
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Convolution2D, Activation
from keras.optimizers import SGD
from keras.utils import plot_model
import pandas as pd

# Creating some data to view and fit
x = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0], dtype=float)
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0], dtype=float)
plt.scatter(x, y)
# Imput and output
# House price prediction problem
# Demo tensor for housing price prediction
house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant(939700)
input_shape = x[0].shape
output_sape = y[0].shape
print(input_shape, output_sape)
X = tf.constant(x, dtype=tf.float64)
Y = tf.constant(y, dtype=tf.float64)
tf.cast(tf.constant(X), dtype=tf.float32)
y = tf.cast(tf.constant(y), dtype=tf.float32)


# Steps in modeling with tenserflow
# 1. Creating a model-define the input and output layers ,as well as the hidden layers of a deep learning model
# 2. Compiling the model-define the loss function( in other words,the function which tells our model how wrong it is) and the optimizer (tells our model how to improve the patterns its learning)and evaluate matrics (what we can use to interprerat the performance of our model
# 3. fitting a model -letting the model try to find pattens between X and Y
# Set random seed
# Set random seed
def small_model():
    tf.random.set_seed(42)

    # Create a model using the Sequential API
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.mae,  # mae is short for mean absolute error
                  optimizer=tf.keras.optimizers.SGD(),  # SGD is short for stochastic gradient descent
                  metrics=["mae"])

    # Fit the model
    # model.fit(X, y, epochs=5) # this will break with TensorFlow 2.7.0+
    model.fit(tf.expand_dims(X, axis=-1), y, epochs=5)
    print(model.predict([13.0]))


# improve the model
# We can improve our model by altering the steps we took to create a model
# 1.Create a model--we might add more neurons,increase the value of hidden units,change the activation fucntionof each layer
# Compiling the model --change optimization functions (learning rates of the function)
# Fitting model--give it more data
def imprv():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)

    ])
    model.compile(loss=tf.keras.losses.mae,  # mae is short for mean absolute error
                  optimizer=tf.keras.optimizers.SGD(),  # SGD is short for stochastic gradient descent
                  metrics=["mae"])
    model.fit(tf.expand_dims(X, axis=-1), Y, epochs=100)

    print(model.predict([17.0]))


def second_improved_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation=None),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=["mae"])
    model.fit(tf.expand_dims(X, axis=-1), Y, epochs=100)
    print(model.predict([17.0]))


def shor_things_for_the_model():
    # The learning rate is the most important parameter
    # Evaluating model
    # A typical workflow
    # Build it->Evaluate ut-> tweak->fit it -.Evaluate it ->tweak it->etc.
    # VIsualize:
    # DAta
    # What does it look like
    # The model itself
    # The training of the model
    # The predictions
    # Make a biggerdataset
    A = tf.range(-100, 100, 4)
    print(A)
    B = A + 10
    print(B)
    plt.plot(A, B)
    # The 3 sets concept....
    # 1 set The training set 70-80% of ur data
    # 2 VAlidation - your model gets tuned on this data which is 10-15% data available
    # 3 the test set the model gets evaluated on this daata ,this set is typically 10-15% of ur data
    # Check the lenght of how many samples we have
    print(len(A))
    # Split the data into train and test set
    A_train = A[:40]  # First 40
    B_train = B[:40]
    A_test = A[40:]  # Last 10
    B_test = B[40:]
    plt.figure(figsize=(10, 7))
    plt.scatter(A_train, B_train, c="b")
    plt.scatter(A_test, B_test, c="g")
    # plt.show()
    # lets have a loo of how to build a NN to use our data
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=["mae"])
    model.fit(tf.expand_dims(A_train, axis=-1), B_train, epochs=100)
    print(model.summary())
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(40, input_shape=[1]),
        tf.keras.layers.Dense(1, name="output_layer")
    ], name="One_if_our_mostmdls")
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=["mae"])

    # Total params=total numbers of parameters in the model
    # Trainable parameters=these are the parameters the model can update
    # Non-trainable params=these parameters arent updated during training,these is typical when u bring already learn patters during transfer learning
    model.fit(tf.expand_dims(A_train, axis=-1), B_train, epochs=100, verbose=1)
    model.summary()
    plot_model(model=model, to_file='model.png', show_shapes=True)
    print(model.predict([17.0]))

    # If u feel that u are going to use some kind of functiuality , is a good ideea to create a function


def ploting_things():
    def plot_predictions(train_Data,
                         train_labels,
                         test_data,
                         test_label,
                         predictions):
        # Plot training and test to compare each others
        plt.figure(figsize=(10, 7))
        # Plot training data
        plt.scatter(train_labels, train_Data, c="b", label="Training data")
        plt.scatter(test_label, test_data, c="g", label="Tested data")
        plt.scatter(test_data, predictions, c="r", label="Predictions")
        plt.legend();
        plt.show()

    def initialize_a_model():
        # Visualize our model predictions
        # Often you'll see  this in the form of y_test  or y_truth
        # Make some predictions
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=[1])  # define the input_shape to our model
        ])
        model.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.SGD(),
                      metrics=["mae"])
        A = tf.range(-100, 100, 4)
        B = A + 10
        A_train = A[:40]  # First 40
        B_train = B[:40]
        A_test = A[40:]  # Last 10
        B_test = B[40:]
        model.fit(tf.expand_dims(A_train, axis=1), B_train, epochs=28, verbose=0)
        y_predictions = model.predict(A_test)
        plot_predictions(train_Data=A_train, train_labels=B_train, test_data=A_test, test_label=B_test,
                         predictions=y_predictions)

    initialize_a_model()


# Evaluating our mdoels with regression evaluation metrics
# Depending the problem that u are working on , there would be different models of evaluation
# Sincwe wew are working on Regression problerms ,two of the main metrics:
# MAE-mean absolute error-"On average ,how wrong is each of my model's prediction"
# MSE-mean swuared error-"Square the avarege error
# Evaluate the model on the test_set
def paly_with_the_errors():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])  # define the input_shape to our model
    ])
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=["mae"])
    A = tf.range(-100, 100, 4)
    B = A + 10
    A_train = A[:40]  # First 40
    B_train = B[:40]
    A_test = A[40:]  # Last 10
    B_test = B[40:]
    model.fit(tf.expand_dims(A_train, axis=1), B_train, epochs=40, verbose=0)
    print(model.evaluate(A_test, B_test))
    print(model.predict([17.0]))
    X = tf.range(-100, 100, 4)
    y = A + 10
    X_train = X[:40] # first 40 examples (80% of data)
    y_train = y[:40]

    X_test = X[40:] # last 10 examples (20% of data)
    y_test = y[40:]

    y_preds = model.predict(X_test)
    mae = tf.metrics.mean_absolute_error(y_true=y_test,
                                         y_pred=tf.squeeze(y_preds))
    mse= tf.metrics.mean_squared_error(y_true=y_test,
                                         y_pred=tf.squeeze(y_preds))
    hubb= tf.losses.huber(y_true=y_test,
                        y_pred=tf.squeeze(y_preds))
    print(mae)
    print(mse)
    print(hubb)
    #LEt's make some fucntions
    def mae(y_true,y_pred):
        return tf.metrics.mean_absolute_error(y_true=y_true,
                                              y_pred=y_pred)
    def mse(y_true,y_pred):
        return tf.metrics.mean_squared_error(y_true=y_true,
                                              y_pred=y_pred)
#Running experiments to improve the model
#Get mpre data
#Make your model larger
#Train for longer
#LEts do 3 modelling experiments
#1 1 layer, 100 epoches
#2 layers,epoches 100
#3 3 layers 100 epoches
def save_and_copare_things():
    np.random.seed(1337)
    tf.random.set_seed(1337)
    def mae(y_true,y_pred):
            return tf.metrics.mean_absolute_error(y_true=y_true,
                                                  y_pred=y_pred)
    def mse(y_true,y_pred):
            return tf.metrics.mean_squared_error(y_true=y_true,
                                                  y_pred=y_pred)

    X = tf.range(-100, 100, 4)
    y = X+ 10
    X_train = X[:40] # first 40 examples (80% of data)
    y_train = y[:40]

    X_test = X[40:] # last 10 examples (20% of data)
    y_test = y[40:]
    a="u"
    def plot_predictions(train_data=X_train,
                         train_labels=y_train,
                         test_data=X_test,
                         test_labels=y_test,
                         predictions=a):
        """
        Plots training data, test data and compares predictions.
        """
        plt.figure(figsize=(10, 7))
        # Plot training data in blue
        plt.scatter(train_data, train_labels, c="b", label="Training data")
        # Plot test data in green
        plt.scatter(test_data, test_labels, c="g", label="Testing data")
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", label="Predictions")
        # Show the legend
        plt.legend();
        plt.show()
    model_1=tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])
    model_1.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.SGD(),
                      metrics=["mae"])
    model_1.fit(tf.expand_dims(X_train, axis=1), y_train, epochs=40, verbose=0)
    y_preds_1 = model_1.predict(X_test)
    plot_predictions(predictions=y_preds_1)
    mae_1 = mae(y_test, y_preds_1.squeeze()).numpy()
    mse_1 = mse(y_test, y_preds_1.squeeze()).numpy()
    print(mae_1)
    print(mse_1)
    model_2=tf.keras.Sequential([
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])
    model_2.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.SGD(),
                      metrics=["mae"])
    model_2.fit(tf.expand_dims(X_train, axis=1), y_train, epochs=40, verbose=0)

    y_preds_2 = model_2.predict(X_test)
    plot_predictions(predictions=y_preds_2)
    mae_2 = mae(y_test, y_preds_2.squeeze()).numpy()
    mse_2 = mse(y_test, y_preds_2.squeeze()).numpy()
    print(mae_2)
    print(mse_2)
    model_3=tf.keras.Sequential([
        tf.keras.layers.Dense(500),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])
    model_3.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.SGD(),
                      metrics=["mae"])
    model_3.fit(tf.expand_dims(X_train, axis=1), y_train, epochs=40, verbose=0)
    y_preds_3 = model_3.predict(X_test)
    plot_predictions(predictions=y_preds_3)
    mae_3 = mae(y_test, y_preds_3.squeeze()).numpy()
    mse_3 = mse(y_test, y_preds_3.squeeze()).numpy()
    print(mae_3)
    print(mse_3)
    #You wnat to star with small wxperiments and after  increase the complexity
    #Lets see how we can compare our medel's result using pandas dataframe
    model_result=[["model_1",mae_1,mse_1,],
                  ["model_2",mae_2,mse_2,],
                  ["model_3",mae_3,mse_3,]]
    print(model_result)
    #One of ur main goals sould be to minimize the time between ur experiments.The more experiments u do, the more things u'll figure out ehich dont work and in turn ,get closer to figuring out what does work
    #Remember the machine learning practitioner's motto"Visualize,visualize,visualize
    #Resources
    #TenserBoard-a component to help track of modelling experiments
    #Weights & Biasses-a tool for  tracking all kinds of of ML experiments
    #Saving models
    #Saving it allows us to use it wvwrywhere we want
    model_2.save("MYMODE")
    print(model_2.predict([17.0]))
loadded_h5=tf.keras.models.load_model("MYMODE.h5")
loadded=tf.keras.models.load_model("MYMODE")
loadded.summary()
print(loadded.predict([17.0]))
print(loadded_h5.predict([17.0]))
