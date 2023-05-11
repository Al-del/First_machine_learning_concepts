#Introduction to regresion
#Regresion=predicting numerical variable based on some pther combinations of variables
#Import tenserflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#Creating some data to view and fit
x=np.array([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0],dtype=float)
y=np.array([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0],dtype=float)
plt.scatter(x,y)
#Imput and output
#House price prediction problem
#Demo tensor for housing price prediction
house_info=tf.constant(["bedroom","bathroom","garage"])
house_price=tf.constant(939700)
input_shape=x[0].shape
output_sape=y[0].shape
print(input_shape,output_sape)
X=tf.constant(x,dtype=tf.float64)
Y=tf.constant(y,dtype=tf.float64)
tf.cast(tf.constant(X),dtype=tf.float32)
y = tf.cast(tf.constant(y),dtype=tf.float32)
#Steps in modeling with tenserflow
#1. Creating a model-define the input and output layers ,as well as the hidden layers of a deep learning model
# 2. Compiling the model-define the loss function( in other words,the function which tells our model how wrong it is) and the optimizer (tells our model how to improve the patterns its learning)and evaluate matrics (what we can use to interprerat the performance of our model
# 3. fitting a model -letting the model try to find pattens between X and Y
#Set random seed
# Set random seed
def small_model():
    tf.random.set_seed(42)

    # Create a model using the Sequential API
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
                  optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
                  metrics=["mae"])

    # Fit the model
    # model.fit(X, y, epochs=5) # this will break with TensorFlow 2.7.0+
    model.fit(tf.expand_dims(X, axis=-1), y, epochs=5)
    print(model.predict([13.0]))
# improve the model
#We can improve our model by altering the steps we took to create a model
#1.Create a model--we might add more neurons,increase the value of hidden units,change the activation fucntionof each layer
#Compiling the model --change optimization functions (learning rates of the function)
#Fitting model--give it more data
def imprv():
    model=tf.keras.Sequential([
        tf.keras.layers.Dense(1)

    ])
    model.compile(loss=tf.keras.losses.mae,  # mae is short for mean absolute error
                  optimizer=tf.keras.optimizers.SGD(),  # SGD is short for stochastic gradient descent
                  metrics=["mae"])
    model.fit(tf.expand_dims(X, axis=-1), Y, epochs=100)

    print(model.predict([17.0]))
def second_improved_model():
    model=tf.keras.Sequential([
        tf.keras.layers.Dense(100,activation=None),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=["mae"])
    model.fit(tf.expand_dims(X, axis=-1), Y, epochs=100)
    print(model.predict([17.0]))
#The learning rate is the most important parameter
#Evaluating model
#A typical workflow
#Build it->Evaluate ut-> tweak->fit it -.Evaluate it ->tweak it->etc.
#VIsualize:
#DAta
#What does it look like
#The model itself
#The training of the model
#The predictions
#Make a biggerdataset
A=tf.range(-100,100,4)
print(A)
B=A+10
print(B)
plt.plot(A,B)
#The 3 sets concept....
#1 set The training set 70-80% of ur data
#2 VAlidation - your model gets tuned on this data which is 10-15% data available
#3 the test set the model gets evaluated on this daata ,this set is typically 10-15% of ur data
#Check the lenght of how many samples we have
print(len(A))
#Split the data into train and test set
A_train=A[:40]#First 40
B_train=B[:40]
A_test=A[40:]#Last 10
B_test=B[40:]
plt.figure(figsize=(10,7))
plt.scatter(A_train,B_train,c="b")
plt.scatter(A_test,B_test,c="g")
#plt.show()
#lets have a loo of how to build a NN to use our data
model=tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])
model.fit(tf.expand_dims(A_train, axis=-1),B_train,epochs=100)
print(model.summary())
tf.random.set_seed(42)
model=tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])
print(model.summary())
#Total params=total numbers of parameters in the model
#Trainable parameters=these are the parameters the model can update
#Non-trainable params=these parameters arent updated during training,these is typical when u bring already learn patters during transfer learning
model.fit(tf.expand_dims(A_train, axis=-1),B_train,epochs=100,verbose=1)