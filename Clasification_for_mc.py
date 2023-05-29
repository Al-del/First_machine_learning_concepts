import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
n_samples=1000
#Create circles
X,y=make_circles(n_samples,
                 noise=0.03,
                 random_state=42)
circles=pd.DataFrame({"X0":X[:,0],"X1": X[:,1],"Label": y})
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdYlBu)
#plt.show()
print(X.shape,y.shape)
print(len(X),len(y))
print(X[10],y[10])
#Steps in modelling
tf.random.set_seed(42)
def create_first_model():
    model_1=tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])
    model_1.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    model_1.fit(tf.expand_dims(X, axis=1),y,epochs=200,verbose=0)
    print(model_1.evaluate(X,y))
def second_function():
    model_2=tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1)
    ])
    model_2.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    model_2.fit(tf.expand_dims(X, axis=1),y,epochs=100,verbose=0)
    print(model_2.evaluate(X,y))
#Imrpoving our model
#Creae a model-add more layers
#Compillng models-chose a different optimization
#Fitting our model-fit our model for more epoches
model_3=tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])
model_3.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
model_3.fit(tf.expand_dims(X, axis=1), y, epochs=100, verbose=0)
print(model_3   .evaluate(X, y))
def plot_decision_boundary(model,X,y):
        x_min,x_max=X[:,0].min()-0.1,X[:,0].max()+0.1
        y_min, y_max = y[:, 0].min() - 0.1, y[:, 0].max() + 0.1
        xx,yy=np.meshgrid(np.linspace(x_min,x_max,100),
                          np.linspace(y_min,y_max,100))

