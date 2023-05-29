import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import matplotlib.pyplot as plt

insurance=pd.read_csv("insurance.csv")
ins_one_hot=pd.get_dummies(insurance)
#Create X and Y values
X=ins_one_hot.drop("charges",axis=1)
y=ins_one_hot["charges"]
#Create training and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
tf.random.set_seed(42)
insurence_model=tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
insurence_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=8),
                        metrics=["mae"])
#insurence_model.fit(tf.constant(X_train,dtype=tf.float32),tf.constant(y_train,dtype=tf.float32),epochs=300)
#print(insurence_model.evaluate(tf.constant(X_test,dtype=tf.float32),y_test))
#To try to improve our model
#Train for longer
#Add an extra lyer
np.random.seed(42)
insurence_1_model=tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
insurence_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=10),
                        metrics=["mae"])
#insurence_model.fit(tf.constant(X_train,dtype=tf.float32),tf.constant(y_train,dtype=tf.float32),epochs=300)
#print(insurence_model.evaluate(tf.constant(X_test,dtype=tf.float32),y_test))
np.random.seed(1332)
insurence_model_3=tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
insurence_model_3.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["mae"])
history=insurence_model_3.fit(tf.constant(X_train,dtype=tf.float32),tf.constant(y_train,dtype=tf.float32),epochs=10)
print(insurence_model_3.evaluate(tf.constant(X_test,dtype=tf.float32),y_test))
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epoches")
plt.show()
#how long should u train for?
#It depends on the provlem u r working for
#SO tenserflow has a solution
#It is calledthe earlystopping callback
#A tenserflow component that stops training when it starts to overfit
 #Preprocessing data
 #In terms of scalling values,neuronal networks tend to perform normalisation
#Create a colum tranformer
ct=make_column_transformer(
    (MinMaxScaler(),["age","bmi","children"]),
    (OneHotEncoder(handle_unknown="ignore"),["sex"],["smoker"],["region"])
)
#Create X and y
X=insurance.drop("charges", axis=1)
y=ins_one_hot["charges"]
#Create training and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
ct.fit(X_train)
X_train_normal=ct.transform(X_train)
X_test_normal=ct.transform(X_test)
print(X_test_normal[0])
# Create column transformer (this will help us normalize/preprocess our data)
