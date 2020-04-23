# Artificial_Neural_Network_for_Regression
#import_Libraries
import numpy as np
import pandas as pd
import tensorflow as tf 

#Check the tensorflow version
tf.__version__
#//If you have any problem during processing email me @ rdulal71@gmail.com //

#PART-1 Data Processing
#Import Data Set
dataset = pd.read_excel("File directory or the File name in excel")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Print the Data Set
print(X)
print(y)

#Spliting the dataset into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Part-2 Building an ANN
#Initialize ANN
ann = tf.keras.models.Sequential()

#Adding input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units ='input numbers', activation = 'relu'))

#Adding second hidden layer
ann.add(tf.keras.layers.Dense(units = 'input numbers', activation = 'relu'))

#Adding Output Layer
ann.add(tf.keras.layers.Dense(units = 1))

#Part-3 Training an ANN
#Compile the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Training the ANN model on train set
ann.fit(X_train,y_train, batch_size = 32, epochs = 'expected number')

#Prediction of the Result of the test set
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))


#ThankYou!


#//If you have any problem during processing email me @ rdulal71@gmail.com //
