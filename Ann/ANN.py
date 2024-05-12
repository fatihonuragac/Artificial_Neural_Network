# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:24:18 2024

@author: fatihonuragac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x_l = np.load("X.npy")
Y_l = np.load("Y.npy")
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')

#%%
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
#%%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.15,random_state=42)
#%%
x_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
x_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
#%%
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
import tensorflow as tf

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(units=8, kernel_initializer='uniform',activation='relu',input_dim=x_train.shape[1]))
    classifier.add(Dense(units=4,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))






















