# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 03:54:32 2019

@author: Mohit
"""



import os
import json

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

file_repository ="/home/moheet07/Grading/"
path=file_repository
#path ="C:/Users/Mohit/OneDrive/Documents/DR Diagnosis/"

path_OI =path+"1. Original Images/"

np.random.seed(18134271)
tf.set_random_seed(18134271)

path_gt = path + "2. Groundtruths/"
os.chdir(path)
#train_df = pd.read_csv(path+'image.csv')

train_dataframe = pd.read_csv(path+'image.csv')
train_dataframe = pd.read_csv(path+'2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv')

test_dataframe= pd.read_csv(path+'2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv')

print(train_dataframe.shape)
print(test_dataframe.shape)

Count_train = train_dataframe.shape[0]
x_train = np.empty((Count_train, 256, 315, 3), dtype=np.uint8)

print('xtrain')

#Function to call the images
def image_Giver(path, image_id):
     
    img = Image.open(path +image_id)
    return img

path_train = path + 'Training_All/'
for i, image_id in enumerate(tqdm(train_dataframe['Image name'])):
    x_train[i, :, :, :] = image_Giver(path_train, image_id+'.jpg')
    
Count_test = test_dataframe.shape[0]
x_test = np.empty((Count_test, 256, 315, 3), dtype=np.uint8)

path_test = path_OI+'b. Testing Set/'
for i, image_id in enumerate(tqdm(test_dataframe['Image name'])):
    x_test[i, :, :, :] = image_Giver(path_test, image_id+'.jpg')  

#Creating a dataframe of dependent variable     
y_train = pd.get_dummies(train_dataframe['Retinopathy grade']).values

print('y_train')

y_train_multilabel = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multilabel[:, 1] = y_train[:, 1]

for i in range(0, -1, -1):
    y_train_multilabel[:, i] = np.logical_or(y_train[:, i], y_train_multilabel[:, i+1])

print("Original Supervised y-train labels:", y_train.sum(axis=0))
print("Multilabel version Supervised Data:", y_train_multilabel.sum(axis=0))


x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train_multilabel, 
    test_size=0.15, 
    random_state=18134271
)

print("3")




BATCH_SIZE = 32

def Augment_images():
    return ImageDataGenerator(
        zoom_range=0.15, 
        fill_mode='constant',
        cval=0.,  
        horizontal_flip=True, 
        vertical_flip=True 
    )

# Using original generator
    
print('Image Data generator')
image_generator = Augment_images().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=18134271)


class kappa_metrics(Callback):
    def train_begin(self, logs={}):
        self.val_kappas = []

    def epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model_denseNet.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Better Validation Kappa, hence Saving model")
            self.model_denseNet.save('model_des_DR.h5')

        
    
DENSENET_121_WEIGHTS_PATH = (r'https://github.com/titu1994/DenseNet/releases/download'
                             r'/v3.0/DenseNet-BC-121-32.h5')    
 
   
densenet = DenseNet121(
    weights=path+'DenseNet-BC-121-32_no-top.h5',include_top=False,  input_shape=(256, 315,3)
)

def build_model():
    model_denseNet = Sequential()
    model_denseNet.add(densenet)
    model_denseNet.add(layers.GlobalAveragePooling2D())
    model_denseNet.add(layers.Dropout(0.5))
    model_denseNet.add(layers.Dense(2, activation='sigmoid'))
    
    model_denseNet.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model_denseNet

model_denseNet = build_model()
model_denseNet.summary()

kappa_metrics = Metrics()

history = model_denseNet.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[kappa_metrics]
)


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()

model_denseNet.load_weights('model_des_DR.h5')

y_pred = model_denseNet.predict(x_test) > 0.5
y_pred = y_pred.astype(int).sum(axis=1) - 1

test_df['diagnosis'] = y_pred
test_df.to_csv('submission.csv',index=False)

pred = pd.read_csv('submission.csv')
y_test = pred['Retinopathy grade']
y_pred = pred['diagnosis']

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))    



