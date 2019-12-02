#vgg16 Model performance test

import json
import math
import os

from PIL import Image
import numpy as np
from keras import layers
from keras.applications.resnet50 import ResNet50
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score

import tensorflow as tf
from tqdm import tqdm
#file_repository = "C:/Users/Mohit/OneDrive/Documents/Grading/"
#path ="C:/Users/Mohit/OneDrive/Documents/Grading/"

file_repository ="/home/moheet07/Grading_2/"
path=file_repository


path_OI =path+"3. Original Images/"

np.random.seed(18134271)
tf.set_random_seed(18134271)

path_gt = path + "2. Groundtruths/"
os.chdir(path)
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
y_train_multilabel[:, 4] = y_train[:, 4]

#Creating multilabel supervised data
for i in range(3, -1, -1):
    print(i)
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
        
        y_pred = self.model_resNet.predict(X_val) > 0.5
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
            self.model_resNet.save('model_resNet_train.h5')

        return
    
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 315, 3))
resnet.summary()


def build_model():
    model_resNet = Sequential()
    model_resNet.add(resnet)
    model_resNet.add(layers.GlobalAveragePooling2D())
    model_resNet.add(layers.Dropout(0.5))
    model_resNet.add(layers.Dense(5, activation='sigmoid'))
    
    model_resNet.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model_resNet

model_resNet = build_model()
model_resNet.summary()

kappa_metrics = kappa_metrics()
print('8')

history = model_resNet.fit_generator(
    image_generator,
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

model_resNet.load_weights('model_resNet_train.h5')
y_val_pred = model_resNet.predict(x_val)

y_pred = model_resNet.predict(x_test) > 0.5
y_pred = y_pred.astype(int).sum(axis=1) - 1

test_dataframe['predictions'] = y_pred
test_dataframe.to_csv('result.csv',index=False)

metaData = pd.read_csv('result.csv')
y_test = metaData['Retinopathy grade']
y_pred = metaData['predictions']

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))    



