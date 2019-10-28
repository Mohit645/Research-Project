# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 23:45:35 2019

@author: Mohit
"""

import os
import shutil
import cv2
file_repository="C:\\Users\\Mohit\\OneDrive\\Documents\\Sem3\\AllSegmentationGroundtruths\\"
os.chdir(file_repository)

import os
from glob import glob
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt



#Copying files to new location and making new repository

path ="C:\\Users\\Mohit\\OneDrive\\Documents\\Sem3\\NewSegmentation1\\"
#os.mkdir(path)
os.chdir(path)
newClasses = ['HE\\', 'Background\\']

for i in newClasses:
    os.mkdir(path+i)

set1 = ["Testing Set\\", 'Training Set\\']
oldClasses = ['1. Microaneurysms\\', '2. Haemorrhages\\', '3. Hard Exudates\\', '4. Soft Exudates\\', '5. Optic Disc\\']

#Reading the images


for i in set1:
    for j in oldClasses:
        images = [f for f in os.listdir(file_repository+i+j) if os.path.splitext(f)[-1] == '.tif']
        for k in images:
            src_dir = file_repository+i+j+k
            if(j=='3. Hard Exudates\\'):
                dest_dir = path+newClasses[0]
                
            else:
                dest_dir = path+newClasses[1]
            shutil.copy(src_dir,dest_dir)

#Resizing the images(256*256)
for j in newClasses:
    images = [f for f in os.listdir(path+j) if os.path.splitext(f)[-1] == '.tif'] 
    for i in images:    
        img = Image.open(path+j+i)
        img = img.resize((256,256), Image.ANTIALIAS)
        img.save(path+j+i)

#Splitting the dataset
#Creating test and training sets
os.mkdir(path+'Train\\')
os.mkdir(path+'Test\\')
os.mkdir(path+'Validation\\')

#Creating test and training sets
os.mkdir(path+'Train\\')
os.mkdir(path+'Test\\')

from sklearn.model_selection import train_test_split
for i in newClasses:
    newPath =path+i
    images = [f for f in os.listdir(newPath) if os.path.splitext(f)[-1] == '.jpg' or os.path.splitext(f)[-1] == '.tif']
    train,test = train_test_split(images, test_size = 1/4, random_state = 0)
    
    dst_dir = path+"Test\\"+i
    os.mkdir(dst_dir)
    
    for k in test:
        src_dir=path+i+k
        shutil.copy(src_dir,dst_dir)    

    dst_dir = path+"Train\\"+i
    os.mkdir(dst_dir)    
    for j in train:
        src_dir=path+i+j
        shutil.copy(src_dir,dst_dir)
    del(images)
    
#Splitting Test into Test and Validation
from sklearn.model_selection import train_test_split
for i in newClasses:
    newPath =path+'Test\\'+i
    images = [f for f in os.listdir(newPath) if os.path.splitext(f)[-1] == '.jpg' or os.path.splitext(f)[-1] == '.tif']
    validation,test = train_test_split(images, test_size = 1/2, random_state = 0)
    
    dst_dir = path+"Test1\\"+i
    os.mkdir(dst_dir)
    
    for k in test:
        src_dir=path+i+k
        shutil.copy(src_dir,dst_dir)    

    dst_dir = path+"Validation\\"+i
    os.mkdir(dst_dir)    
    for j in validation:
        src_dir=path+i+j
        shutil.copy(src_dir,dst_dir)
    del(images)

os.rmdir(path+'Test\\')

os.rename(path+'Test1', path+'Test')





#DataAugmentation
newPath=path+'Train\\'
images = [f for f in os.listdir(newPath+newClasses[0]) if os.path.splitext(f)[-1] == '.tif']          
itr = round(10000/len(images))

for image in images:
        img = load_img(newPath+'HE\\'+image) 
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(rotation_range=90)
        it = datagen.flow(samples, batch_size=32, save_to_dir=newPath+'HE\\' ,save_prefix='AUG_ISIC_GEN_'+image, save_format='jpg')
        for j in range(itr):
            batch = it.next()
            
            

images = [f for f in os.listdir(newPath+newClasses[1]) if os.path.splitext(f)[-1] == '.tif']  
itr =  round(10000/len(images)) 

for image in images:
        img = load_img(newPath+'Background\\'+image) 
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(rotation_range=90)
        it = datagen.flow(samples, batch_size=32, save_to_dir=newPath+'Background\\' ,save_prefix='AUG_ISIC_GEN_'+image, save_format='jpg')
        for j in range(itr):
            batch = it.next()
            
#Augmenting the validation set

newPath=path+'Train\\'
images = [f for f in os.listdir(newPath+newClasses[0]) if os.path.splitext(f)[-1] == '.tif']          
itr = round(10000/len(images))

for image in images:
        img = load_img(newPath+'HE\\'+image) 
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(rotation_range=90)
        it = datagen.flow(samples, batch_size=32, save_to_dir=newPath+'HE\\' ,save_prefix='AUG_ISIC_GEN_'+image, save_format='jpg')
        for j in range(itr):
            batch = it.next()
            
            

images = [f for f in os.listdir(newPath+newClasses[1]) if os.path.splitext(f)[-1] == '.tif']  
itr =  round(10000/len(images)) 

for image in images:
        img = load_img(newPath+'Background\\'+image) 
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(rotation_range=90)
        it = datagen.flow(samples, batch_size=32, save_to_dir=newPath+'Background\\' ,save_prefix='AUG_ISIC_GEN_'+image, save_format='jpg')
        for j in range(itr):
            batch = it.next()
     



import keras
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from keras.optimizers import Adam 
from keras.preprocessing.image import ImageDataGenerator # to augmenting our images for increasing accuracy
from keras.utils.vis_utils import plot_model
import scipy
from sklearn.model_selection import train_test_split # to split our train data into train and validation sets
import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(13) 




num_classes = 2 
batch_size = 50 # taken from benchmark
epochs = 20 # 20 Epoch is enough
img_rows, img_cols = 32, 32 
input_shape = (img_rows, img_cols,3) # We'll use this while building layers

model = Sequential()
model.add(Conv2D(32,(3,3), activation = 'relu', input_shape =input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(32,(3,3), activation = 'relu', input_shape =input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3), activation = 'relu', input_shape =input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3), activation = 'relu', input_shape =input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3), activation = 'relu', input_shape =input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3), activation = 'relu', input_shape =input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(256,(3,3), activation = 'relu', input_shape =input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(256,(3,3), activation = 'relu', input_shape =input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same'))
model.add(Dropout(0.2))

# To be able to merge into fully connected layer we have to flatten
model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(1000, activation="relu"))
#Second fully connected layer
model.add(Dense(100, activation="relu"))
#activation 
model.add(Dense(num_classes, activation = "sigmoid"))
#model.add(Dense(num_classes, activation = "softmax"))
# Compile the model with loss and metrics
model.compile(optimizer =  Adam() , loss = "binary_crossentropy", metrics=["accuracy"])




#Step6 Fitting images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

from keras.callbacks import ReduceLROnPlateau
annealer  = ReduceLROnPlateau(monitor='val_acc')

import os
path ="C:\\Users\\Mohit\\OneDrive\\Documents\\Sem3\\NewSegmentation1\\"
os.chdir(path)


training_set = train_datagen.flow_from_directory('Train', target_size=(32,32), batch_size=batch_size, class_mode='categorical')
#training_set = ('Train', target_size=(32,32), batch_size=batch_size, class_mode='categorical')

test_set = test_datagen.flow_from_directory('Test', target_size=(32,32), batch_size=batch_size, class_mode='categorical')
#We added the loss function for the model. Since we have multiple classes so we used categorical crossentropy function.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(training_set,
                       steps_per_epoch=271,
                       epochs=epochs,
                       validation_data=test_set,
                       validation_steps=92,
                       callbacks=[annealer])