# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 23:47:38 2019

@author: Mohit
"""

#Building The CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import os

#Initialising CNN

CNNmodel = Sequential()

#Step1 Convolution
#We defined the input image size as 32,32 to reduce the operation cost.
CNNmodel.add(Convolution2D(64,3,3, input_shape=(64,64,3), activation='relu'))

#Step2 Pooling
CNNmodel.add(MaxPool2D(pool_size=(2,2)))

#Adding dropout layer
#To refrain our model to overfit.
CNNmodel.add(Dropout(0.25))

#added second conv layer so that we can extract features more efficiently.
CNNmodel.add(Convolution2D(64,3,3, input_shape=(64,64,3), activation='relu'))
CNNmodel.add(MaxPool2D(pool_size=(2,2)))
CNNmodel.add(Dropout(0.25))


CNNmodel.add(Convolution2D(64,3,3, input_shape=(64,64,3), activation='relu'))
CNNmodel.add(MaxPool2D(pool_size=(2,2)))
CNNmodel.add(Dropout(0.25))

CNNmodel.add(Convolution2D(64,3,3, input_shape=(64,64,3), activation='relu'))
CNNmodel.add(MaxPool2D(pool_size=(2,2)))
CNNmodel.add(Dropout(0.25))



#Step3 Flattening
CNNmodel.add(Flatten())

#Step 4 Full Connection
CNNmodel.add(Dense(output_dim= 128, activation= 'relu'))
#Since we are dealing in multiple classes we set the ac function as softmax.
CNNmodel.add(Dense(output_dim= 5, activation= 'softmax'))

#Step 5
CNNmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#We placed the images in SSD from HDD to reduce the operation cost. 

file_repository="C:\\Users\\Mohit\\OneDrive\\Documents\\Sem3\\AllSegmentationGroundtruths"
os.chdir(file_repository)


#Step6 Fitting images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

from keras.callbacks import ReduceLROnPlateau
annealer  = ReduceLROnPlateau(monitor='val_acc')


training_set = train_datagen.flow_from_directory('Training Set', target_size=(64,64), batch_size=32, class_mode='categorical')

test_set = test_datagen.flow_from_directory('Testing Set', target_size=(64,64), batch_size=32, class_mode='categorical')
#We added the loss function for the model. Since we have multiple classes so we used categorical crossentropy function.
CNNmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

CNNmodel.fit_generator(training_set,
                       steps_per_epoch=241,
                       epochs=5,
                       validation_data=test_set,
                       validation_steps=122,
                       callbacks=[annealer])

