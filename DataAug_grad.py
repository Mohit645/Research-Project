# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 02:10:47 2019

@author: Mohit
"""

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

#file_repository ="C:/Users/Mohit/OneDrive/Documents/Grading/"
file_repository ="/home/moheet07/Grading/"

os.chdir(file_repository)
path_OI =file_repository+"1. Original Images/"
path_gt=file_repository+"2. Groundtruths/"

image_paths = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path_OI, '*', '*.jpg'))}


severity_grading = {
    0: 'Normal',
    1: 'Mild NPDR',
    2: 'Moderate NPDR',
    3: 'Severe NPDR',
    4: 'Proliferative DR'
}

#Reading the meta data we have.
grad = pd.read_csv(os.path.join(path_gt, 'a. IDRiD_Disease Grading_Training Labels.csv'))
grad.head()
len(grad)
#Making a new column in dataframe to accomodate the images path to perform preprocessing required.
grad['image_path'] = grad['Image name'].map(image_paths.get)
grad['severity'] = grad['Retinopathy grade'].map(severity_grading.get)
 
#Transforming the type to the categorical values
grad['severity_id'] = pd.Categorical(grad['severity']).codes

#Getting the count per class to use for data augmentation stage.
sg_normal=0
sg_Mild_NDPR=0
sg_Moderate_NPDR=0
sg_Severe_NPDR=0
sg_Proliferative_DR=0

for i in range(len(grad)):
    if(grad['severity'][i]=='Normal'):
        sg_normal=sg_normal+1
        
    elif(grad['severity'][i]=='Mild NPDR'):
        sg_Mild_NDPR=sg_Mild_NDPR+1
        
    elif(grad['severity'][i]=='Moderate NPDR'):
        sg_Moderate_NPDR=sg_Moderate_NPDR+1
        
    elif(grad['severity'][i]=='Severe NPDR'):
        sg_Severe_NPDR=sg_Severe_NPDR+1
        
    elif(grad['severity'][i]=='Proliferative DR'):
        sg_Proliferative_DR=sg_Proliferative_DR+1
        
print(sg_normal,sg_Mild_NDPR,sg_Moderate_NPDR,sg_Severe_NPDR,sg_Proliferative_DR)
#Validating total number of images should be equal to 10,015.
print(sg_normal+sg_Mild_NDPR+sg_Moderate_NPDR+sg_Severe_NPDR+sg_Proliferative_DR)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

y_pos = np.arange(len(severity_grading))
performance = [sg_normal, sg_Mild_NDPR, sg_Moderate_NPDR, sg_Severe_NPDR, sg_Proliferative_DR]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.ylabel('Cases')
plt.title('DR Severity')

plt.show()

#Number of times image augmentation has to be done.
#Since Moderate severity has highest
diff_Mild = sg_Moderate_NPDR-sg_Mild_NDPR
diff_Severe = sg_Moderate_NPDR-sg_Severe_NPDR
diff_Proliferate = sg_Moderate_NPDR-sg_Proliferative_DR

#Calculating the number of iterations required for image augmentation 
Normal_itr = round((1500-sg_normal) / sg_normal)
Mild_itr = round((1500-sg_Mild_NDPR)/sg_Mild_NDPR)
Moderate_itr = round((1500-sg_Moderate_NPDR)/sg_Moderate_NPDR)
Severe_itr = round((1500-sg_Severe_NPDR)/sg_Severe_NPDR)
Proliferate_itr = round((1500-sg_Proliferative_DR)/sg_Proliferative_DR)

Grad = list()
Grad = ['0/','1/','2/','3/','4/']

import glob
import shutil
import os

for i in Grad:
    os.mkdir(file_repository+i)
    
for index, row in grad.iterrows():
    
    #Splitting the images as per the class names.
    if(row["severity"] == 'Normal'):
        src_dir=path_OI+'a. Training Set/'+row["Image name"]+'.jpg'
        dst_dir = file_repository+'0'
        shutil.copy(src_dir,dst_dir) 
                
    elif(row["severity"]=='Mild NPDR'):
        src_dir=path_OI+'a. Training Set/'+row["Image name"]+'.jpg'
        dst_dir = file_repository+'1'
        shutil.copy(src_dir,dst_dir) 
     
    elif(row["severity"]=='Moderate NPDR'):
        src_dir=path_OI+'a. Training Set/'+row["Image name"]+'.jpg'
        dst_dir = file_repository+'2'
        shutil.copy(src_dir,dst_dir) 
            
    elif(row["severity"]=='Severe NPDR'):
        src_dir=path_OI+'a. Training Set/'+row["Image name"]+'.jpg'
        dst_dir = file_repository+'3'
        shutil.copy(src_dir,dst_dir) 

    elif(row["severity"]=='Proliferative DR'):
        src_dir=path_OI+'a. Training Set/'+row["Image name"]+'.jpg'
        dst_dir = file_repository+'4'
        shutil.copy(src_dir,dst_dir)  

itr=0
Grad_aug = Grad
for i in Grad_aug:
    path=file_repository+i
    images = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.jpg']    
    
    if(i=='0/'):
        itr=Normal_itr
    elif(i=='1/'):
        itr=Mild_itr
    elif(i=='2/'):
        itr=Moderate_itr
    elif(i=='3/'):
        itr=Severe_itr
    elif(i=='4/'):
        itr=Proliferate_itr
        
    for image in images:
        img = load_img(file_repository+i+image)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(zca_whitening=True,zoom_range=0.3,fill_mode='nearest',
                                     horizontal_flip=False,vertical_flip=True)
        it = datagen.flow(samples, batch_size=32, save_to_dir=file_repository+i ,save_prefix='AUG'+image, save_format='jpg')
        for j in range(itr):
            batch = it.next()

newdf=pd.DataFrame()
number=0
img_finaldf=pd.DataFrame()
for i in Grad:
    path=file_repository+i
    images = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.jpg']
    img_df = pd.DataFrame(images)
    img_df.columns=['Image name']
    lab=number
    number=number+1
    label=list()
    for i in range(len(img_df)):
        label.append(lab)
    img_df.insert(1,"Retinopathy grade",label,True) 
    
    img_finaldf = img_finaldf.append(img_df)
    
img_finaldf = img_finaldf.sample(frac=1)    
    
grad_csv = img_finaldf.to_csv (file_repository+'image.csv', index = None, header=True)  

 


os.mkdir('Training_All/')
for i in Grad:
    
    src_dir=file_repository+i
    dst_dir = file_repository+'Training_All/'
    images = [f for f in os.listdir(src_dir) if os.path.splitext(f)[-1] == '.jpg']    
    for image in images:
        shutil.copy(src_dir+image,dst_dir)     

  
    

    
           
