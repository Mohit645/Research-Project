# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:57:40 2019

@author: Mohit
"""

#Preprocessing For severity grading
import os
import shutil
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix


path ="C:/Users/Mohit/OneDrive/Documents/Grading/"
os.chdir(path)
path_OI =path+"1. Original Images/"
path_gt=path+"2. Groundtruths/"


#url = https://www.programcreek.com/python/example/570/Image.ANTIALIAS
set1=['a. Training Set/', 'b. Testing Set/']
for j in set1:
    print("Resizing "+ j)   
        
    images = [f for f in os.listdir(path_OI+j) if os.path.splitext(f)[-1] == '.jpg'] 
    for i in images:    
        img = Image.open(path_OI+j+i)
        print(i)
        img = img.resize((384,256), Image.ANTIALIAS)
        img.save(path_OI+j+i)
        
#Subtracting the local average colour        
# Blur the image
for j in set1:
    
    images = [f for f in os.listdir(path_OI+j) if os.path.splitext(f)[-1] == '.jpg']
    for i in images:
            
        #img = Image.open(path_OI+j+i)     
        img = cv2.imread(path_OI+j+i)   
        blurred = cv2.blur(cv2.UMat(img), ksize=(15, 15))        
        dst = cv2.addWeighted(img, 4, blurred, -4, 128)
        img = cv2.UMat.get(dst)
        img = Image.fromarray(img)
        img.save(path_OI+j+i)
        
        

#Trimming the black border
        
for j in set1:
    
    images = [f for f in os.listdir(path_OI+j) if os.path.splitext(f)[-1] == '.jpg']
    for i in images:
        
        img = cv2.imread(path_OI+j+i)
        y=0
        x=20
        h=256
        w=315
        crop = img[y:y+h, x:x+w]
        img = Image.fromarray(crop)
        img.save(path_OI+j+i)
        

#Convert the images to 50 percent gray
'''for j in set1:
    
    images = [f for f in os.listdir(path_OI+j) if os.path.splitext(f)[-1] == '.jpg']
    for i in images:
        img = cv2.imread(path_OI+j+i)
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path_OI+j+i, image_gray)'''


  
            
        

