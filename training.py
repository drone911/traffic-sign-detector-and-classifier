# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 00:44:52 2019

@author: JIGAR'S PC
"""
import numpy as np
from extractor import get_training_data
import glob
import sklearn.preprocessing as prepro
import cv2
import math
from classifier import Classifier

training_path=glob.os.getcwd()+"\Final_Training_IN"
model_path=glob.os.getcwd()+"\models\traffic-sign-keras-cropped_v1.1.h5"

images,labels=get_training_data(training_path,[],(64,64),unchanged=False,crop=False,num=200)

labels=np.array(labels)
labels_oh=prepro.OneHotEncoder().fit_transform(labels.reshape(-1,1)).toarray() 

comb=list(zip(images,labels_oh))
np.random.shuffle(comb)
images,labels_oh=zip(*comb)

images=np.array(images)
labels_oh=np.array(labels_oh)
images=images.reshape(-1,64,64,1)
for count,image in enumerate(images):
    image_norm=np.zeros_like(image,dtype=np.float32)
    images_norm=cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    images[count]=image_norm
    
train_images=images[0:math.floor(0.8*len(labels))]
train_labels_oh=labels_oh[0:math.floor(0.8*len(labels))]
val_images=images[math.floor(0.8*len(labels)):]
val_labels_oh=labels_oh[math.floor(0.8*len(labels)):]

cl=Classifier((64,64,1))
model=cl.train(train_images,train_labels_oh,val_images,val_labels_oh,num_classes=25)
cl.evalaute(model,val_images,val_labels_oh)
cl.save(model,model_path)

