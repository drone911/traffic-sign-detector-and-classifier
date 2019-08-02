# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:11:50 2019

@author: JIGAR'S PC
"""
import cv2
import pandas
import glob
import random
def get_training_data(training_path,lose,img_size,unchanged=False,crop=False,num=0):
    train_dirs=glob.os.listdir(training_path)
    for cnt,train_dir in enumerate(train_dirs):
        int_train_dir=(int(train_dir))
        train_dirs[cnt]=int_train_dir
    train_dirs=sorted(train_dirs)
    images=[]
    labels=[]
    for count,d in enumerate(train_dirs):
        if d in lose:
            pass
        if count>17:
            files=glob.glob(training_path+"\\"+str(d)+"\\*.ppm")
        else:
            files=glob.glob(training_path+"\\"+str(d)+"\\*")
            
        if num!=0:
            files=random.sample(files,num)
        if crop and count>17:
            dataframes=glob.glob(training_path+"\\"+str(d)+"\\*.csv")
            df_minor=pandas.read_csv(dataframes[0],index_col=0,sep=";")
        for f in files:
            if unchanged:
                img=cv2.imread(f,cv2.IMREAD_UNCHANGED)
            else:
                img=cv2.imread(f,0)
            if crop and count>21:
                dim=df_minor.loc[f.split('\\')[-1],:]
                img=img[dim[3]:dim[5],dim[2]:dim[4]]
            img=cv2.resize(img,img_size)
            
            images.append(img)
            labels.append(count)
    return images,labels

    
   