# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:33:39 2019

@author: JIGAR'S PC
"""

import cv2
import numpy as np
import math
from classifier import Classifier
import os
SIGNS=['one-way traffic','No vehicles in both directions',
       'All motor vehicles prohibited','No motorcycles',
       'No bicycles','No heavy vehicles','No bullock carts',
       'No pedestrians','No left turn','No right turn',
       'no U-turn','No overtaking','speed limit: 40',
       'Horn prohibited','No parking','No stopping',
       'No straight ahead','One-way traffic','speed limit: 20',
       'speed limit: 30','speed limit: 50','speed limit: 60','speed limit: 70',
       'speed limit: 80','no restrictions','unknown sign']


THRESH=0.65
NUM_CLASSES=25

def clean_images():
	file_list = os.listdir('./')
	for file_name in file_list:
		if '.png' in file_name:
			os.remove(file_name)
def equalize_hist(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def filter_red_color(hsv):
    
    lower_red_mask = cv2.inRange(hsv, np.array([0, 100, 65]), np.array([10, 255, 255]))
    upper_red_mask = cv2.inRange(hsv, np.array([155, 100, 70]), np.array([179, 255, 255]))
    mask = lower_red_mask + upper_red_mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)))
    mask = cv2.Canny(mask, 50, 100)
    mask = cv2.GaussianBlur(mask, (13, 13), 0)
    return mask

def rectangulate(circle):
    x, y, r = circle
    rn = r - 5
    rect = [((x - rn), (y - rn)), ((x + rn), (y + rn))]
    return rect

def crop_sign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    #print(top,left,bottom,right)
    return image[top:bottom,left:right]

def find_signs(original_image, classifier, model, count, current_sign_type):
    image=equalize_hist(original_image)
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (7, 7), 0), cv2.COLOR_BGR2HSV)
    mask = filter_red_color(hsv)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 100, param1=30, param2=50)
#    print(circles)
    sign=None
    text = ""
    sign_type = -1
    rect=None
    
    if circles is not None :
        
        circles = np.round(circles[0, :]).astype("int")
        if circles[0][2] != 0:
            rect=((rectangulate(circles[0])))           
            sign = crop_sign(original_image,rect)
    if sign is not None:
        signcpy=cv2.resize(sign,(64,64))
        signcpy=cv2.cvtColor(signcpy,cv2.COLOR_BGR2GRAY)
        signcpy.resize(64*64,1)
        sign_oh = classifier.predict(model,signcpy)
        print(sign_oh)
        if np.max(sign_oh)<THRESH:
            sign_type=NUM_CLASSES
        else:
            sign_type=np.argmax(sign_oh)
                                                                                
        text = SIGNS[sign_type]
        #cv2.imwrite(str(count)+'_'+text+'.png', sign)

    if sign_type > 0 and sign_type != current_sign_type:        
        cv2.rectangle(original_image, rect[0],rect[1], (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(original_image,text,(rect[0][0], rect[0][1] -15), font, 2,(0,0,255),2,cv2.LINE_4)
    return rect, original_image, sign_type, text


if __name__=="__main__":
    
    clean_images()
    cl=Classifier((64,64,1))
    
    model=cl.load(model_path="models\\traffic-sign-IN-keras-cropped_v1.1.h5")
    file_name="samples\\no-straight-ahead-no-straight-ahead-sign-jungle-133602726.jpg"
    
    is_image=False
    file_type=file_name.split(".")[-1]
    if file_type=="jpg":
        is_image=True
    vidcap = cv2.VideoCapture(file_name)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(3)  
    height = vidcap.get(4) 
    if not is_image:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, fps , (640,480))
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    roiHist = None

    success = True
    similitary_contour_with_circle = 0.65
    count = 0
    current_sign = None
    current_text = ""
    current_size = 0
    sign_count = 0
    coordinates = []
    position = []
    file = open("Output.txt", "w")
    
    while True:
        success,frame = vidcap.read()
        if not success:
            print("FINISHED")
            break
        width = frame.shape[1]
        height = frame.shape[0]
       
        print("Frame:{}".format(count))
        coordinate, image, sign_type, text = find_signs(frame, cl , model, count, current_sign)
        if coordinate is not None:
            cv2.rectangle(image, coordinate[0],coordinate[1], (255, 255, 255), 1)
        #print("Sign:{}".format(sign_type))
        if sign_type > 0 and (not current_sign or sign_type != current_sign):
            current_sign = sign_type
            current_text = text
            top = int(coordinate[0][1]*1.05)
            left = int(coordinate[0][0]*1.05)
            bottom = int(coordinate[1][1]*0.95)
            right = int(coordinate[1][0]*0.95)

            position = [count, sign_type if sign_type <= 43 else 43, coordinate[0][0], coordinate[0][1], coordinate[1][0], coordinate[1][1]]
            cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(image,text,(coordinate[0][0], coordinate[0][1] -15), font, 2,(0,0,255),2,cv2.LINE_4)

            tl = [left, top]
            br = [right,bottom]
            current_size = math.sqrt(math.pow((tl[0]-br[0]),2) + math.pow((tl[1]-br[1]),2))

            roi = frame[tl[1]:br[1], tl[0]:br[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
            roiBox = (tl[0], tl[1], br[0], br[1])

        elif current_sign:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
            
            (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
            pts = np.int0(cv2.boxPoints(r))
            print("Frame:{0} pts:{1}".format(count,pts))
            s = pts.sum(axis = 1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            size = math.sqrt(pow((tl[0]-br[0]),2) +pow((tl[1]-br[1]),2))
            #print(size)

            if  current_size < 1 or size < 1 or size / current_size > 30 or math.fabs((tl[0]-br[0])/(tl[1]-br[1])) > 2 or math.fabs((tl[0]-br[0])/(tl[1]-br[1])) < 0.5:
                current_sign = None
                print("Stop tracking")
            else:
                current_size = size

            if sign_type > 0:
                top = int(coordinate[0][1])
                left = int(coordinate[0][0])
                bottom = int(coordinate[1][1])
                right = int(coordinate[1][0])

                position = [count, sign_type if sign_type <= 43 else 43, left, top, right, bottom]
                cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 0), 1)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
            elif current_sign:
                position = [count, sign_type if sign_type <= 43 else 43, tl[0], tl[1], br[0], br[1]]
                cv2.rectangle(image, (tl[0], tl[1]),(br[0], br[1]), (0, 255, 0), 1)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(image,current_text,(tl[0], tl[1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
                
        if current_sign:
            sign_count += 1
            coordinates.append(position)

        cv2.imshow('Result', image)
        count = count + 1
        cv2.imwrite("fig1.jpg",image)
        image=cv2.resize(image,(640,480))
        if not  is_image:
            out.write(image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    file.write("{}".format(sign_count))
    for pos in coordinates:
        file.write("\n{} {} {} {} {} {}".format(pos[0],pos[1],pos[2],pos[3],pos[4], pos[5]))
    print("Finish {} frames".format(count))
    file.close()
    
    if not is_image:
        vidcap.release()
        cv2.destroyAllWindows()     
