# traffic-sign-detector-and-classifier

Traffic Sign Detector and Classifiy can detect and classify traffic signs in real time. If the output of the dashboard camera of a vehicle be given as an input to it then it will be able to warn the driver about the restrictions of road such as speed limit, no overtaking, allowed vehicle types etc. This results in better following of road safety rules and thus reduction in vehicle related accidents.

Project Structure:

a) there are 4 python files:
  1) main.py: Entry point of program. 
	
  2) classifier.py: describes the classifier. 
	
  3) extractor.py: describes function to return images and their respective labels.	 
	
  4) training.py: trains a Convolutional Neural Network. 
	
b) the dataset is present in Final_Training_IN. Each class has a number associated to it which is also its directory name and used to uniquely identify that class throughout the project. First 18 classes have images of type jpg and the rest of them as type ppm.  

c) directory models contain various pre-trained Convolutional Neural Networks.
	
d) directory samples contain the sample inputs.

Output:

![alt text](https://github.com/drone911/traffic-sign-detector-and-classifier/blob/master/ouput_figures/fig1.PNG)

fig 5.1) speed limit of 40 being detected and classified correctly from input https://github.com/drone911/traffic-sign-detector-and-classifier/tree/master/samples/40_speed_limit.mp4
                                     
![alt text](https://github.com/drone911/traffic-sign-detector-and-classifier/blob/master/ouput_figures/fig2.jpg)
  
fig 5.2) No straight ahead sign being detected and classified correctly from input https://github.com/drone911/traffic-sign-detector-and-classifier/tree/master/samples/no-straight-ahead.PNG
