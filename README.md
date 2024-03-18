# Traffic-sign-detector-and-classifier

Detect and classifies traffic signs in real time. Engineered to warn drivers about the restrictions of road such as speed limit, no overtaking, allowed vehicle types, max vehicle height, etc., and thus potentialy reduce accidents.

## Output:
<p align="center">
  <img src="https://github.com/drone911/traffic-sign-detector-and-classifier/blob/master/sample_outputs/output_40_speed_limit.gif?raw=true" />
</p>
<p align="center">
  <img src="https://github.com/drone911/traffic-sign-detector-and-classifier/blob/master/sample_outputs/fig1.PNG" />
</p>

Figure) Speed limit of 40 being detected and classified correctly from input https://github.com/drone911/traffic-sign-detector-and-classifier/tree/master/samples/40_speed_limit.mp4

## Prerequisites:

1) Python 3.7

2) OpenCV 3.4.1

3) Tensorflow 1.13.1

## Project Structure:

1) there are 4 python files:
  
  	1.1) main.py: Entry point of program. 
	
  	1.2) classifier.py: describes the classifier. 
	
  	1.3) extractor.py: describes function to return images and their respective labels.	 
	
  	1.4) training.py: trains a Convolutional Neural Network. 
	
2) the dataset is present in "Final_Training_IN". Each class has a number associated to it which is also its directory name and used to uniquely identify that class throughout the project. First 18 classes have images of type jpg and the rest of them as type ppm.  

3) directory "models" contain various pre-trained Convolutional Neural Networks.
	
4) directory "samples" contain the sample inputs.
