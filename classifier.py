import numpy as np
from tensorflow.contrib.keras.api.keras.models import Sequential,load_model
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.contrib.keras.api.keras.layers import Dropout, Flatten, Dense 
import cv2

class Classifier():
    def __init__(self,img_shape):
        self.img_shape=img_shape
        
    def inference(self):
        
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.img_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
          
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
          
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
          
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(25, activation='softmax'))   
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def save(self,model,model_path):
        model.save(model_path)
    
    def load(self,model_path):
        model=load_model(model_path)
        return model
    
    def train(self,train_images,train_labels_oh,val_images,val_labels_oh,batch_size=64,get_saved=False,save=False,model_path="model.h5",epochs=10,num_classes=25):
        if get_saved:
            model=self.load(model_path)
        else:
            model=self.inference()
        model.fit(train_images, train_labels_oh, batch_size=batch_size, epochs=epochs, verbose=1,
                   validation_data=(val_images, val_labels_oh))
        if save:
            self.save(model,model_path)
        return model
    
    def evalaute(self,model,test_images,test_labels_oh):
        model.evaluate(test_images,test_labels_oh)
    
    def predict(self,model,image):
        image=np.array(image)
        norm_image = np.zeros_like(image,dtype=np.float)
        norm_image=cv2.normalize(image, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_image=norm_image.reshape((1,self.img_shape[0],self.img_shape[1],self.img_shape[2]))
        label_oh=model.predict(norm_image)
        return label_oh