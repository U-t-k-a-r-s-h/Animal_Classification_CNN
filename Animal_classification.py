from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import cv2

import tensorflow as tf 
from tensorflow import keras
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

DATADIR = "Directory to your training dataset" 
CATAGORIES = ["Cats","Dogs","Elephants","Horse"]	#Subfolders within the above given directory where each folder contains corresponding images
IMG_SIZE = 90
training_data=[]
def create_training_data():
    for category in CATAGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATAGORIES.index(category)
        print("class=",class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e :
                pass
create_training_data()
import random
random.shuffle(training_data)   #the data needs to be shuffled in order to train perfectly

x=[]
y=[]

for features,label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)     #this contains the input images appended in an array
y = np.array(y)                                     #this contains the output labels appended in an array

x=x/255
print("modelling started")

model = Sequential()
model.add(Conv2D(70,(3,3), input_shape = x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(60,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(65))
model.add(Dense(4))
model.add(Activation("softmax"))
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

checkpoint_path = "Enter a directory where you want to save the model for further use"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
model.fit(x,y,epochs=7,use_multiprocessing=True,batch_size=64,callbacks=[cp_callback])
