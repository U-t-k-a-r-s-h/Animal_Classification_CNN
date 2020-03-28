from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import cv2

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint
DATADIR = "Directory for testing data which contains followinng subfolders having corresponding images"
CATAGORIES = ["Cats","Dogs","Elephants","Horse"]
IMG_SIZE = 90
testing_data=[]
def create_testing_data():
    for category in CATAGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATAGORIES.index(category)
        print("class=",class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                testing_data.append([new_array,class_num])
            except Exception as e :
                pass
create_testing_data()
import random
random.shuffle(testing_data)

x=[]
y=[]
for features,label in testing_data:
    x.append(features)
    y.append(label)
x = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y = np.array(y)

print("done till here")

x=x/255
print(x.shape[1:])

model = Sequential()
model.add(Conv2D(70,(3,3), input_shape = x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(40,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(65))
model.add(Dense(4))
model.add(Activation("softmax"))
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

checkpoint_path = "Directory where the already trained model is stored"
model.load_weights(checkpoint_path)
loss,accuracy = model.evaluate(x,y)
print(loss,accuracy)