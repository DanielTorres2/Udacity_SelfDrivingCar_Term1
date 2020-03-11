#!/usr/bin/env python3
import csv
import cv2
import numpy as np
import sklearn
import os
import scipy.ndimage as ndimage
from sklearn.utils import shuffle

samples = []
with open('/opt/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Generates batches for training and validation
def generator(samples, batch_size):
    num_samples = len(samples)
    correction = 0.2 #Compensates for mounting of camera
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                
                #Obtain steering angle for all three cameras with angle compensation
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
               
                #Augmented data
                aug_steering_center = steering_center*-1.0
                aug_steering_left = steering_left*-1.0
                aug_steering_right = steering_right*-1.0
                
                #Images from all three cameras
                img_center = ndimage.imread(batch_sample[0].strip())
                img_left = ndimage.imread(batch_sample[1].strip())
                img_right = ndimage.imread(batch_sample[2].strip())
                
                #Augmented data
                aug_img_center = cv2.flip(img_center,1)
                aug_img_left = cv2.flip(img_left,1)
                aug_img_right = cv2.flip(img_right,1)
                
                images.extend([img_center, img_left, img_right, aug_img_center, aug_img_left, aug_img_right])
                measurements.extend([steering_center, steering_left, steering_right, aug_steering_center, aug_steering_left, aug_steering_right])
                
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 127.5 - 1.0))
#Nvidia
model.add(Convolution2D(24, 5 ,5, subsample=(2,2), activation ="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation ="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation= "relu"))
model.add(Convolution2D(64,3,3, activation= "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)
model.fit_generator(train_generator,
            steps_per_epoch=np.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=np.ceil(len(validation_samples)/batch_size),
            epochs=5, verbose=1)
model.save('model.h5')