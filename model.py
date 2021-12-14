'''
Skin Cancer Prediction Model

Two datasets - labeled train and test with subfolders of malignant and benign tumors

: 
'''



#base imports 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import splitfolders

#deep learning imports
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import Sequential
from keras.layers import Flatten, Dense

print('starting process...')

#train-test-val data split
train_dir = "CancerDataset/output/train"
test_dir = "CancerDataset/output/test"
val_dir = "CancerDataset/output/val"

#data augmentation; generates batches of tensor image data with real-time augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2,shear_range=0.2)
train_generator= train_datagen.flow_from_directory(directory=train_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

val_datagen = ImageDataGenerator(rescale=1.0/255)
val_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

print('data augmentation complete...')

#model 1 - vgg19; 80.85% accuract with a loss of 0.504
vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(128,128,3))

for layer in vgg19.layers:
    layer.trainable = False
    
model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(2,activation='sigmoid'))
model.summary()

#compile model
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics ="accuracy")

print('model compiled successfully...')

#train model
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=len(train_generator)//32,
                              epochs=20,validation_data=val_generator,
                              validation_steps=len(val_generator)//32)

#evaluate model
scores = model.evaluate(test_generator)
#print off evaluation results
print(scores)