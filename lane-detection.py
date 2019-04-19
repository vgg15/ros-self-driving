import cv2
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler 
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import minmax_scale

import keras
from keras.layers import Dense
from keras.models import Sequential

import pandas as pd

#video = cv2.VideoCapture("road_car_view.mp4")

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
DATASET_FOLDER = 'driving_dataset/driving_dataset/'
DATASET_FILENAME = DATASET_FOLDER  + 'data.txt'
ANGLE_RESOLUTION = 0.1
ANGLE_STEPS = int(2/ANGLE_RESOLUTION)+1

x = np.zeros((1,CAMERA_WIDTH*CAMERA_HEIGHT))
y = np.zeros(1)

# Import data
print("> Importing data from " + DATASET_FILENAME)
f = open(DATASET_FILENAME)
i = 0
for line in f:
    # open image in b&w
    row = line.split()
    img = str(row[0])
    angle = float(row[1])    
    frame = cv2.imread(DATASET_FOLDER+img, cv2.COLOR_BGR2GRAY)
    
    # transform frame to b&w
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # resize frame
    frame = cv2.resize(frame,(CAMERA_WIDTH,CAMERA_HEIGHT))

    # unroll
    frame = np.reshape(frame, (1,CAMERA_WIDTH*CAMERA_HEIGHT))

    x = np.append(x,frame,axis=0)
    y = np.append(y,[angle])
    i =i+1
    if i==200:
        break

        
# Prepare data
print("> Prepare data")
x = StandardScaler().fit_transform(x)
y_norm = MaxAbsScaler().fit_transform((StandardScaler().fit_transform(y.reshape(-1, 1))))
y = np.zeros((y_norm.shape[0], ANGLE_STEPS))    
for i,angle in enumerate(y_norm):
    idx = (ANGLE_STEPS-1)/2*(angle)+(ANGLE_STEPS-1)/2
    y[i,math.ceil(idx)] = 1

print("x shape:" + str(np.shape(x)))
print("x sample: ")
print(np.random.permutation(x)[1:5,:])

print("y shape:" + str(np.shape(y)))

print(y_norm[1:5])
print(y[1:5,:])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .1)

# build simple NN
print("> Build Network")
input_units = x_train.shape[1]
inner_units = int(input_units/10)
output_units = ANGLE_STEPS # angle ranges from -1 to 1 in steps of 0.1

model = Sequential()
model.add(Dense(units=inner_units, activation='relu', input_dim=input_units))
model.add(Dense(units=inner_units, activation='relu'))
model.add(Dense(units=output_units, activation='softmax'))

print("> Compile and fit the Network")
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 10, validation_split = .1)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)
#predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)

#key = cv2.waitKey(25)
#if key == 27:
#    break
#video.release()
#cv2.destroyAllWindows()
