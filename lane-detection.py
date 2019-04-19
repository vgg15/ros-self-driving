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
import os
from threading import Thread

DATA_X_FILENAME = 'data_x'
DATA_Y_FILENAME = 'data_y'

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240


DATASET_FOLDER = 'driving_dataset/driving_dataset/'
DATASET_FILENAME = DATASET_FOLDER  + 'data.txt'
MODEL_FILENAME = 'model.h5'

ANGLE_RESOLUTION = 0.1

class ImportThread (Thread):
   def __init__(self, id, linelist):
      Thread.__init__(self)
      self.linelist = linelist
      self.id = id
        
   def run(self):
        x = np.zeros((1,CAMERA_WIDTH*CAMERA_HEIGHT))
        y = np.zeros(1)
        i=0
        listlen = len(self.linelist)         
        for line in self.linelist:
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
            if self.id == 0:
                print("Progress: {0:.2f} %".format(float(i)/listlen*100), end='\r')
            #if i==20:
            #    break
        np.save(DATA_X_FILENAME+str(self.id), x)
        np.save(DATA_Y_FILENAME+str(self.id), y)

#video = cv2.VideoCapture("road_car_view.mp4")

ANGLE_STEPS = int(2/ANGLE_RESOLUTION)+1



# Import data
print("> Importing data from " + DATASET_FILENAME)
#f = open(DATASET_FILENAME)
i = 0

if os.path.isfile(DATA_X_FILENAME+'.npy') and os.path.isfile(DATA_Y_FILENAME+'.npy'):    
    """with open(DATA_X_FILENAME) as f:
        xlist = f.readlines()
        x = np.array(xlist)
    with open(DATA_Y_FILENAME) as f:
        ylist = f.readlines()
        y = np.array(ylist)"""
    print("> Found data files..")
    x= np.load(DATA_X_FILENAME+'.npy')
    y=np.load(DATA_Y_FILENAME+'.npy')
else:         
    print("> Generating data files..")

    with open(DATASET_FILENAME) as f:
        lineList = f.readlines()

    listlen = len(lineList)
    
    print("Dataset has " + str(listlen)+ " samples")
    threadnum = 12
    chunks = np.array_split(np.array(lineList),threadnum)
    threads = []
    for i in range(threadnum):
        threads.append(ImportThread(i, chunks[i].tolist()))
    
    for t in threads:
        t.start()
    

    for t in threads:
        t.join()


    """f = open(DATA_X_FILENAME, "w")
    for i in range(x.shape[0]):
        f.write(str(np.array2string(x[i,:], separator=' ', threshold = CAMERA_WIDTH*CAMERA_HEIGHT*10))+'\n')
    f.close
    f = open(DATA_Y_FILENAME, "w")
    for i in range(y.shape[0]):
        f.write(str(y[i]) + '\n')
    f.close"""
    

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
print(np.random.permutation(y)[1:5,:])

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
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 10, validation_split = .1)
model.save(MODEL_FILENAME)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)
#predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)

#key = cv2.waitKey(25)
#if key == 27:
#    break
#video.release()
#cv2.destroyAllWindows()
