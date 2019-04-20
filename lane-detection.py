import cv2
import numpy as np
import math
import time
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler 
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import minmax_scale

"""
import keras
from keras.layers import Dense
from keras.models import Sequential
"""
import os
from os import system, name 
from threading import Thread
from queue import Queue
import collections

DATA_X_FILENAME = 'data_x'
DATA_Y_FILENAME = 'data_y'

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

NUM_THREADS = 4
NUM_BATCH = 16

DATASET_FOLDER = 'driving_dataset/driving_dataset/'
DATASET_FILENAME = DATASET_FOLDER  + 'data.txt'
MODEL_FILENAME = 'model.h5'

ANGLE_RESOLUTION = 0.1
ANGLE_STEPS = int(2/ANGLE_RESOLUTION)+1

def print_progress(progress, curr_epoch, tot_epoch, eta):
    totpercent = 0.0
    for id, percent in progress.items():
        totpercent = totpercent + percent

    totpercent = int(totpercent/NUM_THREADS)
    bar = ('=' * int(totpercent/5)).ljust(20)        
    sys.stdout.write("\rEpoch %s/%s, Total progress [%s] %s%%, elapsed: %s" % (curr_epoch, tot_epoch, bar, totpercent, eta))    
    sys.stdout.flush()

class ImportThread (Thread):
    def __init__(self, id, linelist, status):
        Thread.__init__(self)
        self.linelist = linelist
        self.id = id
        self.status = status
        self.xlist = []
        self.ylist = []

    def run(self):        
        i=0
        listlen = len(self.linelist)          
        #print("Thread " + str(self.id) + " started with " + str(listlen) + " elements")                        

        for line in self.linelist:            
            row = line.split()
            img = str(row[0])
            angle = float(row[1])    
            
            # open image in b&w
            frame = cv2.imread(DATASET_FOLDER+img, cv2.IMREAD_GRAYSCALE)

            # resize frame
            frame = cv2.resize(frame,(CAMERA_WIDTH,CAMERA_HEIGHT))
            
            # unroll
            frame = np.reshape(frame, (1,CAMERA_WIDTH*CAMERA_HEIGHT))
            
            self.xlist.append(frame[0,:])
            self.ylist.append(angle)

            i = i + 1    
            progress = int(float(i)/listlen*100)
            self.status.put([self.id, progress])
            time.sleep(0.01)
    
    def save(self):       
        #print("Thread " + str(self.id) + " saving data...")               
        np.save(DATA_X_FILENAME+"_"+str(self.id), np.around(np.array(self.xlist), decimals=4))
        np.save(DATA_Y_FILENAME+"_"+str(self.id), np.around(np.array(self.ylist), decimals=4))

def KerasGenerator(x,y, bs):
    i=0
    totlen = len(x)
    while True:
        img = []
        labels = []
        while len(img) < bs:
            img.append(x[i,:]/255.0)
            labels.append(y[i,:])
            i = i + 1
            if i >= totlen:
                i=0
        yield (img, labels)

def main():
    # Import data
    print("> Importing data from " + DATASET_FILENAME)
    
    if os.path.isfile(DATA_X_FILENAME+'.npy') and os.path.isfile(DATA_Y_FILENAME+'.npy'):    
        print("> Importing existing data files...")
        x = np.load(DATA_X_FILENAME+'.npy')
        y = np.load(DATA_Y_FILENAME+'.npy') 
    else:         
        print("> Generating new data files..")
        millis = int(round(time.time() * 1000))
        with open(DATASET_FILENAME) as f:
            lineList = f.readlines()

        listlen = len(lineList)
        
        print("Dataset has " + str(listlen)+ " samples")
        status = Queue()
        progress = collections.OrderedDict()
        
        chunks = np.array_split(np.array(lineList), NUM_BATCH)
        

        i=0
        epoch = 1
        tot_epoch = int(NUM_BATCH/NUM_THREADS)
        while (i < len(chunks)):
            #print("Run Epoch " + str(epoch) + "/" + str(len(chunks)/NUM_THREADS), end='') 
            threads = []
            for j in range(NUM_THREADS):
                t = ImportThread(i, chunks[i].tolist(), status)
                threads.append(t)
                progress[i] = 0.0
                t.daemon = True
                t.start()
                i=i+1

            while any(i.is_alive() for i in threads):
                time.sleep(1)
                while not status.empty():
                    id, percent = status.get()
                    progress[id] = percent/tot_epoch
                    millis2 = int(round(time.time() * 1000))
                    print_progress(progress, epoch, tot_epoch, (millis2-millis)/1000)
            for t in threads:
                t.save()

            for t in threads:
                t.join()
            
            epoch = epoch + 1

        print("")               

        millis2 = int(round(time.time() * 1000))                
        print("> Generation done in %s seconds" % str((millis2-millis)/1000))

        exit()
    print("x shape:" + str(np.shape(x)))
    print("x samples: ")
    print(np.random.permutation(x)[1:5,:])

    print("y shape:" + str(np.shape(y)))
    print("y samples: ")
    print(np.random.permutation(y)[1:5])
    
    # Prepare data
    print("> Preparing data")
    #x = StandardScaler().fit_transform(x)
    y_norm = MaxAbsScaler().fit_transform((StandardScaler().fit_transform(y.reshape(-1, 1))))
    y = np.zeros((y_norm.shape[0], ANGLE_STEPS))    
    for i,angle in enumerate(y_norm):
        idx = (ANGLE_STEPS-1)/2*(angle)+(ANGLE_STEPS-1)/2
        y[i,math.ceil(idx)] = 1

    print("x shape:" + str(np.shape(x)))
    print("x samples: ")
    print(np.random.permutation(x)[1:5,:])

    print("y shape:" + str(np.shape(y)))
    print("y samples: ")
    print(np.random.permutation(y)[1:5]) 

    #print("> Saving normalized data...")
    #np.save(DATA_X_FILENAME+"_norm", np.around(x, decimals=4))
    #np.save(DATA_Y_FILENAME+"_norm", np.around(y, decimals=4))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = .2)

    # build simple NN
    print("> Build Network")
    input_units = x_train.shape[1]
    inner_units = int(input_units/10)
    output_units = ANGLE_STEPS # angle ranges from -1 to 1 in steps of 0.1

    """
    model = Sequential()
    model.add(Dense(units=inner_units, activation='relu', input_dim=input_units))
    model.add(Dense(units=inner_units, activation='relu'))
    model.add(Dense(units=output_units, activation='softmax'))

    print("> Compile and fit the Network")
    BATCH_SIZE = 64
    kg_train = KerasGenerator(x_train, y_train, bs)
    kg_val = KerasGenerator(x_val,y_val, bs)
    
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    model.fit_generator( kg_train, steps_per_epoch=math.ceil(len(x_train) / BATCH_SIZE),
        validation_data=kg_val,
	    validation_steps=math.ceil(len(x_train) / BATCH_SIZE),
	    epochs=5)        

    model.save(MODEL_FILENAME)

    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    print(loss_and_metrics)
    """

if __name__ == "__main__":
    main()
