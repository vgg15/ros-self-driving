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
import glob
import glob

OUTPUT_DIR = 'output/'
DATA_X_FILENAME = OUTPUT_DIR + 'data_x'
DATA_Y_FILENAME = OUTPUT_DIR + 'data_y'
NUMPY_EXT = '.npy'

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

def BatchGenerator(batch_x, batch_y):
    i=0
    batch_len = len(batch_x)
    while True:
        img = np.load(batch_x[i])
        img = img/255.0
        labels = np.load(batch_y[i])
        i = i + 1
        if i >= batch_len:
            i=0
        yield (img, labels)

def main():
    # Import data
    print("> Importing data from " + DATASET_FILENAME)
    
    if os.path.isdir(OUTPUT_DIR):
        print("> Output folder already exists, skipping...")
    else:   
        os.mkdir(OUTPUT_DIR)
        print("> Generating new data files..")
        millis = int(round(time.time() * 1000))
        with open(DATASET_FILENAME) as f:
            linelist = f.readlines()

        listlen = len(linelist)
        
        print("Dataset has " + str(listlen)+ " samples")
        status = Queue()
        progress = collections.OrderedDict()
        
        x = []
        y = []
        for line in linelist:
            x.append(line.split()[0])
            y.append(line.split()[1])
        
        y = np.array(y)
        y_norm = MaxAbsScaler().fit_transform((StandardScaler().fit_transform(y.reshape(-1, 1))))        

        y = y_norm.tolist()

        dataset = []        
        for i in range(len(x)):
            dataset.append(str(x[i])+ " " + str(y[i]).replace('[','').replace(']',''))
        
        dataset = np.random.permutation(np.array(dataset))
        print(dataset[1:5])        
        chunks = np.array_split(dataset, NUM_BATCH)
        
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
    
    # Prepare data
    x_batch_list = sorted(glob.glob(DATA_X_FILENAME+'*'))
    y_batch_list = sorted(glob.glob(DATA_Y_FILENAME+'*'))

    if ((len(x_batch_list) != NUM_BATCH) and (len(y_batch_list) != NUM_BATCH)):
        print("> Some data elements are missing. exiting...")
        exit()

    #for i,angle in enumerate(y_norm):
    #    idx = (ANGLE_STEPS-1)/2*(angle)+(ANGLE_STEPS-1)/2
    #    y[i,math.ceil(idx)] = 1

    x_train, x_test, y_train, y_test = train_test_split(x_batch_list, y_batch_list, test_size = .1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = .1)

    # build simple NN
    input_units = CAMERA_WIDTH*CAMERA_HEIGHT
    inner_units = int(input_units/10)
    output_units = ANGLE_STEPS # angle ranges from -1 to 1 in steps of 0.1

    bg_train = BatchGenerator(x_train, y_train)
    bg_val = BatchGenerator(x_val, y_val)
    
    """
    print("> Build Network")
    model = Sequential()
    model.add(Dense(units=inner_units, activation='relu', input_dim=input_units))
    model.add(Dense(units=inner_units, activation='relu'))
    model.add(Dense(units=output_units, activation='softmax'))

    print("> Compile and fit the Network")
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    model.fit_generator( bg_train, steps_per_epoch=math.ceil(len(x_train) / BATCH_SIZE),
        validation_data=bg_val,
	    validation_steps=math.ceil(len(x_train) / BATCH_SIZE),
	    epochs=5)        

    model.save(MODEL_FILENAME)

    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    print(loss_and_metrics)
    """

if __name__ == "__main__":
    main()
