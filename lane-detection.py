#!/home/alessandro/venv/bin/python3
import cv2
import numpy as np
import math
import time
import sys
import json

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler 
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score
from keras.utils import plot_model
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
from keras.callbacks import *

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import History
import os
from os import system, name 
from threading import Thread
from multiprocessing import Queue
import collections
import glob
import glob

OUTPUT_DIR = 'output/'
DATA_X_FILENAME = OUTPUT_DIR + 'data_x'
DATA_Y_FILENAME = OUTPUT_DIR + 'data_y'
NUMPY_EXT = '.npy'

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 320

NUM_THREADS = 4
NUM_BATCH = 16

DATASET_FOLDER = 'bag_output/'
DATASET_FILENAME = DATASET_FOLDER  + 'data.txt'

ANGLE_RESOLUTION = 0.1
ANGLE_STEPS = int(2/ANGLE_RESOLUTION)+1

# Global variable
DATASET_SIZE = 0

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
            frame = cv2.imread(DATASET_FOLDER+img)

            # resize frame
            frame = cv2.resize(frame,(CAMERA_WIDTH,CAMERA_HEIGHT))

            # apply threshold
            ret, frame = cv2.threshold(frame, 50,255, cv2.THRESH_BINARY)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pts1 = np.float32([[90,110],[CAMERA_WIDTH-90,110],[0,CAMERA_HEIGHT-60],[CAMERA_WIDTH,CAMERA_HEIGHT-60]])
            pts2 = np.float32([[0,0],[CAMERA_WIDTH,0],[0,CAMERA_HEIGHT],[CAMERA_WIDTH,CAMERA_HEIGHT]])
    
    
            M = cv2.getPerspectiveTransform(pts1,pts2)
            frame = cv2.warpPerspective(frame,M,(CAMERA_WIDTH,CAMERA_HEIGHT))

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

def BatchGenerator(batch_x, batch_y, setname):
    loopiter=0
    batch_len = len(batch_x)
    print("generator " + setname + " started with " + str(batch_x))
    while True:
        #print("generator " + setname + ": "+str(batch_x[loopiter]) + " " + str(batch_y[loopiter]))
        img = np.load(batch_x[loopiter])
        img = img/255.0
        labels = np.load(batch_y[loopiter])
        
        y = np.zeros((labels.shape[0], ANGLE_STEPS))
        for i,angle in enumerate(labels):
            idx = (ANGLE_STEPS-1)/2*(angle)+(ANGLE_STEPS-1)/2
            y[i,math.ceil(idx)] = 1
        
        labels = y
        
        loopiter = loopiter + 1
        if loopiter >= batch_len:
            loopiter=0
        yield (img, labels)

"""
Normalize input data:
- remove mean
- divide by standard deviation
- move to -1:1 range
"""
def Normalize(data):
    y = np.array(data)

    StdScaler = StandardScaler()
    MaxScaler = MaxAbsScaler()
    y_norm = StdScaler.fit_transform(y.reshape(-1, 1)) # remove the mean
    y_norm = MaxScaler.fit_transform(y_norm)           # shrink to the -1:1 range       
    y_norm = y_norm.tolist()
    
    # Get normalization parameters
    y_std_dev = StdScaler.var_[0]
    y_mean    = StdScaler.mean_[0]
    y_scaling = MaxScaler.max_abs_[0]

    return (y_norm, y_std_dev, y_mean, y_scaling)

"""
Generate batch files with intemediate data preprocessing such as:
    - normalization
Returns: 
"""
def BatchDataGeneration(dataFilename):
    global DATASET_SIZE

    # Get execution time
    millis = int(round(time.time() * 1000))
    
    try:
        os.mkdir(OUTPUT_DIR)
    except FileExistsError:
        pass

    # Open data file. The file is in the format:
    #   <image_name>.<format> <steering angle[float]>
    with open(dataFilename) as f:
        linelist = f.readlines()

    #linelist = linelist[1:100]
    listlen = len(linelist)
    
    DATASET_SIZE = listlen

    print("  Dataset has " + str(listlen)+ " samples")
            
    # Normalize angle values
    """
    x = []
    y = []
    for line in linelist:
        x.append(line.split()[0]) # get image name
        y.append(line.split()[1]) # get angle
    
    y, y_std_dev, y_mean, y_scaling = Normalize(y)

    # dataset is a list of strings in the format:
    # <image_name>.<format> <normalized steering angle[float]>
    dataset = []        
    for i in range(len(x)):
        dataset.append(str(x[i])+ " " + str(y[i]).replace('[','').replace(']',''))
    
    """
    dataset = linelist
    # shuffle the data randomly
    dataset = np.random.permutation(np.array(dataset))
    
    print(dataset[1:5])        

    # split the dataset in chunks
    chunks = np.array_split(dataset, NUM_BATCH)
    
    # Split work across multiple threads
    i=0
    epoch = 1
    tot_epoch = int(NUM_BATCH/NUM_THREADS)
    
    status = Queue()
    progress = collections.OrderedDict()
    while (i < len(chunks)):
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

def NNCreateModel():
    global DATASET_SIZE

    # build NN
    input_units = CAMERA_WIDTH*CAMERA_HEIGHT
    output_units = ANGLE_STEPS # angle ranges from -1 to 1 in steps of 0.1

    print("> Build Network")

    # Model Architecture
    model = Sequential()
    model.add(Dense(units=int(input_units/100), activation='relu', input_dim=input_units))
    model.add(Dense(units=int(input_units/100), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=int(512), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=int(512), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_units, activation='softmax'))
    #model.add(Dense(units=1, activation='linear'))
        
    # Model configuration
    loss = 'categorical_crossentropy'
    optimizer='adam'
    #optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=keras.metrics.sparse_categorical_accuracy)

    return model

def NNFitModel(model, modelname, x_train, y_train, x_val, y_val):

    # Instatiate batch generators for the model
    bg_train = BatchGenerator(x_train, y_train, 'train')
    bg_val   = BatchGenerator(x_val, y_val, 'val')

    epochs = 200
    checkpoint = ModelCheckpoint(modelname, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    hist = History()
    train_steps = len(x_train)
    val_steps = len(x_val)

    hist = model.fit_generator( bg_train, steps_per_epoch=train_steps,
        validation_data=bg_val,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks_list)        

    return (model, hist.history)

def NNSaveModel(model, modelname, hist):        
    # Save the model
    #odelName = "Model_" + loss + "_" + optimizer + "_epochs" + str(epochs) + "_bs" + str(int(DATASET_SIZE/NUM_BATCH))
    plot_model(model, to_file=modelname+'.png', show_shapes=True)
    model.save(modelname)

    json.dump(hist, open(modelname+".hist", 'w'))

    # Plot training & validation accuracy values
    plt.plot(hist['acc'])
    plt.plot(hist['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(modelname+'_accuracy.png')

    # Plot training & validation loss values
    plt.figure()
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(modelname+'_loss.png')
    plt.show()

def ComputeAccuracy(model, x, y, testset):
    print("")
    print("***** Evaluating " + testset + " Accuracy *****")
    bg_test  = BatchGenerator(x, y, testset)

    j=0
    err = 0
    thr = 1
    for i in range(len(x)):
        imgs, labels = next(bg_test)
        predictions = model.predict(imgs)
        
        p_idx = np.argmax(predictions, axis=1)
        y_idx = np.argmax(labels, axis=1)

        diff = abs(p_idx-y_idx)
        diff = (diff>thr)*1
        err=err+np.sum(diff)
        j=j+len(predictions)

    print(testset + " Acc : %.4f" % (1-float(err)/j))
    print("")
    print("pred " + str(p_idx))
    print("")
    print("labels " + str(y_idx))
    

def main():    
    modelname=""
    if len(sys.argv) > 1:
        modelname = sys.argv[1]
    
    x_batch_list = sorted(glob.glob(DATA_X_FILENAME+'*'))
    y_batch_list = sorted(glob.glob(DATA_Y_FILENAME+'*'))

    # Split dataset in train/val/test sets
    x_train, x_test, y_train, y_test = train_test_split(x_batch_list, y_batch_list, test_size = .1, shuffle=False)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = .1, shuffle=False)

    if (os.path.isfile(modelname)):
        print("> Loading existing NN model " + modelname)        
        model = load_model(modelname)        
    else:        
        print("> Generating new NN model...")
    
        if os.path.isdir(OUTPUT_DIR) and ((len(x_batch_list) == NUM_BATCH) and (len(y_batch_list) == NUM_BATCH)):
            print("> Found already existing batch files")
        else:
            print("> Some batch data elements are missing or number of batch has changed")        
            BatchDataGeneration(DATASET_FILENAME)
        
        model = NNCreateModel() 
    
    if sys.argv[2] == "1":
        print("> Fitting model "+ modelname)
        model, hist = NNFitModel(model, modelname, x_train, y_train, x_val, y_val)
        NNSaveModel(model, modelname, hist)
    
    """if (os.path.isfile(modelname+'.hist')):
        print("> Load previous history file")
        prev_hist = json.load(open(modelname+'.hist', 'r'))
        prev_hist.update(hist)
    """
    
    # Calculate custom accuracy over the sets
    ComputeAccuracy(model, x_train, y_train, 'Train')
    ComputeAccuracy(model, x_val, y_val, 'Val')
    ComputeAccuracy(model, x_test, y_test, 'Test')
    
            
if __name__ == "__main__":
    main()
