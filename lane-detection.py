#!/home/alessandro/venv/bin/python3
import math
import time
import sys
import json
import os
from os import system, name 
from threading import Thread
from multiprocessing import Queue
import collections
import glob

import cv2
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler 
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import keras
from keras import regularizers
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import *
from keras.utils import plot_model
from keras.models import load_model

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

# Global variable
LABELS_MIN = 0
LABELS_MAX = 0
NUM_OUTPUT_CLASSES = 0

def print_progress(progress, curr_epoch, tot_epoch, eta):
    totpercent = 0.0
    for id, percent in progress.items():
        totpercent = totpercent + percent

    totpercent = int(totpercent/NUM_THREADS)
    bar = ('=' * int(totpercent/5)).ljust(20)        
    sys.stdout.write("\rEpoch %s/%s, Total progress [%s] %s%%, elapsed: %s" % (curr_epoch, tot_epoch, bar, totpercent, eta))    
    sys.stdout.flush()

class PreProcessingThread (Thread):
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


"""
@brief: Encode labels from float number to one-hot vector

"""
def encodeLabels(labels, num_classes):
    global LABELS_MIN
    global LABELS_MAX

    y = np.zeros((labels.shape[0], num_classes))
    for i,angle in enumerate(labels):
        #idx = int(np.interp(angle, [-1,1], [0, num_classes]))
        idx = np.interp(angle, [LABELS_MIN, LABELS_MAX], [0, num_classes-1])
        y[i,round(idx)] = 1
    
    return y

"""
@brief: Decode labels from one-hot vector to float number
"""
def decodeLabels(labels, num_classes):
    global LABELS_MIN
    global LABELS_MAX

    idx = np.argmax(labels, axis=1)
    #y = np.interp(idx, [0, num_classes-1], [-1, 1])
    y = np.interp(idx, [0, num_classes-1], [LABELS_MIN, LABELS_MAX])
    return y

"""
@brief: Import input data into the model in batch.
@params:
    - batch_x: list of input data X batch filenames
    - batch_y: list of label data Y batch filenames
    - dataset_name: dataset name 
    
"""
def DataGenerator(batch_x, batch_y, dataset_name):
    global NUM_OUTPUT_CLASSES

    loopiter=0
    batch_len = len(batch_x)
    print("generator " + dataset_name + " started with " + str(batch_x))
    while True:
        #print("generator " + dataset_name + ": "+str(batch_x[loopiter]) + " " + str(batch_y[loopiter]))
        img = np.load(batch_x[loopiter])
        img = img/255.0
        labels = np.load(batch_y[loopiter])

        labels = encodeLabels(labels, NUM_OUTPUT_CLASSES)
        
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

def visualizeLabels(labels):
    global NUM_OUTPUT_CLASSES
    global LABELS_MIN
    global LABELS_MAX
    
    #plt.hist(labels, bins = NUM_OUTPUT_CLASSES,  range = (-1, 1), align='left', histtype='step')
    plt.hist(labels, bins = NUM_OUTPUT_CLASSES)
    #plt.xticks(np.arange(LABELS_MIN, LABELS_MAX+ANGLE_RESOLUTION,0.2))
    plt.title("Steering wheel angle")
    plt.figure()

    labels_e = encodeLabels(labels, NUM_OUTPUT_CLASSES)
    idx = np.argmax(labels_e, axis=1)
    
    plt.hist(idx, bins=NUM_OUTPUT_CLASSES, range = (0, NUM_OUTPUT_CLASSES), align='left')
    plt.xticks(np.arange(NUM_OUTPUT_CLASSES))
    plt.title("Steering wheel labels")

    labels_D = decodeLabels(labels_e, NUM_OUTPUT_CLASSES)

    """
    plt.figure()
    plt.hist(labels_D, bins = NUM_OUTPUT_CLASSES)
    plt.title("Steering wheel angle decoded")
    """
    #print(np.mean((labels-labels_D)))
    plt.show()

def importDataset(dataset_name):
    global LABELS_MIN
    global LABELS_MAX
    global NUM_OUTPUT_CLASSES

    # Open data file. The file is in the format:
    #   <image_name>.<format> <steering angle[float]>
    print("> Opening dataset file: " + dataset_name)
    with open(dataset_name) as f:
        dataset = f.readlines()
    
    print("> Dataset has " + str(len(dataset))+ " samples")

    # shuffle the data randomly
    dataset = np.random.permutation(np.array(dataset))    
    
    # Convert a list of string to  numpy array
    X = []
    Y = []
    for line in dataset:
        X.append(line.split()[0])
        Y.append(float(line.split()[1]))
    Y = np.array(Y)

    LABELS_MIN = Y.min()
    LABELS_MAX = Y.max()    

    NUM_OUTPUT_CLASSES = int(abs(LABELS_MAX-LABELS_MIN)/ANGLE_RESOLUTION)
    if (NUM_OUTPUT_CLASSES % 2) == 0:
        NUM_OUTPUT_CLASSES = NUM_OUTPUT_CLASSES + 1
    
    visualizeLabels(Y)

    

"""
Generate batch files:
    - normalization
Returns: 
"""
def BatchDataGeneration(dataset):
    # Get execution time
    millis = int(round(time.time() * 1000))
    
    try:
        os.mkdir(OUTPUT_DIR)
    except FileExistsError:
        pass
            
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
            t = PreProcessingThread(i, chunks[i].tolist(), status)
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

def NNCreateModel(modelname):
    global NUM_OUTPUT_CLASSES

    # build NN
    input_units = CAMERA_WIDTH*CAMERA_HEIGHT
    output_units = NUM_OUTPUT_CLASSES # angle ranges from -1 to 1 in steps of 0.1

    print("> Build Network")

    # Model Architecture
    model = Sequential()    
    kernel_regularizer=regularizers.l2(0.01)

    model.add(Dense(units=int(input_units/100), activation='relu', input_dim=input_units))
    #model.add(Dropout(0.5))
    model.add(Dense(units=int(input_units/100), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=int(512), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=int(512), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_units, activation='softmax'))
    #model.add(Dense(units=1, activation='linear'))
    
    #model.load_weights('model-1_cc_adam.h5_weights.h5')
    
    # Model configuration
    loss = 'categorical_crossentropy'
    optimizer = 'adam' #optimizers.Adam(lr=0.001) # decay=1e-1
    #optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    return model

def NNFitModel(model, modelname, x_train, y_train, x_val, y_val):

    # Instatiate batch generators for the model
    bg_train = DataGenerator(x_train, y_train, 'train')
    bg_val   = DataGenerator(x_val, y_val, 'val')

    epochs = 20
    checkpoint = ModelCheckpoint(modelname, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    #earlystop = EarlyStopping(monitor='loss', min_delta=0.01, patience=4, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./tensorboard_logs/' + modelname)
    callbacks_list = [checkpoint, tensorboard]

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
    plot_model(model, to_file=modelname+'.png', show_shapes=True)
    model.save(modelname)
    model.save_weights(modelname+'_weights.h5')

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
    bg_test  = DataGenerator(x, y, testset)

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
    
    dataset = importDataset(DATASET_FILENAME)
    
    exit()

    x_batch_list = sorted(glob.glob(DATA_X_FILENAME+'*'))
    y_batch_list = sorted(glob.glob(DATA_Y_FILENAME+'*'))

    if (os.path.isfile(modelname)):
        print("> Loading existing NN model " + modelname)        
        model = load_model(modelname)     
        model.save_weights(modelname+'_weights.h5')           
    else:        
        print("> Generating new NN model...")
    
        if os.path.isdir(OUTPUT_DIR) and ((len(x_batch_list) == NUM_BATCH) and (len(y_batch_list) == NUM_BATCH)):
            print("> Found already existing batch files")
        else:
            print("> Some batch data elements are missing or number of batch has changed")        
            BatchDataGeneration(dataset)
        
        model = NNCreateModel(modelname) 
      
    # Split dataset in train/val/test sets
    x_train, x_test, y_train, y_test = train_test_split(x_batch_list, y_batch_list, test_size = .1, shuffle=False)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = .1, shuffle=False)

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
