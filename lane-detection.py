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
import random

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
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import *
from keras.utils import plot_model
from keras.models import load_model

LOG_DIR = 'logs/'
OUTPUT_DIR = 'output/'
DATA_X_FILENAME = OUTPUT_DIR + 'data_x'
DATA_Y_FILENAME = OUTPUT_DIR + 'data_y'
NUMPY_EXT = '.npy'

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

NUM_THREADS = 4
NUM_BATCH = 128

DATASET_FOLDER = 'bag_output_rec3/'
DATASET_FILENAME = DATASET_FOLDER  + 'data.txt'

ANGLE_RESOLUTION = 0.1

# Global variable - DO NOT MANUALLY MODIFY!
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

        for line in self.linelist:            
            row = line.split()
            img = str(row[0])
            angle = float(row[1])    
            
            # open image in b&w
            frame = cv2.imread(DATASET_FOLDER+img)

            # resize frame
            frame = cv2.resize(frame,(CAMERA_WIDTH,CAMERA_HEIGHT))

            # apply threshold
            ret, frame = cv2.threshold(frame, 70, 255, cv2.THRESH_BINARY)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pts1 = np.float32([[125,30],[CAMERA_WIDTH-128,30],[0,CAMERA_HEIGHT-60],[CAMERA_WIDTH,CAMERA_HEIGHT-60]])
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
        y[i,int(round(idx))] = 1
    
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
        #img = img.reshape(img.shape[0], CAMERA_WIDTH, CAMERA_HEIGHT, 1)
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

def visualizeDataset(X, labels):
    global NUM_OUTPUT_CLASSES
    global LABELS_MIN
    global LABELS_MAX
    
    ### Input sample ###
    
    for i in range(25):
        img = cv2.imread(DATASET_FOLDER+X[i], cv2.IMREAD_GRAYSCALE)
        plt.subplot(5,5,i+1) 
        plt.imshow(img)
        plt.axis('off')
        #plt.tight_layout()
        plt.title("Angle: %.2f" % float(labels[i]))

    plt.figure()

    ### Labels ###
    plt.subplot(321)
    plt.hist(labels, bins = 50)
    plt.title("Steering wheel angles")

    ###
    plt.subplot(322)
    labels_e = encodeLabels(labels, NUM_OUTPUT_CLASSES)
    idx = np.argmax(labels_e, axis=1)
    
    plt.hist(idx, bins=NUM_OUTPUT_CLASSES, range = (0, NUM_OUTPUT_CLASSES), align='left')
    plt.xticks(np.arange(NUM_OUTPUT_CLASSES))
    plt.title("Steering wheel labels")

    ###
    _, _, y_train, y_val = train_test_split(labels, labels, test_size = .1, shuffle=False)

    plt.subplot(323)
    plt.hist(y_train, bins = 50)
    plt.title("Training set angles")

    ###
    plt.subplot(324)
    y_train = encodeLabels(y_train, NUM_OUTPUT_CLASSES)
    idx = np.argmax(y_train, axis=1)
    
    plt.hist(idx, bins=NUM_OUTPUT_CLASSES, range = (0, NUM_OUTPUT_CLASSES), align='left')
    plt.xticks(np.arange(NUM_OUTPUT_CLASSES))
    plt.title("Training set labels")
    
    ###
    plt.subplot(325)
    plt.hist(y_val, bins = 50)
    plt.title("Validation set angles")

    ###
    plt.subplot(326)
    y_val = encodeLabels(y_val, NUM_OUTPUT_CLASSES)
    idx = np.argmax(y_val, axis=1)
    
    plt.hist(idx, bins=NUM_OUTPUT_CLASSES, range = (0, NUM_OUTPUT_CLASSES), align='left')
    plt.xticks(np.arange(NUM_OUTPUT_CLASSES))
    plt.title("Validation set labels")

    """
    labels_D = decodeLabels(labels_e, NUM_OUTPUT_CLASSES)
    plt.figure()
    plt.hist(labels_D, bins = NUM_OUTPUT_CLASSES)
    plt.title("Steering wheel angle decoded")
    print(np.mean((labels-labels_D)))
    """
    plt.tight_layout()
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
    dataset = np.random.RandomState(seed=136945).permutation(np.array(dataset))    
    
    # Convert a list of string to  numpy array
    X = []
    Y = []
    for line in dataset:
        X.append(line.split()[0])
        Y.append(float(line.split()[1]))
    
    print(dataset[0:5])
    X = np.array(X)
    Y = np.array(Y)

    LABELS_MIN = Y.min()
    LABELS_MAX = Y.max()    

    NUM_OUTPUT_CLASSES = int(abs(LABELS_MAX-LABELS_MIN)/ANGLE_RESOLUTION)
    if (NUM_OUTPUT_CLASSES % 2) == 0:
        NUM_OUTPUT_CLASSES = NUM_OUTPUT_CLASSES + 1
    
    print("> Creating " + str(NUM_OUTPUT_CLASSES) + " classes with resolution " + str(ANGLE_RESOLUTION))
    visualizeDataset(X,Y)

    return dataset

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

    print("> Build Network")
   
    input_units  = CAMERA_WIDTH*CAMERA_HEIGHT
    output_units = NUM_OUTPUT_CLASSES
    
    logName = modelname

    for i in range(10):
        ### Hyper-parameters tuning ###
        lr           = 10**random.uniform(-4,-1)    
        lr_decay     = 10**random.uniform(-4,-1)
        dense_layers = random.randint(1, 5)
        dense_units  = random.randint(32, 1024) 
        l2_reg       = 10**random.uniform(-4, -1)
        dropout      = random.uniform(0.1, 0.5)
        momentum     = random.uniform(0.5, 0.9)

        loss = 'categorical_crossentropy'
        adam_optimizer = keras.optimizers.Adam(lr=lr , decay=lr_decay)
        sgd_optimizer  = keras.optimizers.SGD(lr=lr, decay=lr_decay, momentum=momentum)
        optimizers     = {'adam':adam_optimizer, 'sgd': sgd_optimizer}
        optimizer_name = random.choice(['adam', 'sgd'])
        optimizer = optimizers[optimizer_name]

        #### Model Architecture ###
        model = Sequential()    
        kernel_regularizer=regularizers.l2(l2_reg)

        model.add(Dense(units=dense_units, activation='relu', input_dim=input_units, kernel_regularizer=kernel_regularizer))

        for i in range(dense_layers):
            model.add(Dense(units=dense_units, activation='relu', kernel_regularizer=kernel_regularizer))

        model.add(Dense(units=output_units, activation='softmax'))
        
        model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
        #plot_model(model, to_file=LOG_DIR + modelname+'.png', show_shapes=True)
        
        logName = modelname + "-lr_{:.2}-dec_{:.2}-l_{}-u_{}-reg_{:.2}-drop_{:.2}-mom_{:.2}-opt_{}".format(lr,lr_decay,dense_layers,dense_units, l2_reg, dropout, momentum, optimizer_name)
        print(logName)

    exit()
    return model, modelname

def NNFitModel(model, modelname, x_train, y_train, x_val, y_val):

    # Instatiate batch generators for the model
    bg_train = DataGenerator(x_train, y_train, 'train')
    bg_val   = DataGenerator(x_val, y_val, 'val')

    epochs = 50
    checkpoint = ModelCheckpoint(modelname, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    #earlystop = EarlyStopping(monitor='loss', min_delta=0.01, patience=4, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir=LOG_DIR + modelname)
    callbacks_list = [checkpoint, tensorboard]

    hist = History()
    train_steps = len(x_train)
    val_steps = len(x_val)

    hist = model.fit_generator(bg_train, steps_per_epoch=train_steps,
        validation_data=bg_val,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks_list)        

    return (model, hist.history)

def NNSaveModel(model, modelname, hist):
    # Save the model    
    model.save(LOG_DIR+modelname)
    #model.save_weights(modelname+'_weights.h5')

    #json.dump(hist, open(modelname+".hist", 'w'))

    # Plot training & validation accuracy values
    """
    plt.plot(hist['acc'])
    plt.plot(hist['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(LOG_DIR+modelname+'_accuracy.png')

    # Plot training & validation loss values
    plt.figure()
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(LOG_DIR+modelname+'_loss.png')
    plt.show()
    """

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
        
        model, modelname = NNCreateModel(modelname) 
      
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
