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
import argparse


import cv2
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler 
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt

import keras
from keras import regularizers
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import *
from keras.utils import plot_model
from keras.models import load_model

LOG_DIR = 'logs/'
OUTPUT_DIR = 'output_64x64px_rgb/'
DATA_X_FILENAME = OUTPUT_DIR + 'data_x'
DATA_Y_FILENAME = OUTPUT_DIR + 'data_y'
NUMPY_EXT = '.npy'

CAMERA_WIDTH = 64
CAMERA_HEIGHT = 64

NUM_THREADS = 4
NUM_BATCH = 128

DATASET_FOLDER = 'bag_output_rec3/'
DATASET_FILENAME = DATASET_FOLDER  + 'data.txt'

ANGLE_RESOLUTION = 0.5

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
            
            # open image
            frame = cv2.imread(DATASET_FOLDER+img)

            # resize frame
            frame = cv2.resize(frame,(CAMERA_WIDTH, CAMERA_HEIGHT))

            # apply threshold
            #ret, frame = cv2.threshold(frame, 70, 255, cv2.THRESH_BINARY)
            
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #pts1 = np.float32([[125,30],[CAMERA_WIDTH-128,30],[0,CAMERA_HEIGHT-60],[CAMERA_WIDTH,CAMERA_HEIGHT-60]])
            #pts2 = np.float32([[0,0],[CAMERA_WIDTH,0],[0,CAMERA_HEIGHT],[CAMERA_WIDTH,CAMERA_HEIGHT]])
    
            #M = cv2.getPerspectiveTransform(pts1,pts2)
            #frame = cv2.warpPerspective(frame,M,(CAMERA_WIDTH,CAMERA_HEIGHT))

            # unroll
            #frame = np.reshape(frame, (1,CAMERA_WIDTH*CAMERA_HEIGHT))
            
            self.xlist.append(frame)
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
    #print("generator " + dataset_name + " started with " + str(batch_x))
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
        

    ### Hyper-parameters tuning ###
    # Optimizer
    lr             = 0.0006 #10**random.uniform(-4,-1)    
    lr_decay       = 0.00016 #10**random.uniform(-4,-1)
    momentum       = 0.72 #random.uniform(0.5, 0.9)
    adam_optimizer = keras.optimizers.Adam(lr=lr , decay=lr_decay)
    sgd_optimizer  = keras.optimizers.SGD(lr=lr, decay=lr_decay, momentum=momentum)
    optimizers     = {'adam':adam_optimizer, 'sgd': sgd_optimizer}
    optimizer_name = random.choice(['adam', 'sgd'])
    optimizer      = optimizers[optimizer_name]
    
    # Conv2D
    filters      = 32 #random.choice([8, 16, 32])
    filter_size  = 3 #random.choice([3, 5, 7])
    stride       = 1 #random.randint(1,3)
    conv_layers  = 1 #random.randint(1,3)
    padding      = 'valid' #'same'

    # Dense
    dense_layers = 2 #random.randint(1, 3)
    dense_units  = random.randint(32, 512)     
    l2_reg       = 10**random.uniform(-3, -2)
    dropout      = random.uniform(0.1, 0.3)

    loss = 'categorical_crossentropy'

    #### Model Architecture ###
    model = Sequential()
    if (l2_reg == 0):
        kernel_regularizer = None
    else:
        kernel_regularizer=regularizers.l2(l2_reg)

    # Convolutional layers
    model.add(Conv2D(filters, filter_size, strides = stride, input_shape = (CAMERA_HEIGHT, CAMERA_WIDTH, 3), 
            padding=padding, data_format="channels_last", activation='relu', use_bias=True, kernel_regularizer=kernel_regularizer))
    for i in range(1, conv_layers):
        filters *= 2
        model.add(Conv2D(filters, filter_size, strides = stride,
            padding='valid', data_format="channels_last", activation='relu', use_bias=True, kernel_regularizer=kernel_regularizer))
        if ((i+1) % 2) == 0:
            model.add(MaxPooling2D(pool_size=(3,3), strides=2, data_format='channels_last'))

    model.add(Flatten())

    # Dense layers
    for _ in range(dense_layers):
        model.add(Dense(units=dense_units, activation='relu', kernel_regularizer=kernel_regularizer))
        if (dropout != 0):
            model.add(Dropout(dropout))

    model.add(Dense(units=output_units, activation='softmax'))
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
    
    modelname = modelname + "-lr_{:.2}-dec_{:.2}-CL_{}-f_{}-fs_{}-s_{}-DL_{}-u_{}-reg_{:.2}-drop_{:.2}-mom_{:.2}-opt_{}".format(
        lr, lr_decay, conv_layers, filters, (filter_size), (stride), dense_layers, dense_units, l2_reg, dropout, momentum, optimizer_name)
    modelname = modelname.replace('[', '_').replace(' ', '_').replace(',', '')

    print(modelname)

    return model, modelname

def NNFitModel(model, modelname, x_train, y_train, x_val, y_val):

    # Instatiate batch generators for the model
    bg_train = DataGenerator(x_train, y_train, 'train')
    bg_val   = DataGenerator(x_val, y_val, 'val')

    epochs = 20
    checkpoint = ModelCheckpoint(LOG_DIR + modelname + "/model.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=5)
    earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir=LOG_DIR + modelname, write_images = True)
    callbacks_list = [checkpoint, tensorboard, earlystop]

    hist = History()
    train_steps = len(x_train)
    val_steps = len(x_val)

    hist = model.fit_generator(bg_train, steps_per_epoch=train_steps,
        validation_data=bg_val,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks_list)        

    plot_model(model, to_file=LOG_DIR + modelname+'/model.png', show_shapes=True)

    return (model, hist.history)

def NNSaveModel(model, modelname, hist):
    # Save the model    
    model.save(LOG_DIR+modelname)
    #model.plot(modelname+'_weights.h5')

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

def ComputeAccuracy(model, thr, x, y, testset):
    print("")
    print("***** Evaluating " + testset + " Accuracy *****")
    bg_test  = DataGenerator(x, y, testset)

    j=0
    err = 0
    y = []
    p = []

    for i in range(len(x)):
        imgs, labels = next(bg_test)
        predictions = model.predict(imgs)
        
        p_idx = np.argmax(predictions, axis=1)
        y_idx = np.argmax(labels, axis=1)

        y.extend(y_idx)
        p.extend(p_idx)
        
        diff = abs(p_idx-y_idx)
        diff = (diff>thr)*1
        err=err+np.sum(diff)
        j=j+len(predictions)

    print("")
    if testset == 'Test':
        print("pred " + str(p_idx))
        print("")
        print("labels " + str(y_idx))
    print("Metrics: ")
    print(testset + " Acc : %.4f" % (1-float(err)/j))
    #print(accuracy_score(y,p, normalize=True, sample_weight=None))
    prec, recall, fscore, _ = precision_recall_fscore_support(y,p, average='macro')
    print("Precision: " + str(prec))
    print("Recall: " + str(recall))
    print("F-score: " + str(fscore))

def main():    

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Load and fit existing model")
    parser.add_argument("--fit", type=int, help="Create and fit a new model for <n> times")
    parser.add_argument("--estimate", type=int, help = "Estimate the model on train/val/test sets with <thr> threshold")
    parser.add_argument("modelname", help = "Model name")

    try:
        args = parser.parse_args()
    except:
        parser.print_help(sys.stderr)
        exit()
    
    modelname = args.modelname
    
    dataset = importDataset(DATASET_FILENAME)
    
    x_batch_list = sorted(glob.glob(DATA_X_FILENAME+'*'))
    y_batch_list = sorted(glob.glob(DATA_Y_FILENAME+'*'))

    # Split dataset in train/val/test sets
    x_train, x_test, y_train, y_test = train_test_split(x_batch_list, y_batch_list, test_size = .1, shuffle=False)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = .1, shuffle=False)

    if (args.load):
        print("> Loading existing NN model " + modelname)        
        model = load_model(LOG_DIR+modelname+"/model.h5")     
        print("> Fitting model " + modelname)
        model, hist = NNFitModel(model, modelname, x_train, y_train, x_val, y_val)
        model.save_weights(modelname+'_weights.h5')
    elif (args.fit):            
        if os.path.isdir(OUTPUT_DIR) and ((len(x_batch_list) == NUM_BATCH) and (len(y_batch_list) == NUM_BATCH)):
            print("> Found already existing batch files")
        else:
            print("> Some batch data elements are missing or number of batch has changed")        
            BatchDataGeneration(dataset)
        
        
        for i in range(args.fit):
            logName = modelname
            print("> Generating new NN model {}/{}...".format(i+1, args.fit))
            model, logName = NNCreateModel(logName)
            print("> Fitting model " + logName)
            model, hist = NNFitModel(model, logName, x_train, y_train, x_val, y_val)
            #NNSaveModel(model, logName, hist)
    elif (args.estimate >= 0):
        print("> Loading existing NN model " + modelname)        
        model = load_model(modelname)
        # Calculate custom accuracy over the sets
        ComputeAccuracy(model, args.estimate, x_train, y_train, 'Train')
        ComputeAccuracy(model, args.estimate, x_val, y_val, 'Val')
        ComputeAccuracy(model, args.estimate, x_test, y_test, 'Test')
            
if __name__ == "__main__":
    main()
