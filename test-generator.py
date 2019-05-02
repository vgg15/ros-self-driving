import numpy as np
import math
import glob
from sklearn.model_selection  import train_test_split
import cv2

ANGLE_STEPS = 21
OUTPUT_DIR = 'output_64x64px_rgb/'
DATA_X_FILENAME = OUTPUT_DIR + 'data_x'
DATA_Y_FILENAME = OUTPUT_DIR + 'data_y'

NUM_OUTPUT_CLASSES=21
LABELS_MIN = -1
LABELS_MAX = 1
    
def encodeLabels(labels, num_classes):
    global LABELS_MIN
    global LABELS_MAX

    y = np.zeros((labels.shape[0], num_classes))
    for i,angle in enumerate(labels):
        #idx = int(np.interp(angle, [-1,1], [0, num_classes]))
        idx = np.interp(angle, [LABELS_MIN, LABELS_MAX], [0, num_classes-1])
        y[i,int(round(idx))] = 1

    return y

def decodeLabels(labels, num_classes):
    global LABELS_MIN
    global LABELS_MAX

    idx = np.argmax(labels, axis=1)
    #y = np.interp(idx, [0, num_classes-1], [-1, 1])
    y = np.interp(idx, [0, num_classes-1], [LABELS_MIN, LABELS_MAX])
    return y

def BatchGenerator(batch_x, batch_y, dataset_name):
    global NUM_OUTPUT_CLASSES

    loopiter=0
    batch_len = len(batch_x)
    #print("generator " + dataset_name + " started with " + str(batch_x))
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

def simple_generator_function():
    yield 1
    yield 2
    yield 3

x_batch_list = sorted(glob.glob(DATA_X_FILENAME+'*'))
y_batch_list = sorted(glob.glob(DATA_Y_FILENAME+'*'))

x_train, x_test, y_train, y_test = train_test_split(x_batch_list, y_batch_list, test_size = .1)

print(x_train)
bg_train = BatchGenerator(x_train, y_train, 'train')
x_old = np.zeros((1,64*64))
for i in range(100):
    x, y = next(bg_train)
    #img = np.reshape(x[i], (64,64))
    label = decodeLabels(y, NUM_OUTPUT_CLASSES)
    cv2.imshow("{:.2}".format(label[i]), x[i])
    if cv2.waitKey(0) == 27:
        exit()
    cv2.destroyAllWindows()
    print(np.array_equal(x,x_old))
    x_old = x
    print(x[1:3,6:25])
    print(y[1:3,:])

    #input("..")


