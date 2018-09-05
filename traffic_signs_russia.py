import pickle
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.preprocessing import LabelBinarizer

from sklearn.utils import shuffle
import random
import csv
import glob
import cv2
from load_rtsd_data import load_data

import keras
from keras.models import Sequential, model_from_json, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K

train_flag = False
test_flag = False
show_examples = False

# input image dimensions
img_rows, img_cols = 48, 48

def normImage( img_set ):
    res = np.zeros(img_set.shape, np.float)
    #res = []
    for i in range(len(img_set)):
        img = img_set[i].astype(dtype=float)
        img_min = np.min(img)
        img_max = np.max(img)
        img = (img - img_min)/(img_max-img_min)
        #img = (img - img_min)/img_max
        res[i] = img
        #res.append(img)
        #np.append(res,img)

    return res

X_train_orig,y_train, X_valid_orig, y_valid, X_test_orig, y_test = load_data("/Users/Telematika/Documents/traffic_signs/rtsd-r1")

n_train = len(X_train_orig)
n_valid = len(X_valid_orig)
n_test = len(X_test_orig)
image_shape = X_train_orig[0].shape
n_classes = len(set(y_valid))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


X_train = normImage(X_train_orig)
X_valid = normImage(X_valid_orig)
X_test =  normImage(X_test_orig)

X_train, y_train = shuffle(X_train, y_train)
hist = {}
for idx in set(y_train.tolist()):
    print("class {} -> {}".format(idx,y_train.tolist().count(idx)))
    hist[idx] = y_train.tolist().count(idx)


plt.figure(figsize=(20,20))
plt.stem(hist.keys(), hist.values())
plt.xlabel('Class ID')
plt.ylabel('Number of elements in class')
#plt.savefig("out_images/rus/y_classes.jpg")

#plt.show()

print(type(X_train),type(y_train))
print(X_train.shape, y_train.shape)
#exit(1)

#signnames_file = 'dicts/numbers_to_classes.csv'
#signs_dict = {}
#with open(signnames_file) as signnames:
#    reader = csv.DictReader(signnames, delimiter=',')
#    for row in reader:
#        signs_dict[int(row['class_number'])] = row['sign_class']


whitelist_file = 'dicts/whitelist_classes.csv'
whitelist_dict = {}
with open(whitelist_file) as whitenames:
    reader = csv.DictReader(whitenames, delimiter=',')
    for row in reader:
        whitelist_dict[int(row['class_number'])] = row['sign_class']

demo_pic_cnt = 3
if show_examples:
    plt.figure(figsize=(15,10))
    for i in range(demo_pic_cnt):
        index = random.randint(0, len(X_valid))
        demo_img = X_valid[index]
        demo_img_orig = X_valid_orig[index]
        title_str_orig = whitelist_dict[y_valid[index]] + " before normalize"
        title_str = whitelist_dict[y_valid[index]] + " after normalize"

        # Originals
        plt.subplot(2, demo_pic_cnt, i+1)
        plt.imshow(demo_img_orig)
        plt.title(title_str_orig)

        # Normalize
        plt.subplot(2, demo_pic_cnt, i+1 + demo_pic_cnt)
        plt.imshow(demo_img)
        plt.title(title_str)

    #plt.show()

EPOCHS = 5
BATCH_SIZE = 128


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

model = Sequential()

if train_flag:


    model.add(Conv2D(100, kernel_size=(7, 7),
                 activation='relu',
                 input_shape=(48,48,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    #
    model.add(Conv2D(150, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(250, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(X_test, y_test))

    print("Save model to file...")
    model.save_weights("checkpoint/model.h5")
    model.save("checkpoint/signs.h5")

    model_yaml = model.to_json()
    with open("checkpoint/model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    print("OK")

else:
    ### Load saved data
    yaml_file = open("checkpoint/model.yaml", "r")
    model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml( model_yaml )
    print("Load model from file...")
    model.load_weights("checkpoint/model.h5")
    print("OK")

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


if test_flag:
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


path = 'images/russia/*.jpg'
#path = 'custom_images/GTSRB/*.ppm'
files = glob.glob(path)

batch_size = len(files)
batch_x = np.zeros((batch_size, 48, 48, 3), dtype=np.uint8)

print("Load {} files".format(batch_size))

for i in range(batch_size):
    try:
        img = cv2.imread(files[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resized_image = cv2.resize(img, (32, 32))
        batch_x[i, :, :, :] = cv2.resize(img, (48, 48))
    except ValueError as e:
        print("Can't get image {}. Reason: {}".format(i,e))


X_mydata = normImage(batch_x)
Y_testdata = np.zeros(shape=(batch_size,))
print(Y_testdata)

testresults_file = 'images/russia/results2.csv'

with open(testresults_file) as testresults:
    reader = csv.DictReader(testresults, delimiter=',')
    cnt = 0
    for row in reader:
        print(cnt,row)
        Y_testdata[cnt] = row['class']
        cnt += 1


print(Y_testdata)

# convert class vectors to binary class matrices
Y_testdata = keras.utils.to_categorical(Y_testdata, n_classes)

print(Y_testdata)


score = model.evaluate(X_mydata,Y_testdata,verbose=2)
print("score: {}".format(score))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict(X_mydata,verbose=2)
top_k_num = 5

for i in range(len(X_mydata)):
    max_val = 0.0
    idx = -1
    for j in range(n_classes):
        if ( y_pred[i,j] > max_val ):
            max_val = y_pred[i,j]
            idx = j
    print("Image {} {}: has class {} with prob {:.3f}%".format(i,files[i],idx, max_val*100))








