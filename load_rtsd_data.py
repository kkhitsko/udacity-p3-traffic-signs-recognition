import numpy as np
import csv
import glob
#import matplotlib.image as mpimg
from scipy import misc
from random import randint
import numpy as np

stop_classes = [3,5,6,10,11,13,15,20,21,22,30,31,32,36,50,51,52,54,59,62]


whitelist_file = 'whitelist_classes.csv'
whitelist_dict = {}
with open(whitelist_file) as whitenames:
    reader = csv.DictReader(whitenames, delimiter=',')
    for row in reader:
        whitelist_dict[int(row['class_number'])] = row['sign_class']


def load_data(dir):
    """
    Load all data from directory
    :param dir:
    :return:
    """

    ### Try to load csv file with image validations
    validation_file = dir + '/gt_train.csv'
    validation_dict = {}
    test_dict = {}
    test_file = dir + '/gt_test.csv'

    x_train_len = 0
    x_valid_len = 0
    x_test_len = 0
    fail_data_cnt = 0
    fail_test_cnt = 0

    X_train = np.zeros(shape=(30000,48,48,3))
    Y_train = np.zeros(shape=(30000,))
    X_valid = np.zeros(shape=(10000,48,48,3))
    Y_valid = np.zeros(shape=(10000,))
    X_test = np.zeros(shape=(10000,48,48,3))
    Y_test = np.zeros(shape=(10000,))
    with open(validation_file) as file:
        cnt = 0
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            validation_dict[row['filename']] = int(row['class_number'])
            img = misc.imread(dir + "/train/"+row['filename'])
            cl_num = int(row['class_number'])
            if cl_num not in stop_classes:
                rnd = randint(0,4)
                if rnd != 0:
                    #np.append(X_train,img)
                    X_train[x_train_len] = img
                    #Y_train[x_train_len] = cl_num
                    Y_train[x_train_len] = whitelist_dict[int(cl_num)]

                    x_train_len += 1
                else:
                    X_valid[x_valid_len] = img
                    #Y_valid[x_valid_len] = int(row['class_number'])
                    Y_valid[x_valid_len] = whitelist_dict[int(cl_num)]

                    x_valid_len += 1
                cnt += 1
                if cnt % 1000 == 0:
                    print("Load {} train images".format(cnt))
            else:
                fail_data_cnt += 1
        X_train = X_train[:x_train_len]
        X_valid = X_valid[:x_valid_len]
        Y_train = Y_train[:x_train_len]
        Y_valid = Y_valid[:x_valid_len]
    print("Dictionary size: {}".format(len(validation_dict)))

    with open(test_file) as file:
        cnt = 0
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            validation_dict[row['filename']] = int(row['class_number'])
            img = misc.imread(dir + "/test/"+row['filename'])
            cl_num = int(row['class_number'])
            if cl_num not in stop_classes:
                X_test[x_test_len] = img
                #Y_test[x_test_len] = int(row['class_number'])
                Y_test[x_test_len] = whitelist_dict[int(cl_num)]
                x_test_len += 1
                cnt += 1
                if cnt % 1000 == 0:
                    print("Load {} test images".format(cnt))
            else:
                fail_test_cnt += 1

        X_test = X_test[:x_test_len]
        Y_test = Y_test[:x_test_len]
    print("Failed data: {}; failed test: {}".format(fail_data_cnt,fail_test_cnt))

    return X_train,Y_train, X_valid, Y_valid, X_test, Y_test



