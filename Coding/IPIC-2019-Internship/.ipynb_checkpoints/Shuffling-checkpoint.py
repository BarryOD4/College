### Libaries needed
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

import random
K.set_learning_phase(1)



def stratified_shuffling(img_set, label_set, val_percent=0.1):
    """Takes in the image set and label set and randomly
       reorginizes the data. 
       
       Returns scaled image/label test set and validation set
       
       Used with CNN
       
       # Arguments
           img_set:- array of images
           label_set:- array of accompanying labels
           val_percent:- float. val, data split between validation and train set
        
       Returns training set for: Images, Labels
        '''''' validation set for: Images, Labels
    """
    
    # transform training label to one-hot encoding
    lb = preprocessing.LabelBinarizer()
    lb.fit(label_set)
    label_set = lb.transform(label_set)

    # split training and validating data and normalises data
    print('Stratified shuffling...')
    sss = StratifiedShuffleSplit(10, val_percent, random_state = 2)
    for train_idx, val_idx in sss.split(img_set, label_set):
        X_train_tmp, X_val = img_set[train_idx], img_set[val_idx]/255
        Y_train_tmp, Y_val = label_set[train_idx], label_set[val_idx]

    X_train = X_train_tmp/255
    Y_train = Y_train_tmp
    print('Finish stratified shuffling...')
    
    return X_train, Y_train, X_val, Y_val


def stratified_shuffling_rcnn(img_set, label_set, label_amount, val_percent):
    """Takes in the image set and label set and randomly
       reorginizes the data. 
       
       Returns scaled image/label test set and validation set
       
       Used with Regression-CNN
       
       # Arguments
           img_set:- array of images
           label_set:- array of accompanying labels
           label_amount:- int. val, amount of unique labels
           val_percent:- float. val, data split between validation and train set
        
       Returns training set for: Images, Labels
        '''''' validation set for: Images, Labels
    """

    # split training and validating data and normalises data
    print('Stratified shuffling...')
    sss = StratifiedShuffleSplit(10, val_percent, random_state = 2)
    for train_idx, val_idx in sss.split(img_set, label_set):
        X_train_tmp, X_val = img_set[train_idx], img_set[val_idx]/255
        Y_train_tmp, Y_val = label_set[train_idx], label_set[val_idx]/label_amount

    X_train = X_train_tmp/255

    Y_train = Y_train_tmp/label_amount
    print('Finish stratified shuffling...')

    return X_train, Y_train, X_val, Y_val