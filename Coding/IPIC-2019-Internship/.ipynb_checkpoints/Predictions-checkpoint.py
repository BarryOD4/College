import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.preprocessing.image import load_img,img_to_array

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

import random
K.set_learning_phase(1)


## Predictions.py -----------------------------
## Functions used to make a model perdict on a test set


#-----------------------------------#
### USED FOR CLASSIFICATION CNN'S
#-----------------------------------#


def make_predictions(model, test_set, val_set):
    """Uses model on test set and returns to user the
       amount of images it guessed correctly.
       
       # Arguments
           model:- requires keras model
           test_set:- array of images
           val_set:- array of accompanying labels with images
       
       Returns amount of images correctly guessed from test set.
    """
    
    ## Uses model to predict some amount of images
    predict = model.predict_classes(test_set, batch_size=5, verbose=1)
    
    ## We use the length of these two arrays when we sift through the data to find
    ##  the right predictions and wrong predictions
    images = len(test_set)

    ## Initialises variables for loop
    correctly_guessed = 0

    ## Begins loop to find total correct predictions
    for i in range(images):
        if predict[i] == np.argmax(val_set[i]):
            correctly_guessed += 1

    ## Returns amount of predictions were correct
    print('\nCorrectly guessed    = ', correctly_guessed)
    print('Inorrectly guessed   = ', (images - correctly_guessed))
    
    
def show_predictions(model, test_set, val_set, image_guess, img_res, data='OSNR', GRAY=True):
    """
        Uses model on test set and returns to user the
       amount of images it guessed correctly.
       
        Along with the amount of images, returns some amount
       of images with a guess alongside the correct answer
       
       'data' terms that are accepted;
           'OSNR' - conversion for OSNR images (at 0.5dB intervals, starting at 12dB)
           'disp' - conv. for dispersion images (at 10 ps/nm/km intervals, starting at 0)
           'disp' - conv. for dispersion images (at 100 ps/nm/km intervals, starting at 0~100)
        
        # Arguments
           model:- requires keras model
           test_set:- array of images
           val_set:- array of accompanying labels with images
           image_guess:- int. value, how many images to show
           img_res:- int. val, input image resolution
           GRAY:- Boolean, if image is gray or multi channel
           
       Returns amount of images correctly guessed from test set.
    """
    
    ## Uses model to predict some amount of images
    predict = model.predict_classes(test_set, batch_size=5, verbose=1)
    
    ## Initialises variables for loop
    correctly_guessed = 0

    ## Defines figure dimensions
    fig = plt.figure(figsize=(20,30))

    ## Begins loop to find correct predictions and relay results to user
    ##  Searches through the prediction array and compares it to the actual array.
    ## Displays image with the prediction and answer on the title
    for i in range(image_guess):
        correct = False
        actual = np.argmax(val_set[i])

        if predict[i] == actual:
            correctly_guessed += 1
            correct = True

        plt.subplot(6,3,i+1)
        fig.subplots_adjust(left=0.01,
                            right=0.7,
                            bottom=0.1,
                            top=1.2,
                            wspace=0.5,
                            hspace=0.2
                           )
        if GRAY == False:
            plt.imshow(test_set[i].reshape(img_res,img_res,3))
        else:
            plt.imshow(test_set[i].reshape(img_res,img_res), cmap='gray')

        if correct == True:
            if data == 'disp':
                plt.title('Correct! \nPrediction = {}ps/nm   Truth = {}ps/nm'
                          .format((10+10*predict[i]), (10+10*(actual))), fontsize=15)
                
            if data == 'disp-short':
                plt.title('Correct! \nPrediction = {} ~ {}ps/nm   Truth = {} ~{}ps/nm'
                      .format(100*(predict[i]), (100+100*predict[i]), 100*(actual), (100+100*(actual)), fontsize=15))
                
            if data == 'OSNR':
                plt.title('Correct! \nPrediction = {}dB   Truth = {}dB'
                      .format((12+0.5*predict[i]), (12+0.5*(actual))), fontsize=15)
                
            
        else:
            if data == 'disp':
                plt.title('\nPrediction = {}ps/nm   Truth = {}ps/nm'
                          .format((10+10*predict[i]), (10+10*(actual))), fontsize=15)
                
            if data == 'disp-short':
                plt.title('\nPrediction = {} ~ {}ps/nm   Truth = {} ~{}ps/nm'
                      .format(100*(predict[i]), (100+100*predict[i]), 100*(actual), (100+100*(actual)), fontsize=15))
                
            if data == 'OSNR':
                plt.title('\nPrediction = {}dB   Truth = {}dB'
                      .format((12+0.5*predict[i]), (12+0.5*(actual))), fontsize=15)

    ## Returns amount of predictions that were correct
    print('Correctly guessed    = ', correctly_guessed)
    print('Inorrectly guessed   = ', (image_guess-correctly_guessed))
    
    
    
#-----------------------------------##-----------------------------------#



#-----------------------------------#
### USED FOR REGRESSION-CNN'S
#-----------------------------------#
    
def predictions_rcnn(model, test_set, val_set, convert=False, data='OSNR', types=0, errorbars=False, mse=0):
    """
       Shows a RCNN's predictions and compares 
       predictions to actual value.
       If 'convert' is True, converts x-axis and y-axis
       values to real life values
       
       # Arguments
           model:- requires keras model
           test_set:- array of images
           val_set:- array of accompanying labels with images
           convert:- Boolean, converts axes to understandable units
           data:- string, determines type of data in test_set
           types:- int. val, amount of classes
           errorbars:- Boolean, adds errorbars to graph
           mse:- int. val, Mean Square Error, used in errorbar calculation
           
       Returns graph of correlation between guesses and actual answers
    """
    
    predictions = model.predict(test_set)
    
    
    plt.figure(figsize=(12,12))
    
    if convert is True:
        if data == 'OSNR':
            conversion = (max(types) + 1)*0.5
            val_set, predictions, mse = (12 + val_set*conversion), (12 + predictions*conversion), (mse*conversion)
            error = 'Mean Square Error = {0:3.3f} dB'.format(mse)
            plt.xlabel('True (dB)', fontsize=15)
            plt.ylabel('Prediction (dB)', fontsize=15)
            
        if data == 'disp' or data == 'disp-short':
            if data == 'disp':
                conversion = (max(types) + 1)*10
            else:
                conversion = (max(types) + 1)*100
            val_set, predictions, mse = (val_set*conversion), (predictions*conversion), (mse*conversion)
            error = 'Mean Square Error = {0:3.3f} ps/nm/km'.format(mse)
            plt.xlabel('True (ps/nm/km)', fontsize=15)
            plt.ylabel('Prediction (ps/nm/km)', fontsize=15)
    
        
    else:
        error = 'Mean Square Error = {0:3.3f}'.format(mse)
        plt.xlabel('True', fontsize=15)
        plt.ylabel('Prediction', fontsize=15)
    
    if errorbars == True:
        plt.errorbar(val_set, predictions, yerr=mse, fmt='.b', ecolor='b', barsabove=True, capsize=3, label=error)
        
    plt.title('Prediction vs Actual ({})'.format(data), fontsize=15)
    plt.plot(val_set, predictions, 'ro', markersize=5, label='Prediction Values')
    plt.plot(np.linspace(start=0,stop=max(val_set), num=100), np.linspace(start=0,stop=max(val_set), num=100), 'k-', label='True Values')
    plt.legend()
    
    

#-----------------------------------##-----------------------------------#