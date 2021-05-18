### Libaries needed
import matplotlib.pyplot as plt

from keras import backend as K
from keras.preprocessing.image import load_img,img_to_array

import random
K.set_learning_phase(1)



def view_history(history, show_acc=True):
    """Plots training history for a model
    
       # Arguments
           history:- history from model training process
           show_acc:- Boolean, shows accuracy vs epoch graph
    """
    
    history_dict = history.history
    
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = history.epoch
    
    plt.figure(figsize=(12,12))
    
    if show_acc == True:
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        
        plt.plot(epochs[1:-1], acc[1:-1], 'go', label='Training accuracy', markersize=3)
        plt.plot(epochs[1:-1], val_acc[1:-1], 'm', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs',fontsize='large')
        plt.ylabel('Accuracy',fontsize='large')
        plt.legend()
        plt.show
    else:
        plt.plot(epochs[1:100], loss[1:100], 'ro', label='Training loss', markersize=3)
        plt.plot(epochs[1:100], val_loss[1:100], 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs',fontsize='large')
        plt.ylabel('Loss',fontsize='large')
        plt.legend()
        plt.show()

