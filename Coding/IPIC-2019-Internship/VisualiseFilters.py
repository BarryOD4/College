## Multiple function and Classes that are used to visualise filters and activation maps

### Links and tutorials are as follows:
###  https://www.kaggle.com/ernie55ernie/mnist-with-keras-visualization-and-saliency-map
###  https://github.com/experiencor/deep-viz-keras/blob/master/saliency.py
###  


## Import packages + libraries for later use
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





class SaliencyMask(object):
    def __init__(self, model, output_index=0):
        pass

    def get_mask(self, input_image):
        pass

    def get_smoothed_mask(self, input_image, stdev_spread=.2, nsamples=50):
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image, dtype = np.float64)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples

class GradientSaliency(SaliencyMask):

    def __init__(self, model, output_index = 0):
        # Define the function to compute the gradient
        input_tensors = [model.input]
        gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
        self.compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

    def get_mask(self, input_image):
        # Execute the function to compute the gradient
        x_value = np.expand_dims(input_image, axis=0)
        gradients = self.compute_gradients([x_value])[0][0]

        return gradients

# https://github.com/experiencor/deep-viz-keras/blob/master/visual_backprop.py
class VisualBackprop(SaliencyMask):
    def __init__(self, model, output_index = 0):
        inps = [model.input]           # input placeholder
        outs = [layer.output for layer in model.layers]    # all layer outputs
        self.forward_pass = K.function(inps, outs)         # evaluation function
        
        self.model = model

    def get_mask(self, input_image):
        x_value = np.expand_dims(input_image, axis=0)
        
        visual_bpr = None
        layer_outs = self.forward_pass([x_value, 0])

        for i in range(len(self.model.layers) - 1, -1, -1):
            if 'Conv2D' in str(type(self.model.layers[i])):
                layer = np.mean(layer_outs[i], axis = 3, keepdims = True)
                layer = layer - np.min(layer)
                layer = layer / (np.max(layer) - np.min(layer) + 1e-6)

                if visual_bpr is not None:
                    if visual_bpr.shape != layer.shape:
                        visual_bpr = self._deconv(visual_bpr)
                    visual_bpr = visual_bpr * layer
                else:
                    visual_bpr = layer

        return visual_bpr[0]
    
    def _deconv(self, feature_map):
        x = Input(shape = (None, None, 1))
        y = Conv2DTranspose(filters = 1, 
                            kernel_size = (3, 3), 
                            strides = (2, 2), 
                            padding = 'same', 
                            kernel_initializer = Ones(), 
                            bias_initializer = Zeros())(x)

        deconv_model = Model(inputs=[x], outputs=[y])

        inps = [deconv_model.input]   # input placeholder                                
        outs = [deconv_model.layers[-1].output]           # output placeholder
        deconv_func = K.function(inps, outs)              # evaluation function
        
        return deconv_func([feature_map, 0])[0]
    
    
    ##  https://www.kaggle.com/ernie55ernie/mnist-with-keras-visualization-and-saliency-map
## All visualizing code used from here and adapted for program

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    #x *= 255
    #x = np.clip(x, 0, 255).astype('uint8')
    return x
    
    
def vis_img_in_filter(model, train_set, img_num, img_res, layer_num, GRAY=True):
    
    
    layer_names=get_layer_names(model)
    layer_name = '{}'.format(layer_names[layer_num])
    
    if GRAY == True:
        img = np.array(train_set[img_num]).reshape((1, img_res,img_res, 1)).astype(np.float64)
    else:
        img = np.array(train_set[img_num]).reshape((1, img_res,img_res, 3)).astype(np.float64)
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_output = layer_dict[layer_name].output
    img_ascs = list()
    
    for filter_index in range(layer_output.shape[3]):
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, model.input)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([model.input], [loss, grads])

        # step size for gradient ascent
        step = 5.

        img_asc = np.array(img)
        # run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([img_asc])
            img_asc += grads_value * step

        img_asc = img_asc[0]
        
        if GRAY == True:
            img_ascs.append(deprocess_image(img_asc).reshape((img_res,img_res)))
        else:
            img_ascs.append(deprocess_image(img_asc).reshape((img_res,img_res,3)))
            
    if layer_output.shape[3] >= 35:
        plot_x, plot_y = 6, 6
    elif layer_output.shape[3] >= 23:
        plot_x, plot_y = 4, 6
    elif layer_output.shape[3] >= 11:
        plot_x, plot_y = 3, 4
    elif layer_output.shape[3] >= 7:
        plot_x, plot_y = 2, 4
    else:
        plot_x, plot_y = 2, 2
    
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (15, 15))
    
    if GRAY == True:
        ax[0, 0].imshow(img.reshape((img_res,img_res)), cmap = 'gray')
    else:
        ax[0, 0].imshow(img.reshape((img_res,img_res,3)))
    
    ax[0, 0].set_title('Input image')
    fig.suptitle('Input image and %s filters' % (layer_name,))
    fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        if x == 0 and y == 0:
            continue
        ax[x, y].imshow(img_ascs[x * plot_y + y - 1], cmap = 'gray')
        
        ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))
        
        
def get_layer_names(model, show=False):
    """Get layer names and display back to user"""
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    i=0
    layer_names = []
    
    for layers in layer_dict:
        if 'flatten' in layers:
            break
        if show == True:
            print(i, ':', layers)
        layer_names.append(layers)
        i += 1
    return layer_names



def activation_maps(model, test_set, test_label_set, img_res, show_img=4, GRAY=True):

    Y_train_label = test_label_set
    
    fig, ax = plt.subplots(show_img, 4, figsize = (20, 20))

    for i in range(show_img):
        img = np.array(test_set[i])
        fig.subplots_adjust(left=.01,
                        right=.8,
                        bottom=0.1,
                        top=1.2,
                        wspace=0.1,
                        hspace=.4
                       )

        vanilla = GradientSaliency(model, Y_train_label[i])
        
        mask = vanilla.get_mask(img)
        if GRAY == True:
            filter_mask = (mask > 0.0).reshape((img_res,img_res))
        else:
            filter_mask = (mask > 0.0).reshape((img_res,img_res,3))
            
        smooth_mask = vanilla.get_smoothed_mask(img)
        if GRAY == True:
            filter_smoothed_mask = (smooth_mask > 0.0).reshape((img_res,img_res))
        else:
            filter_smoothed_mask = (smooth_mask > 0.0).reshape((img_res,img_res,3))
        
        if GRAY == True:
            ax[i, 0].imshow(img.reshape((img_res,img_res)), cmap = 'gray')
            cax = ax[i, 1].imshow(mask.reshape((img_res,img_res)), cmap = 'jet')
            fig.colorbar(cax, ax = ax[i, 1])
            ax[i, 1].imshow(mask.reshape((img_res,img_res)) * filter_mask, cmap = 'jet')
            cax = ax[i, 2].imshow(mask.reshape((img_res,img_res)), cmap = 'gray')
            fig.colorbar(cax, ax = ax[i, 3])
            ax[i, 3].imshow(smooth_mask.reshape((img_res,img_res)) * filter_smoothed_mask, cmap = 'jet')
            
        else:    
            ax[i, 0].imshow(img.reshape((img_res,img_res,3)), cmap = 'gray')
            cax = ax[i, 1].imshow(mask.reshape((img_res,img_res,3)), cmap = 'jet')
            fig.colorbar(cax, ax = ax[i, 1])
            ax[i, 1].imshow(mask.reshape((img_res,img_res,3)) * filter_mask, cmap = 'jet')
            cax = ax[i, 2].imshow(mask.reshape((img_res,img_res,3)), cmap = 'gray')
            fig.colorbar(cax, ax = ax[i, 3])
            ax[i, 3].imshow(smooth_mask.reshape((img_res,img_res,3)) * filter_smoothed_mask, cmap = 'jet')
        
        
        