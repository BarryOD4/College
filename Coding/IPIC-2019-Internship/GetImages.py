### Libaries needed
import os
import cv2
import numpy as np

def get_images(images, dimension, GRAY=True, data='OSNR'):
    ## If `error: (-215) scn == 3 || scn == 4` occurs
    ##  check if there is a ipynb checkpoint in file
    
    """
    Takes two inputs:       file-directory and x-y lengths (square)
    
    Returns two arrays:     X is an array of scaled images
                            Y is an array of labels
            
    Takes images from file, reads then processes, and returns
    them as square numpy arrays for easier reading in ConvNet.
    
    # Arguments
        images:- list of image directories
        dimension:- length of square side of scaled image
            e.g., dimension = 64   |==>  image resolution of 64x64
        GRAY:- Boolean. Choose between RGB channel and grayscale
    """
    
    X, y = [], []
    dim_xy = (dimension, dimension)
    
    for image in images:
        
        ## Converts image to array of values, converts image to grey-scale
        ##  then scales them to square image
        rImage = cv2.imread(image)
        if GRAY == True:
            colr_rImage = cv2.cvtColor(rImage, cv2.COLOR_BGR2GRAY)
        else:
            colr_rImage = rImage
        
        X.append(cv2.resize(colr_rImage,
                            dim_xy,
                            interpolation=cv2.INTER_CUBIC,))
        try:
            if data == 'disp':
                ## Collects labels for each image
                j = image.split('_')
                y.append(j[5])

            if data == 'disp-short':
                ## Collects labels for each image
                j = image.split('_')
                y.append(int(int(j[5])//100))

            if data == 'OSNR':
                ## Collects labels for each image
                if 'p5' in image:
                    y.append(image[-9:-7] + '.5')
                else:
                    y.append(image[-7:-5])
        except:
            print("""Please choose a 'data' of;
                     'OSNR' - Optical Signal-to-Noise Ratio
                     'disp'- The dispersion of an eye-diagram
                     'disp-short' - A smaller set of dispersion""")
            
        
    X = np.array(X)
        
    if X.shape == (len(X), dimension, dimension):
        X = np.expand_dims(X, axis=3)
        

    return X,y