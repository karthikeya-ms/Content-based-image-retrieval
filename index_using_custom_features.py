# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 21:25:19 2024

@author: karthikeya_sk
"""

import numpy as np
import pandas as pd
import os
from skimage import io, img_as_ubyte
from skimage.transform import resize
import h5py
from skimage.filters import roberts, sobel #to extract edges
from skimage.color import rgb2lab, rgb2gray, lab2rgb
import cv2
from skimage.io import imread
import matplotlib.pylab as plt

from skimage.filters.rank import entropy
from skimage.morphology import disk


def extract_custom_features(img):
    
    #Color features
    
    LAB_img = rgb2lab(img)
    A_img = LAB_img[:,:,1]
    A_feat = A_img.mean()
    
    B_img = LAB_img[:,:,2]
    B_feat = B_img.mean()
    
    #Textural features based on the gray image
    
    gray_img = rgb2gray(img)
    gray_img = resize(gray_img, (256,256)) #Resize to smaller size
    gray_img = img_as_ubyte(gray_img)
    
    entropy_img = entropy(gray_img, disk(3))
    entropy_mean = entropy_img.mean()
    entropy_std = entropy_img.std()
    
    roberts_img = roberts(gray_img)
    roberts_mean = roberts_img.mean()
    
    sobel_img = sobel(gray_img)
    sobel_mean = sobel_img.mean()
    
    #Gabor 1
    kernel1 = cv2.getGaborKernel((9, 9), 3, np.pi/4, np.pi, 0.5, 0, ktype=cv2.CV_32F)    
    gabor1 = (cv2.filter2D(gray_img, cv2.CV_8UC3, kernel1)).mean()
    
    #Gabor 2
    kernel2 = cv2.getGaborKernel((9, 9), 3, np.pi/2, np.pi/4, 0.9, 0, ktype=cv2.CV_32F)    
    gabor2 = (cv2.filter2D(gray_img, cv2.CV_8UC3, kernel2)).mean()
    
    #Gabor 3
    kernel3 = cv2.getGaborKernel((9, 9), 5, np.pi/2, np.pi/2, 0.1, 0, ktype=cv2.CV_32F)    
    gabor3 = (cv2.filter2D(gray_img, cv2.CV_8UC3, kernel3)).mean()
    
    custom_features = np.array([A_feat, B_feat, entropy_mean, entropy_std, roberts_mean, 
                                sobel_mean, gabor1, gabor2, gabor3])
    
    return custom_features

if __name__ == "__main__":
    
    path = "C:/Users/Lenovo/OneDrive/Desktop/all_images/"
    
    feats = []
    names = []
    
    for im in os.listdir(path):
        
        print("Extracting image features from image - ", im)
        img = io.imread(path+im)
        
        #Extract images
        X = extract_custom_features(img)
        feats.append(X)
        names.append(im)
        
    
    feats = np.array(feats)
    
    feature_file = "Customfeatures.h5"
    print("Saving features to h5 file")
    
    
    
    h5f = h5py.File(feature_file, 'w')
    h5f.create_dataset('dataset_1', data = feats)
    h5f.create_dataset('dataset_2', data = np.string_(names))
    h5f.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
