# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 23:51:59 2024

@author: karthikeya_sk
"""

import numpy as np
import h5py
from skimage import io


# read features database (h5 file)
h5f = h5py.File("CustomFeatures.h5",'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()
        
#Read the query image
queryImg = io.imread("C:/Users/Lenovo/OneDrive/Desktop/query_images/histo.jpg") #histo.jpg, tiger3.jpg

#Worked great on histo image as color information helps.
#But results are not good for tiger3

print(" searching for similar images")


from index_using_custom_features import extract_custom_features

#Extract Features
X = extract_custom_features(queryImg)


# Compute the Cosine distance between 1-D arrays
# https://en.wikipedia.org/wiki/Cosine_similarity

scores = []

from scipy import spatial

for i in range(feats.shape[0]):
    score = 1-spatial.distance.cosine(X, feats[i])
    scores.append(score)
scores = np.array(scores)   
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]


# Top 3 matches to the query image
max_num_matches = 3
imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:max_num_matches])]
print("top %d images in order are: " %max_num_matches, imlist)