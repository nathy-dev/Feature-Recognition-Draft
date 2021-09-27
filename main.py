# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2

import os
import tensorflow as tf
import csv

# import the handfeature extractor class
import handshape_feature_extractor
import frameextractor as fe

# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
#file_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "FanDown": 10, "FanOn": 11,
             #"FanOff": 12, "FanUp": 13, "LightsOff": 14, "LightsOn": 15, "SetThermo": 16}

#File and feature extraction for traindata
BASE = os.path.dirname(os.path.abspath(__file__))
train_data = os.path.join(BASE, 'traindata')
dirs = os.listdir(train_data)
i = 0
frame_folder =  os.path.join(BASE, 'frame_folder')
hfe = handshape_feature_extractor.HandShapeFeatureExtractor()
results = []
for file in dirs:
    ext = os.path.splitext(file)[-1].lower()
    if ext == ".mp4":
        if dirs[i] is not None:
            full_path = os.path.join(train_data, file)
            fe.frameExtractor(full_path, frame_folder, i)
            try:
                img = cv2.imread(os.path.join(frame_folder, str(i+1).zfill(5) + '.png'), cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                print(str(e))
            results.append(hfe.extract_feature(img))
            i += 1


#
# # =============================================================================
# # Get the penultimate layer for test data
# # =============================================================================
# # your code goes here
# # Extract the middle frame of each gesture video

test_data = os.path.join(BASE, 'test')
test_dirs = os.listdir(test_data)
i = 0
test_folder =  os.path.join(BASE, 'test_folder')

test_results = []
for file in test_dirs:
    ext = os.path.splitext(file)[-1].lower()
    if ext == ".mp4":
        if dirs[i] is not None:
            full_path = os.path.join(test_data, file)
            fe.frameExtractor(full_path, test_folder, i)
            try:
                img = cv2.imread(os.path.join(test_folder, str(i+1).zfill(5) + '.png'), cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                print(str(e))
            test_results.append(hfe.extract_feature(img))
            i += 1

print(test_results)


# # =============================================================================
# # Recognize the gesture (use cosine similarity for comparing the vectors)
# # =============================================================================
output = []
cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)


l = 0
#result = cosine_loss(train, test)
for each in test_results:
    k = test_results[l]
    for each in results:
        n = results[l]
        cos_sim = cosine_loss(k,n)
        index_label = l
        if l <= 49:
            next_cos_sim = cosine_loss(k,results[l + 1])
            if next_cos_sim < cos_sim:
                cos_sim = next_cos_sim
                index_label += 1
    label = index_label % 17
    output.append(label)
    l += 1





NewList= [[x] for x in output]

result_file = open('Results.csv', 'w', newline='')

with result_file:
    writer = csv.writer(result_file)
    writer.writerows(NewList)




#file = open('Results.csv', 'w+', newline='')

#with file:
   # write = csv.writer(file)
    #write.writerows(output)
