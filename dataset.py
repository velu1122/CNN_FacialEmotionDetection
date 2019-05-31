# -*- coding: utf-8 -*-
"""
@author: velmurugan.jeyaram
"""

import cv2
import glob
import numpy as np
import random
import os
import utils

class EmotionData():
    
    def __init__(self):
        self.i = 0
        self.training_images = None
        self.training_labels = None      
        self.test_images = None
        self.test_labels = None
    
    def get_files(self, emotion): #Define function to get file list, randomly shuffle it and split 80/20
        files = glob.glob("SampleDataset\\%s\\*" %emotion)
        random.shuffle(files)
        training = files[:int(len(files)*0.8)] #get first 80% of file list
        prediction = files[-int(len(files)*0.2):] #get last 20% of file list
        return training, prediction
    
    def one_hot_encode(self, vec, vals=5):
        '''
        For use to one-hot encode 5- possible labels
        '''
        n = len(vec)
        out = np.zeros((n, vals))
        out[range(n), vec] = 1
        return out
    
    def make_sets(self):
        train_data = []
        train_labels = []
        val_data = []
        val_labels = []
        for emotion in utils.emotions:
            print(" working on %s" %emotion)
            training, validation = self.get_files(emotion)
            #Append data to training and prediction list, and generate labels 0-7
            for item in training:
                try:
                    image = cv2.imread(item) #open image
                    newimg = cv2.resize(image, (int(32), int(32)))
                    rgb = cv2.cvtColor(newimg, cv2.IMREAD_COLOR) #convert to grayscale
                    rgb_imge = np.array(rgb)
                    train_data.append(rgb_imge) #append image array to training data list
                    train_labels.append(utils.emotions.index(emotion))
                except Exception:
                    os.remove(item)
                    print("Invalid file " + str(item)+ " removed")
            for item in validation:
                try:
                    image = cv2.imread(item)
                    newimg = cv2.resize(image, (int(32), int(32)))
                    rgb = cv2.cvtColor(newimg, cv2.IMREAD_COLOR)
                    rgb_imge = np.array(rgb)
                    val_data.append(rgb_imge)
                    val_labels.append(utils.emotions.index(emotion))
                except Exception:
                    os.remove(item)
                    print("Invalid file " + str(item)+ " removed")
        self.training_images = np.asarray(train_data)/255
        self.training_labels = self.one_hot_encode(train_labels,5)
        self.test_images = np.asarray(val_data)/255
        self.test_labels = self.one_hot_encode(val_labels,5)        
        #return training_data, training_labels, prediction_data, prediction_labels
       
    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i+batch_size].reshape(-1,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = self.i + batch_size
        return x, y