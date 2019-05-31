# -*- coding: utf-8 -*-
"""
@author: velmurugan.jeyaram
"""
import cv2
import glob
import numpy as np
import os
import tensorflow as tf
import utils

PredictPath = os.path.join("Predict\\",'*')

print("*****Prediction starts*****")

# Inference for Images
def get_to_prediction_files():
    predict_data=[]
    fileNameList=[]
    predict_images = None
    toPredictFiles = glob.glob(PredictPath)
    for item in toPredictFiles:
        image = cv2.imread(item) #open image
        fileNameList.append(os.path.basename(item)) # to get only file names
        newimg = cv2.resize(image, (int(32), int(32)))
        rgb = cv2.cvtColor(newimg, cv2.IMREAD_COLOR) #convert to grayscale
        predict_data.append(np.array(rgb))
        predict_images = np.asarray(predict_data)/255
    return (predict_images, fileNameList)

# Load the tensoflow Model pb file
with tf.Session(graph=tf.Graph()) as sess:
    imgFiles, fileNames = get_to_prediction_files()
    
    tf.saved_model.loader.load(sess, ["Emotions"], "./Savedmodel/model/")
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    hold_prob = graph.get_tensor_by_name("hold_prob:0")
    model = graph.get_tensor_by_name("add_3:0")
    
    classification = sess.run(model, feed_dict = {x: imgFiles, hold_prob:1.0})
    
    print('The Prediction for the {} images::'.format(len(imgFiles)))
    for index,item in enumerate(classification):
        #print(item)
        print(fileNames[index]+":"+utils.emotions[np.argmax(item)])

'''
with tf.Session() as sessRestore:
    #sessRestore.run(tf.global_variables_initializer())
    
    #Get input np array to inference and filenames of the prediction images
    imgFiles, fileNames = get_to_prediction_files()

    saverRestore = tf.train.Saver()
    saverRestore.restore(sessRestore, "./Savedmodel/Emotiondetection.ckpt")
    print("Model restored.")
    

    classification = sessRestore.run(y_pred, feed_dict = {x: imgFiles, hold_prob:1.0})

    #Alternate code to print the inference
    #temp = tf.argmax(classification,1)
    #print(sessRestore.run(temp))
    #print(fileNames[index]+":"+emotions[temp[index].eval()])

'''