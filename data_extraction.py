#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:11:03 2020

@author: quanglinhle
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from os import listdir
from os.path import join, isfile
import cv2

def data_extraction(path='./train_data/', size=256):
    files = [path + img_f for img_f in listdir(path) if isfile(join(path, img_f))]
    data = []
    exception_count = 1
    for f in files:
        try:
            img = np.asarray(Image.open(f))
            nx, ny = img.shape[0]//size, img.shape[1]//size
            for i in range(nx):
                for j in range(ny):
                    extracted_img = np.reshape(img[size*i:size*(i+1), size*j:size*(j+1), :], (1,size,size,3))
                    data.append(extracted_img)
        except:
            print("exception" ,exception_count, ":", f)
            exception_count +=1
            continue
    
    data = np.concatenate(data, axis=0)
    np.random.shuffle(data)
    
    #HR ground truth train/val/test images
    HR_img = data.astype('float32')/255.
    #HR_img_test = data[3540:, :, :, :].astype('float32')/255.
    
    #LR train/val/test images
    #use maxpool2d of tensorflow to generate LR images from HR images
    data = tf.convert_to_tensor(data)
    data = tf.nn.max_pool(data, ksize=(2,2), strides=2, padding='VALID')
    data = np.asarray(data)
    
    LR_img = data.astype('float32')/255.
    
    return HR_img, LR_img

def data_downsample(path='./train_data/', size=256):
    files = [path + img_f for img_f in listdir(path) if isfile(join(path, img_f))]
    data = []
    exception_count = 1
    for f in files:
        try:
            img = np.asarray(Image.open(f))
            img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
            img = np.reshape(img, (1,size,size,3))
            data.append(img)
        except:
            print("exception", exception_count, ":", f)
            exception_count +=1
            continue
    data = np.concatenate(data, axis=0)
    
    #HR ground truth train/val/test images
    HR_img = data.astype('float32')/255.
    
    #LR train/val/test images
    #use maxpool2d of tensorflow to generate LR images from HR images
    data = tf.convert_to_tensor(data)
    data = tf.nn.max_pool(data, ksize=(2,2), strides=2, padding='VALID')
    data = np.asarray(data)
    
    LR_img = data.astype('float32')/255.
    return HR_img, LR_img

        
        
        
        
        
        
        
        
        
        