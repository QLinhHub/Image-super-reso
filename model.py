#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 23:30:21 2020

@author: quanglinhle
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def conv2dbase(x, filters=96, kernel_size=3, strides=1, padding='same', alpha=0.2):
    y = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=alpha)(y)

    return y

def denseblock(x, filters=96, k=3):
    for i in range(k):
        x_temp = conv2dbase(x, filters=filters, kernel_size=3)
        x = Concatenate(axis=-1)([x, x_temp])
    
    # ghi may cai except khi nhap k = 0 do cac kieu
    #do this for higher parameter and computational effciciency    
    if(k > 3):
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = conv2dbase(x, filters=filters*4, kernel_size=1)
        
    x = conv2dbase(x, filters=filters, kernel_size=3)
        
    return x

def rdb(x, filters=96, kr=3):
    #y = conv2dbase(x, filters=64, kernel_size=3)
    y = denseblock(x, filters=filters, k=kr)
    y = Concatenate(axis=-1)([y, x])
    y = conv2dbase(y, filters=filters, kernel_size=3)
    y = Add()([y, x])
    
    return y

def SRmodel(ks=3, kr=3, filters=96, lr=0.01, decay=1e-4):
    resb = []
    inputs = Input(shape=(128, 128, 3))
    x = conv2dbase(inputs, filters=filters, kernel_size=3)
    
    resb.append(x)
    for i in range(ks):
        resb.append(rdb(resb[i], kr=kr, filters=filters))
    # ghi may cai except khi nhap k = 0 do cac kieu
    resb = resb[1:]
    concat = Concatenate(axis=-1)(resb)
    
    y = conv2dbase(concat, filters=filters, kernel_size=1)
    y = conv2dbase(concat, filters=filters, kernel_size=3)
    
    y = Add()([y, x])
    y = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(y)
    #y = conv2dbase(y, filters=32, kernel_size=1)
    
    outputs = conv2dbase(y, filters=3, kernel_size=3, alpha=0.0)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    
    model.compile(optimizer=Adam(lr=lr, decay=decay), loss='mse')
    
    return model    
    
    
    
    
    
    
    
    
    
    
    
