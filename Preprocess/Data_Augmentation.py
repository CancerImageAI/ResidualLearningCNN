# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:00:07 2019

@author: Jing Gong
@e-mail: gongjing1990@163.com
@Fudan University Shanghai Cancer Center
"""

import numpy as np
from skimage.util import random_noise
from skimage.transform import rotate
from skimage import exposure
from scipy import ndimage

def generator_class(prob=1):
    generator = np.random.choice([True, False],1,p=[prob, 1-prob])
    return generator

def img_noise(image,prob=1):
    generator = generator_class(prob)
    if generator:
        img = random_noise(image, mode = 'gaussian')
    else:
        img = []
    return img

def img_rotate(image,angle,prob=1):
    generator = generator_class(prob)
    if generator:
        img = rotate(image,angle)
    else:
        img = []
    return img

def img_flip_h(image,prob=1):
    generator = generator_class(prob)
    if generator:
        img = np.fliplr(image)
    else:
        img = []
    return img

def img_flip_v(image,prob=1):
    generator = generator_class(prob)
    if generator:
        img = np.flipud(image)
    else:
        img = []
    return img

def img_exposure_gamma(image,gamma,prob=1):
    generator = generator_class(prob)
    if generator:
        img = exposure.adjust_gamma(image,gamma)
    else:
        img = []
    return img

def img_exposure_log(image, log, prob=1):
    generator = generator_class(prob)
    if generator:
        img = exposure.adjust_log(image,log)
    else:
        img = []
    return img

def img_exposure_sigmoid(image, sigmoid, prob=1):
    generator = generator_class(prob)
    if generator:
        img = exposure.adjust_sigmoid(image,sigmoid)
    else:
        img = []
    return img

def img_augmentation(image,prob=1):
    img = {}
    img['image'] = image
    img['noise'] = img_noise(image, prob)
    img['rotate_90'] = img_rotate(image, 90, prob)
    img['rotate_180'] = img_rotate(image, 180, prob)
    img['rotate_270'] = img_rotate(image, 270, prob)
    img['flip_h'] = img_flip_h(image, prob)
    img['flip_v'] = img_flip_v(image, prob)
    img['gamma_0.8'] = img_exposure_gamma(image, 0.8, prob)
    img['gamma_1.2'] = img_exposure_gamma(image, 1.2, prob)
    img['log_0.8'] = img_exposure_log(image, 0.8, prob)
    img['log_1.2'] = img_exposure_log(image, 1.2, prob)
#    img['sigmoid_0.2'] = img_exposure_sigmoid(image, 0.1, prob)
#    img['sigmoid_0.4'] = img_exposure_sigmoid(image, 0.3, prob)
    return img
    
