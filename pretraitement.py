# -*- coding: utf-8 -*-
"""
Created on Mon Mai  20 23:08:04 2020

@author: Abdellah-Bencheikh
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation as inter
from PIL import Image as im


""" binarisation otsus"""
def binary_otsus(image, filter:int=1):
    # Binarize an image 0's and 255's using Otsu's Binarization
    if len(image.shape) == 3:
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_img = image
    # Otsus Binarization
    if filter != 0:
        blur = cv.GaussianBlur(gray_img, (3,3), 0)
        binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    else:
        binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        #_, binary_img = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
    return binary_img


""" Détecter et corriger l'inclinaison de l'image  : Corriger le biais d'image de sorte que le texte soit horizontal et non à un angle."""
def determine_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def correct_skew(binary_img):
    ht, wd = binary_img.shape
    bin_img = (binary_img // 255.0)
    delta = 0.1
    limit = 3
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = determine_score(bin_img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    # corriger l'inclinaison
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img = im.fromarray((255 * data).astype("uint8"))
    pix = np.array(img)
    return pix

""" Projection horisontale et verticale """ 
def projection(gray_img, axis:str='horizontal'):
    #Calculer la projection horizontale ou verticale d'une image grise 
    if axis == 'horizontal':
        # la some de chaque ligne
        projection_bins = np.sum(gray_img, 1).astype('int32')
    elif axis == 'vertical':
        # la some de chaque colonne
        projection_bins = np.sum(gray_img, 0).astype('int32')
    return projection_bins


""" Enregister l'image """ 
def save_image(img, folder, title):
    cv.imwrite('Figures/'+folder+'/'+title+'.png', img)

""" Afficher l'image """
def show_image(gray_img):
    plt.imshow(gray_img, 'gray')
    plt.show()

