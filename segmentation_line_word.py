
# -*- coding: utf-8 -*-
"""
Created on Mon Mai  20 23:08:04 2020

@author: Abdellah-Bencheikh
"""
import numpy as np
import cv2 as cv
from pretraitement import binary_otsus, correct_skew, save_image


""" Preprocess de l'image """ 
def preprocess(image):
    # convertir image en gris.
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_img = cv.bitwise_not(gray_img)
    # binariser l'image
    binary_img = binary_otsus(gray_img, 0)
    # Détecter et corriger l'inclinaison de l'image
    deskewed_img = correct_skew(binary_img)
    return deskewed_img


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


def projection_segmentation(clean_img, axis, cut=3):
    segments = []
    start = -1
    cnt = 0
    projection_bins = projection(clean_img, axis)
    # comparaison entre les pixels de l'histogramme les pixel (vide ou non)
    for idx, projection_bin in enumerate(projection_bins):
        if projection_bin != 0:
            cnt = 0
        if projection_bin != 0 and start == -1:
            start = idx
        if projection_bin == 0 and start != -1:
            cnt += 1
            if cnt >= cut:
                if axis == 'horizontal':
                    # enregistrer les pixels qui contient la chaîne de caractère
                    segments.append(clean_img[max(start-1, 0):idx, :])
                elif axis == 'vertical':
                    # enregistrer les pixels qui contient le mot
                    segments.append(clean_img[:, max(start-1, 0):idx])
                cnt = 0
                start = -1
    return segments


""" Segmentation de ligne """ 
def line_horizontal_projection(image, cut=3):
    # Prétraiter l'image d'entrée
    clean_img = preprocess(image)
    # Segmentation    
    lines = projection_segmentation(clean_img, axis='horizontal', cut=cut)
    return lines


""" Segmentation de mots """
def word_vertical_projection(line_image, cut=4):
    line_words = projection_segmentation(line_image, axis='vertical', cut=cut)
    line_words.reverse()
    return line_words

"""Extraire la liste des lignes"""
def extract_lines(img, visual=0):
    lines = line_horizontal_projection(img)
    return lines


"""Extraire la liste des mots"""
def extract_words(img, visual=0):
    lines = line_horizontal_projection(img)
    words = []
    i = 0
    for idx, line in enumerate(lines):
        if visual:
            save_image(line, 'lines', f'line{idx}')
        line_words = word_vertical_projection(line)
        for w in line_words:
            words.append((w, line,i))
        i +=1
    if visual:
        for idx, word in enumerate(words):
            save_image(word[0], 'words', f'word{idx}')
    return words


if __name__ == "__main__":
    
    img = cv.imread('Figures/test1.jpg')
    extract_words(img,1)

# cv.imshow('clean_img',clean_img)
# cv.waitKey(0)