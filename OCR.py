# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:12:49 2020

@author: Abdellah-Bencheikh
"""

import numpy as np
import cv2 as cv
import pickle
from segmentation_caractere import segment
from segmentation_line_word import extract_words
from tensorflow.keras.models import load_model
from textblob_ar import TextBlob
from textblob_ar.correction import TextCorrection

chiffre = ['0','1','2','3','4','5','6','7','8','9']
""" load model"""
global model
model = load_model('Output/model.h5')
global mlb_Label
path_mlb = 'Output/mlb.pk'
file = open(path_mlb,'rb')
mlb = pickle.load(file)

def prepare_char(char_img):
    dim = tuple((32, 32)) 
    resized = cv.resize(char_img, dim, interpolation = cv.INTER_AREA)
    resized = np.array(resized)
    resized = resized /255
    resized = np.reshape(resized,[-1,32,32,1])
    return resized

""" prediction caractère"""
def run2(obj):
    word, line,idx = obj
    # Pour chaque mot de l'image
    char_imgs = segment(line, word)
    txt_word = ''
    # Pour chaque caractère du mot
    for char_img in char_imgs:
        try:
            ready_char = prepare_char(char_img)
        except:
            continue
        proba = model.predict(ready_char)[0]
        ids = np.argsort(proba)[::-1][:1]
        predicted_char=str(mlb.classes_[ids[0]])
        #predicted_char = alph[ids[0]]
        txt_word += predicted_char
    # return prediction word
    if txt_word[0] in chiffre:
        txt_word = txt_word[::-1]
    # correcter le mot 
    #if predicted_char == 'ي' or predicted_char = 'ئ' or predicted_char ='ى':
    #    txt_word = TextCorrection().correction(txt_word, top=True)
    #txt_word = TextCorrection().correction(txt_word, top=True)
    return txt_word

""" prediction text """
def run(image):
    # Lire l'image de test
    full_image = image
    predicted_text = ''
    predicted_words = []
    words = extract_words(full_image)         # [ (word, its line,idl),(word, its line,idl),..  ]
    for idx, w in enumerate(words):
        word, line, idl = w
        if idx != 0 and words[idx-1][2] != idl:
            predicted_words.append('\n')
        predicted_word = run2(w)
        predicted_words.append(predicted_word)
    # append in the total string.
    for word in predicted_words:
        predicted_text += word
        predicted_text += ' '
    # return prediction text
    return predicted_text


if __name__ == "__main__":
    filepath = 'image_test/test1.jpg'
    img=cv.imread(filepath)
    text = run(img)
    print(text)
