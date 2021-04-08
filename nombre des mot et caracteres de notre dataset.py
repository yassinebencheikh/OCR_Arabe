# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:14:44 2020

@author: Abdellah-Bencheikh
"""

import cv2 as cv
import os
from glob import glob
from tqdm import tqdm


img_paths = glob('..\\Dataset\\scan/*.jpg')
txt_paths = glob('..\\Dataset\\texte/*.txt')

script_path = os.getcwd()

directory = {}

chiffre = ['0','1','2','3','4','5','6','7','8','9']
chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق','ك', 'ل', 'م', 'ن', 'ه', 'و','ي','ء', 'أ', 'إ', 'آ','ة', 'ؤ','ئ','ى','لا','لأ','لإ','لآ','0','1','2','3','4','5','6','7','8','9']
for char in chars:
    directory[char] = 0

# """ vérifier si le caractère est 'لا' return True """
def check_lamAlf(word, idx):
    if idx != len(word)-1 and word[idx] == 'ل':
        if word[idx+1] == 'ا' or word[idx+1] == 'أ' or word[idx+1] == 'إ' or word[idx+1] == 'آ':
            return True
        
    return False

""" lire les caractères du mot"""
def get_word_chars(word):
    i = 0
    chars = []
    while i < len(word):
        if check_lamAlf(word, i):
            chars.append(word[i:i+2])
            i += 2
        else:
            chars.append(word[i])
        i += 1
    return chars

"""preparation dataset"""
def prepare_dataset(limit=1000):
    print("Processing Images")
    error = 0
    cro = 0
    nb_mot = 0
    nb_char = 0
    for img_path, txt_path in tqdm(zip(img_paths[:limit], txt_paths[:limit]), total=len(img_paths)):
        assert(img_path.split('\\')[-1].split('.')[0] == txt_path.split('\\')[-1].split('.')[0])
        
        # Texte Partie de l'image
        with open(txt_path, 'r', encoding='utf8') as fin:
            lines = fin.readlines()
            line = lines[0].rstrip()
            txt_words = line.split()
            
        for txt_word in txt_words:
            nb_mot += 1
            # Obtenez les caractères du texte
            txt_chars = get_word_chars(txt_word)
            for txt_char in txt_chars:
                nb_char += 1
                
    print('le nombre des mots = ', nb_mot)
    print('le nombre des caracteres = ', nb_char)
    


if __name__ == "__main__":
    prepare_dataset()
    
    


    