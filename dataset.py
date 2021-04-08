# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:14:44 2020

@author: Abdellah-Bencheikh
"""

import cv2 as cv
import os
from segmentation_line_word import extract_words
from segmentation_caractere import segment
from glob import glob
from tqdm import tqdm


img_paths = glob('..\\Dataset\\scanned/*.jpg')
txt_paths = glob('..\\Dataset\\text/*.txt')

script_path = os.getcwd()
width = 32
height = 32
dim = (width, height)
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
    for img_path, txt_path in tqdm(zip(img_paths[:limit], txt_paths[:limit]), total=len(img_paths)):
        assert(img_path.split('\\')[-1].split('.')[0] == txt_path.split('\\')[-1].split('.')[0])
        
        # Texte Partie de l'image
        with open(txt_path, 'r', encoding='utf8') as fin:
            lines = fin.readlines()
            line = lines[0].rstrip()
            txt_words = line.split()
    
        # Partie d'image
        img = cv.imread(img_path)
        img_words = extract_words(img)
        # Obtenez les mots pour l'image
        for img_word, txt_word in tqdm(zip(img_words, txt_words), total=len(txt_words)):
            
            # Obtenez les caractères du texte
            txt_chars = get_word_chars(txt_word)
            if txt_chars[0] in chiffre:
                txt_chars = txt_chars[::-1]
            # Obtenez les caractères de l'image
            line = img_word[1]
            word = img_word[0]
            img_chars = segment(line, word)
            
            if len(txt_chars) == len(img_chars):
                for img_char, txt_char in zip(img_chars, txt_chars):
                    
                    number = directory[txt_char]+1
                    destination = f'../Dataset/chars/{txt_char}'
                    if not os.path.exists(destination):
                        os.makedirs(destination)
                    os.chdir(destination)
                    cv.imwrite(f'{number}.png', img_char)
                    os.chdir(script_path)
                    directory[txt_char] += 1
                    cro += 1
            else:
                error += 1
    print('le taux de segmentation des caractère = ', cro*100/(cro+error))
    print('\nDone')
    


if __name__ == "__main__":
    prepare_dataset()
    
    


    