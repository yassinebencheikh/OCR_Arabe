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
from pretraitement import save_image


"""preparation dataset"""
def prepare_dataset(path_dir):
    liste_chars = []
    folders=os.listdir(path_dir)
    for im in folders:
        file=os.path.join(path_dir, im)
        img = cv.imread(file)
        words = extract_words(img)
        for idx, w in enumerate(words):
            word, line,idl = w
            char_imgs = segment(line, word)
            for char_img in char_imgs:
                liste_chars.append(char_img)
    for idx, c in enumerate(liste_chars):
        save_image(c, 'caracteres', f'caractere{idx}')


if __name__ == "__main__":
    img_paths = 'Figures\\scanned'
    prepare_dataset(img_paths)
    


