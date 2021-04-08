# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:22:48 2020

@author: Abdellah-Bencheikh
"""

import numpy as np
import cv2
import os
import warnings
import Augmentor 
from PIL import Image
warnings.filterwarnings("ignore")
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 

def augmentation(path_image,save_to_dir,n_sample,save_prefix ='image', save_format ='png'):
    # Initialisation de la classe ImageDataGenerator. 
    # Nous passerons les paramètres d'augmentation dans le constructeur.
    datagen = ImageDataGenerator( rotation_range = 0.1 , width_shift_range = 0.1 , 
                                 height_shift_range = 0.1 ,zoom_range = 0.0 , 
                                 horizontal_flip = False , vertical_flip = False)
    # Chargement d'un exemple d'image 
    img = load_img(path_image) 
    # Conversion de l'image échantillon d'entrée en un tableau 
    x = img_to_array(img) 
    # Remodelage de l'image d'entrée 
    x = x.reshape((1, ) + x.shape) 
    # Génération et sauvegarde de 5 échantillons augmentés en utilisant les paramètres définis ci-dessus.
    i = 0
    for batch in datagen.flow(x, batch_size = 1, save_to_dir =save_to_dir,save_prefix ='image', save_format ='png'): 
        i += 1
        if i > n_sample: 
            break

def augmented(input_dir,output_dir):
    i = 0
    # Parcourez chaque class de l'ensemble d'entraînement
    folders=os.listdir(input_dir)
    for class_dir in folders:
        if not os.path.isdir(os.path.join(input_dir, class_dir)):
            continue
        # Parcourez chaque image d'entraînement pour la class actuelle
        for img_path in  os.listdir(os.path.join(input_dir, class_dir)):
            output = os.path.join(output_dir, class_dir)
            if not os.path.exists(output):
                os.makedirs(output)
            augmentation(img_path,output ,12, save_prefix =class_dir, save_format ='png')
            i +=1

output_dir = "Data_Augmente"
input_dir = "chars0"
augmented(input_dir,output_dir)


# Autre Méthode
def augmented2(input_dir,n_samples, output_directory="output"):
    # Parcourez chaque image de l'ensemble d'entraînement
    folders=os.listdir(input_dir)
    for class_dir in folders:
        print(class_dir)
        folder=os.path.join(input_dir, class_dir)
        if not os.path.isdir(folder):
            continue
        name_images=os.listdir(os.path.join(input_dir, class_dir))
        N=len(name_images)
        # La classe Pipeline gère la création de pipelines d'extension et la
        #génération de données augmentées en appliquant des opérations à ce pipeline.
        p = Augmentor.Pipeline(folder,output_directory=output_directory) 
        # Définition des paramètres d'augmentation et génération de 5 échantillons
        #p.flip_random(0.0)
        #p.flip_left_right(0.01)
        #p.flip_top_bottom(0.01)
        #p.black_and_white(0.1) 
        #p.rotate(0.0, 0, 0) 
        #p.skew(0.4, 0.5) 
        p.zoom(probability = 0.01, min_factor = 1.0, max_factor = 1.1) 
        p.sample(N*n_samples) 


directory= "test"
augmented2(directory,15)


