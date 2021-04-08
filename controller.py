# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 09:18:24 2020

@author: Abdellah-Bencheikh
"""

import cv2 as cv
from OCR import run
from PIL import Image as Imga
from PIL import ImageTk
from tkinter.filedialog import askdirectory, askopenfilename
from fram import OCR_IHM
import tkinter as tk

global image,predicted_text, img_name

# définition de classe 
class OCR_arabe(OCR_IHM) :
    def __init__(self):
        """Constructeur de l'IHM."""
        super().__init__()
    # methode de convertir l'image
    def actionConverti(self) :
        self.texte.delete(1.0, tk.END)
        global predicted_text
        predicted_text = run(image)
        self.texte.insert(tk.INSERT, predicted_text)
    # methode de recherche l'image dans un repertoire
    def actionRechercher(self) :
        global image 
        global img_name
        filepath = askopenfilename(title="Ouvrir une image")
        if len(filepath)>0:
            img=cv.imread(filepath)
            img_name = filepath.split('/')[5].split('.')[0]
            image  = img
            #w=350
            dim = tuple((400, 200))
            #char_img = binarize(char_img)
            img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
            #img = cv.resize(img, (w,int((w/img.shape[1])*img.shape[0])), 0, 0, cv.INTER_NEAREST )
            img_RGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            #Affichage image originale
            img_RGB = Imga.fromarray(img_RGB)
            img_RGB = ImageTk.PhotoImage(img_RGB)
            self.Cadre.configure(image=img_RGB)
            self.Cadre.image=img_RGB
    # methode d'enregistrer l'image
    def actionSave(self) :
        global predicted_text
        #file = open(f'{img_name}.txt', 'w')
        saveHere = askdirectory(initialdir='/', title='Select File')
        #file.write(os.path.join(saveHere, f'{img_name}.text'))
        with open(f'{saveHere}/{img_name}.txt', 'w', encoding='utf8') as fo:
            fo.writelines(predicted_text)
        print('save')

# programme principal 
# instancie l'application
app = OCR_arabe() 