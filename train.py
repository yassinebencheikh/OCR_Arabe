# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:14:44 2020

@author: Abdellah-Bencheikh
"""

import numpy as np
import os
import cv2 as cv
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import  ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.preprocessing import MultiLabelBinarizer


global path_data, current_path
path_data = 'Dataset\\chars'
current_path = os.getcwd()

"""Binarize"""
def binarize(char_img):
    #_, binary_img = cv.threshold(char_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    _, binary_img = cv.threshold(char_img, 127, 255, cv.THRESH_BINARY)
    return binary_img

""" préparation caractère"""
def prepare_char(char_img):
    dim = tuple((32, 32))
    #char_img = binarize(char_img)
    resized = cv.resize(char_img, dim, interpolation = cv.INTER_AREA)
    resized = np.array(resized)
    resized = resized /255
    return resized

def featurizer(char_img):
    char_img = np.reshape(char_img,[-1,32,32,1])
    return char_img

"""Read data"""
def read_data(path_data, current_path):
    #listes vides pour contenir les vecteurs d'entité et les étiquettes pour les donnée d'apprentissage"""
    X = []
    Y = []
    folders_train = os.listdir(path_data)
    for label in folders_train:
        folders_current = os.path.join(path_data, label)
        print(label)
        os.chdir(folders_current)
        current_label = label
        i = 900
        for x in os.listdir(os.getcwd()):
            if i<=0:
                break
            # lit l'image et la redimensionne à une taille fixe
            char = cv.imread(x)
            gray = cv.cvtColor(char, cv.COLOR_BGR2GRAY)  
            ready_char = prepare_char(gray)
            #ready_char = featurizer(ready_char)
            Y.append(current_label)
            X.append(ready_char)
            i -=1
        os.chdir(current_path)
    return X,Y

""" preparation data train"""
def prepare_data(path_data, current_path):
    X,Y = read_data(path_data, current_path)
    X, Y = shuffle(X, Y)
    X = np.array(X)
    X = np.asarray(X).reshape(X.shape[0],32,32,1).astype('float32')
    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0],-1)
    # changer les étiquettes au format d'encodage à chaud (28 entrées tous les zéros sauf un)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y)
    return X, Y, mlb

""" Modele"""
def Create_model():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),input_shape=(32,32,1),padding='same',activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,kernel_size=(5,5),padding='same',activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=(5,5),padding='same',activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512,activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(49,activation='softmax'))
    return model

""" Training  """
def train():
    # charger les données et les étiquettes de formation
    global path_data, current_path
    X, Y, mlb = prepare_data(path_data, current_path)
    data_train, data_val, Label_train, Label_val = train_test_split(X, Y, test_size = 0.20,random_state=2)
    # Create modele
    model = Create_model()
    # optimizer adam
    optimizer = Adam(lr=0.001, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    # Réduisez le taux d'apprentissage lorsqu'une mesure a cessé de s'améliorer
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.1, mode="min", min_lr=1e-9)
    epochs = 28
    batch_size = 128
    model.fit(data_train, Label_train, epochs=epochs, batch_size=batch_size, validation_data=(data_val, Label_val), callbacks=[reduce_lr])
    # Enregister le  Model 
    model.save('output/model.h5')
    output_Label = 'output/mlb.pickle'
    pickle_out = open(output_Label, "wb")
    pickle.dump(mlb, pickle_out)
    pickle_out.close()




if __name__ == "__main__":
    train()







# scores = model.evaluate(data_val,Label_val)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

# scores = model.evaluate(data_train,Label_train)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

# # Trcarer Précision VS époques Ou Loss VS epoques
# def plotAccuracy(history, mesure= 'accuracy'):  
#     plt.plot(history.history[mesure])
#     plt.plot(history.history['val_'+mesure])
#     plt.title('model '+mesure)
#     plt.ylabel(mesure)
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show() 


# plotAccuracy(history, 'accuracy')

# plotAccuracy(history, 'loss')