# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:12:49 2020

@author: Abdellah-Bencheikh
"""


from __future__ import print_function

import os, editdistance

PREDICTED_PATH = 'train_prediction\\text'
TRUTH_PATH = '..\\dataset\\text'


distances = []
accuracies = []

for file_name in os.listdir(PREDICTED_PATH):
    with open(os.path.join(PREDICTED_PATH, file_name), encoding='utf8') as f:
        predicted = ''.join(f.read().split())
    with open(os.path.join(TRUTH_PATH, file_name), encoding='utf8') as f:
        truth = ''.join(f.read().split())
    # calculer les erreurs des caracteres
    distance = editdistance.eval(predicted, truth)
    distances.append(distance)
    accuracie = max(0, 1 - distance / len(truth))
    accuracies.append(accuracie)
    print(f'{file_name}: {distance}, {accuracie}')

print(f'Distance totale = {sum(distances)}')
print('Précision moyenne = %.2f%%' % (sum(accuracies) / len(accuracies) * 100))




#####################################################################################


""" calculer l'erreur et accuracy pour les mots """
PREDICTED_PATH = 'train_prediction\\text'
TRUTH_PATH = '..\\dataset\\text'

nb_word = []
distances = []
accuracies = []

for file_name in os.listdir(PREDICTED_PATH):
    with open(os.path.join(PREDICTED_PATH, file_name), encoding='utf8') as f:
        text = f.read()
        predicted = text.split()
    with open(os.path.join(TRUTH_PATH, file_name), encoding='utf8') as f:
        text = f.read()
        truth = text.split()
        # calculer nombre des mots du corpus
        nb_word.append(len(truth))
    # calculer les erreurs des caracteres
    distance = editdistance.eval(predicted, truth)
    distances.append(distance)
    accuracie = max(0, (len(predicted) - distance) / len(truth))
    accuracies.append(accuracie)
    print(f'{file_name}: {distance}, {accuracie}')

print(f'Distance totale = {sum(distances)}')
print('Précision moyenne = %.2f%%' % (sum(accuracies) / len(accuracies) * 100))
print(f'nombres des mots = {sum(nb_word)}')









