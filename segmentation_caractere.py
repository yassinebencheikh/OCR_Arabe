# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:01:48 2020

@author: Abdellah-Bencheikh
"""

import numpy as np
import cv2 as cv
from pretraitement import save_image
from segmentation_line_word import line_horizontal_projection, word_vertical_projection, projection
from skimage.morphology import skeletonize#, thin


"""  supprimer le petit objet connecté dans l'image.(les points, les virgules ...) """
def remove_dots(word_img, threshold=11):
    no_dots = word_img.copy()
    no_dots = np.uint8(no_dots)
    components, labels, stats, GoCs = cv.connectedComponentsWithStats(no_dots, connectivity=8, ltype = cv.CV_32S)
    char = []
    for label in range(1, components):
        _, _, _, _, size = stats[label]
        if size > threshold:
            char.append(label)
    for label in range(1, components):
        _, _, _, _, size = stats[label]
        if label not in  char:
            no_dots[labels == label] = 0
    return no_dots
            

""" remplir les petits trous """
def fill(binary_word, VP):
    (h, w) = binary_word.shape
    flag = 1
    while flag:
        flag = 0
        for row in range(h-1):
            for col in range(1, w-1):
                if binary_word[row][col] == 0 and binary_word[row][col-1] == 1 and binary_word[row][col+1] == 1 and binary_word[row+1][col] == 1 and VP[col] != 0:
                    binary_word[row][col] = 1
                    #flag = 1
    return binary_word

""" Detection ligne de base (ligne horizontal) """
def baseline_detection(word):
    word = word/255
    HP = []
    # image Amincissement (skeletonize)
    #thinLine = thin(word)
    #HP =  projection(thinLine, 'horizontal')
    HP =  projection(word, 'horizontal')
    BaseLineIndex = 0
    mx = 0
    i = 1
    while i<len(HP):
        if HP[i]>mx:
            mx = HP[i]
            BaseLineIndex = i  # index(i)
        i += 1
    VP = projection(word, 'vertical')
    VP = [x for x in VP if x!= 0]
    (values,counts) = np.unique(VP,return_counts=True)
    ind=np.argmax(counts)
    MFV = values[ind]
    return BaseLineIndex, MFV


""" Trouver index de transition maximale """
def horizontal_transitions(word_ig, baseline_idx):
    #word_ig = word_ig/255
    max_transitions = 0
    max_transitions_idx = baseline_idx
    line_idx = baseline_idx
    # nouvelle image temporaire sans points au-dessus de la ligne de base
    while line_idx >= 0:
        current_transitions = 0
        flag = 0
        horizontal_line = word_ig[line_idx, :]
        for pixel in reversed(horizontal_line):
            if pixel == 1 and flag == 0:
                current_transitions += 1
                flag = 1
            elif pixel ==0 and flag == 1:
                flag = 0
        if current_transitions >= max_transitions:
            max_transitions = current_transitions
            max_transitions_idx = line_idx
        line_idx -= 1
    return max_transitions_idx


def vertical_transitions(word_img, cut):
    #word_img = word_img /255
    transitions = 0
    vertical_line = word_img[:, cut]
    flag = 0
    for pixel in vertical_line:
        if pixel == 1 and flag == 0:
            transitions += 1
            flag = 1
        elif pixel == 0 and flag == 1:
            transitions += 1
            flag = 0
    return transitions


""" Identification toutes les régions de séparation """
def cut_points(word_img, VP, MFV, MTI, baseline_idx):
    # drapeau pour connaître le début du mot
    f = 0
    flag = 0
    (h, w) = word_img.shape
    i = w-1
    separation_regions = []
    wrong = 0
    # boucle sur la largeur de l'image de droite à gauche
    while i >= 0:
        pixel = word_img[MTI, i]
        if pixel == 1 and f == 0:
            f = 1
            flag = 1
        if f == 1:
            # Obtenez le début et la fin de la région de séparation (les deux sont des pixels noirs <----)
            if pixel == 0 and flag == 1:
                start = i+1
                flag = 0
            elif pixel == 1 and flag == 0:
                end = i        
                flag = 1
                mid = (start + end) // 2
                left_zero = -1
                left_MFV = -1
                right_zero = -1
                right_MFV = -1
                # seuil pour MFV
                T = 0 # 0
                j = mid - 1
                # boucle du milieu à la fin pour obtenir VP = 0 et VP = MFV les plus proches
                while j >= end:
                    if VP[j] == 0 and left_zero == -1:
                        left_zero = j
                    if VP[j] <= MFV + T and left_MFV == -1:
                        left_MFV = j
                    j -= 1
                j = mid
                # boucle du milieu au début pour obtenir VP = 0 et VP = MFV les plus proches
                while j <= start:
                    if VP[j] == 0 and right_zero == -1:
                        right_zero = j
                    if VP[j] <= MFV + T and right_MFV == -1:
                        right_MFV = j
                    if right_zero != -1 and right_MFV != -1:
                        break
                    j += 1
                # Vérifier d'abord VP = 0
                if VP[mid] == 0:
                    cut_index = mid
                elif left_zero != -1 and right_zero != -1:
                    if abs(left_zero-mid) <= abs(right_zero-mid):
                        cut_index = left_zero
                    else:
                        cut_index = right_zero
                elif left_zero != -1:
                    cut_index = left_zero
                elif right_zero != -1:
                    cut_index = right_zero
                elif left_MFV != -1:
                    cut_index = left_MFV
                elif right_MFV != -1:
                    cut_index = right_MFV
                else:
                    cut_index = mid
                # seg : la région de séparation actuelle
                seg = word_img[:, end:start]
                # HP : projection horizontale de seg
                HP = projection(seg, 'horizontal')
                # SHPA : somme des HP au-dessus de l'indice de référence
                #SHPA = np.sum(HP[:MTI])
                # SHPB : somme des HP au-dessous de l'indice de référence
                SHPB = np.sum(HP[MTI+1:])
                top = 0
                for idx, proj in enumerate(HP):
                    if proj != 0:
                        top = idx
                        break
                cnt = 0
                for k in range(end, cut_index+1):
                    if vertical_transitions(word_img, k) > 2:
                        cnt = 1
                if SHPB == 0 and (baseline_idx - top) <= 5 and cnt == 1:
                    wrong = 1
                else:
                    separation_regions.append((end, cut_index, start))
        i -= 1
    return separation_regions, wrong


""" Vérifiez si un segment a un trou intérieur ou non """
def inside_hole(word_img, end_idx, start_idx):
    # Vérifiez si un segment a un trou ou non
    if end_idx == 0 and start_idx == 0:
        return 0
    sk = skeletonize(word_img)
    j = end_idx + 1
    flag = 1
    while j < start_idx:
        VT = vertical_transitions(sk, j)
        if VT <= 2:
            flag = 0
            break
        j += 1
    return flag


def check_hole(segment):
    # Vérifiez si un segment a un trou ou non
    contours, hierarchy = cv.findContours(segment, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt = 0
    for hier in hierarchy[0]:
        if hier[3] >= 0:
            cnt += 1
    return cnt != 0

# vérifier les points
def check_dots(segment):
    segment = (segment > 0).astype(np.uint8)
    contours, hierarchy = cv.findContours(segment[:, 1:segment.shape[1]-1], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = 0
    for c in contours:
        if len(c) >= 1:
            cnt +=1 
    return cnt > 1


def check_stroke(no_dots_copy, segment, baseline_idx, MFV, SR1, SR2, height):   
    segment = (segment > 0).astype(np.uint8)
    components, labels, stats, cen= cv.connectedComponentsWithStats(segment, connectivity=8, ltype = cv.CV_32S)
    (h, w) = segment.shape
    cnt = 0
    for label in range(1, components):
        if stats[label][4] > 3:
            cnt += 1
        else:
            segment[labels==label] = 0
            
    # un seul composant connecté
    if cnt > 2 or cnt == 0:
        return False
    
    #  le segment n'a pas de trous
    if check_hole(segment) or inside_hole(no_dots_copy, SR1[0], SR1[1]) or inside_hole(no_dots_copy, SR2[0], SR2[1]):
        return False
    
    # la somme de la projection horizontale au-dessus de la ligne de base
    #est supérieure à la somme de la projection horizontale au-dessous de la ligne de base
    segment_HP = projection(segment, 'horizontal')
    SHPA = np.sum(segment_HP[:baseline_idx])
    SHPB = np.sum(segment_HP[baseline_idx+1:])
    if (int(SHPB) - int(SHPA)) > 0:
        return False
    
    #  la hauteur du segment est inférieure deux fois la deuxième valeur de crête de la projection horizontale
    HP_segment = projection(segment, 'horizontal')
    top_pixel_segment = -1
    for i, proj in enumerate(HP_segment):
        if proj != 0:
            top_pixel_segment = i
            break
    height_segment = baseline_idx-top_pixel_segment
    if height_segment > height/2:
        return False
    
    # la valeur de mode de la projection horizontale est égale à la valeur MFV
    Some = np.sum(HP_segment)
    mode = Some//len(HP_segment)
    if mode > MFV: 
        return False
    return True


def check_stroke_fin_word(no_dots_copy, segment, baseline_idx, MFV, height):
    segment = (segment > 0).astype(np.uint8)
    components, labels, stats, cen= cv.connectedComponentsWithStats(segment, connectivity=8, ltype = cv.CV_32S)
    cnt = 0
    (h, w) = segment.shape
    for label in range(1, components):
        if stats[label][4] > 3:
            cnt += 1
        else:
            segment[labels==label] = 0
    # un seul composant connecté
    if cnt > 2 or cnt == 0:
        return False
    # la somme de la projection horizontale au-dessus de la ligne de base
    #est supérieure à la somme de la projection horizontale au-dessous de la ligne de base
    segment_HP = projection(segment, 'horizontal')
    SHPA = np.sum(segment_HP[:baseline_idx])
    SHPB = np.sum(segment_HP[baseline_idx+1:])
    if (int(SHPB) - int(SHPA)) > 0:
        return False
    #  la hauteur du segment est inférieure deux fois la deuxième valeur de crête de la projection horizontale
    HP_segment = projection(segment, 'horizontal')
    top_pixel_segment = -1
    for i, proj in enumerate(HP_segment):
        if proj != 0:
            top_pixel_segment = i
            break
    height_segment = baseline_idx-top_pixel_segment
    if height_segment > (height/2):
        return False
    # la valeur de mode de la projection horizontale est égale à la valeur MFV
    Some = np.sum(HP_segment)
    mode = Some//len(HP_segment)
    if mode > MFV: 
        return False
    return True
    
""" Filtration des régions de séparation """
def filter_regions(word_img, no_dots_copy, SRL:list, VP:list, baseline_idx:int, MTI:int, MFV:int, height:int,top_line:int):
    valid_separation_regions = []
    overlap = []
    components, labels= cv.connectedComponents(word_img, connectivity=8)# word_img[:baseline_idx+10, :]
    # SR_idx : indice de la region de separation
    SR_idx = 0
    #word_img = word_img/255
    while SR_idx < len(SRL):
        # SR : la région de séparation actuelle
        # SRL représente la liste des régions de séparation
        SR = SRL[SR_idx]
        end_idx, cut_idx, start_idx = SR
        # Case 1 : Vertical Projection = 0
        if VP[cut_idx] == 0:
            valid_separation_regions.append(SR)
            SR_idx += 1
            continue
        # Case 2 : aucun chemin connecté entre le début et la fin
        if labels[MTI, end_idx] != labels[MTI, start_idx]:
            valid_separation_regions.append(SR)
            overlap.append(SR)
            SR_idx += 1
            continue
        # Case 3 : Contenir des trous
        no_dots_copy = (no_dots_copy > 0).astype(np.uint8)
        cc, l = cv.connectedComponents(1-(no_dots_copy[:, end_idx:start_idx+1]), connectivity=4)
        #no_dots_copy = no_dots_copy/255
        if cc-1 >= 3 and inside_hole(no_dots_copy, end_idx, start_idx):
            SR_idx += 1
            continue
        
        # Case 4 : Pas de référence entre le début et la fin
        segment = no_dots_copy[:, end_idx+1: start_idx]
        if no_dots_copy[baseline_idx, cut_idx]==0 and no_dots_copy[baseline_idx+1, cut_idx]==0:
            segment_HP = projection(segment, 'horizontal')
            SHPA = np.sum(segment_HP[:baseline_idx-1])
            SHPB = np.sum(segment_HP[baseline_idx:])
            if (int(SHPB) - int(SHPA)) >= 0:
                SR_idx += 1
                continue
            elif VP[cut_idx] <= MFV:
                valid_separation_regions.append(SR)
                SR_idx += 1
                continue
            else :
                SR_idx += 1
                continue
        
        # SR est la dernière région ou l'indice de coupe de la région suivante 
        #est égal à zéro et la hauteur du segment est inférieure à la moitié de la hauteur de ligne 
        if SR_idx == len(SRL) - 1 or VP[SRL[SR_idx+1][1]] == 0:
            if SR_idx == len(SRL) - 1:
                segment = no_dots_copy[:, :SRL[SR_idx][1]+1]
                if check_stroke_fin_word(no_dots_copy, segment, baseline_idx, MFV, height):
                    SR_idx += 1
                    continue
                elif VP[cut_idx] <= MFV:
                    valid_separation_regions.append(SR)
                    SR_idx += 1
                    continue
        # SEGP : le segment entre l'indice de coupe précedent et l'indice de coupe suivante
        # SEG : le segment entre l'indice de coupe actuel et l'indice de coupe suivant
        # SEGN : le segment entre le suivant index de coupe et aprés l'indice de coupe suivant
        # SEGNN : le segment entre aprés l'indice de coupe suivant et aprés l'indice de coupe suivant
        
        SEGP = (-1, -1)
        SEG = (-1, -1)
        SEGN = (-1, -1)
        SEGNN = (-1, -1)
        SEGP_SR1 = (0, 0)
        SEGP_SR2 = (0, 0)
        SEG_SR1 = (0, 0)
        SEG_SR2 = (0, 0)
        SEGN_SR1 = (0, 0)
        SEGN_SR2 = (0, 0)
        SEGNN_SR1 = (0, 0)
        SEGNN_SR2 = (0, 0)
        #current_cut = SR[1]
        if SR_idx == 0:
            SEGP = (SRL[SR_idx][1], word_img.shape[1]-1)
            SEGP_SR1 = (SRL[SR_idx][0], SRL[SR_idx][2])
            SEGP_SR2 = (SRL[SR_idx][1], word_img.shape[1]-1)
        if SR_idx > 0:
            SEGP = (SRL[SR_idx][1], SRL[SR_idx-1][1])
            SEGP_SR1 = (SRL[SR_idx][0], SRL[SR_idx][2])
            SEGP_SR2 = (SRL[SR_idx-1][0], SRL[SR_idx-1][2])
        if SR_idx < len(SRL)-1:
            SEG = (SRL[SR_idx+1][1], SRL[SR_idx][1])
            SEG_SR1 = (SRL[SR_idx][0], SRL[SR_idx][2])
            SEG_SR2 = (SRL[SR_idx+1][0], SRL[SR_idx+1][2])
        if SR_idx < len(SRL)-2:
            SEGN = (SRL[SR_idx+2][1], SRL[SR_idx+1][1])
            SEGN_SR1 = (SRL[SR_idx+1][0], SRL[SR_idx+1][2])
            SEGN_SR2 = (SRL[SR_idx+2][0], SRL[SR_idx+2][2])
        elif SR_idx == len(SRL)-2:
            SEGN = (0, SRL[SR_idx+1][1])
            SEGN_SR1 = (SRL[SR_idx+1][0], SRL[SR_idx+1][2])
            SEGN_SR2 = (0, SRL[SR_idx+1][2])            
        if SR_idx < len(SRL)-3:
            SEGNN = (SRL[SR_idx+3][1], SRL[SR_idx+2][1])
            SEGNN_SR1 = (SRL[SR_idx+2][0], SRL[SR_idx+2][2])
            SEGNN_SR2 = (SRL[SR_idx+3][0], SRL[SR_idx+3][2])
        # SEG est un AVC avec des points
        if SEG[0] != -1 and\
            (check_stroke(no_dots_copy, no_dots_copy[:, SEG[0]:SEG[1]], baseline_idx, MFV, SEG_SR1, SEG_SR2, height) \
            and check_dots(word_img[:, SEG[0]:SEG[1]])):
            # SEG est un coup sans points ش
            if SEGP[0] != -1 and \
                ((check_stroke(no_dots_copy, no_dots_copy[:, SEGP[0]:SEGP[1]], baseline_idx, MFV, SEGP_SR1, SEGP_SR2, height) \
                and not check_dots(word_img[:, SEGP[0]:SEGP[1]]))\
                and (SR_idx == 0 or VP[SRL[SR_idx-1][1]] == 0 or (VP[SRL[SR_idx-1][1]] == 0 and SRL[SR_idx-1] in overlap))):
                SR_idx += 2
                continue
            else:
                valid_separation_regions.append(SR)
                SR_idx += 1
                continue
        # SEG est un coup sans points
        elif SEG[0] != -1\
            and (check_stroke(no_dots_copy, no_dots_copy[:, SEG[0]:SEG[1]], baseline_idx, MFV, SEG_SR1, SEG_SR2, height) \
            and not check_dots(word_img[:, SEG[0]:SEG[1]])):
            # Le cas commence par س
                    
            if SEGP[0] != -1\
                and (check_stroke(no_dots_copy, no_dots_copy[:, SEGP[0]:SEGP[1]], baseline_idx, MFV, SEGP_SR1, SEGP_SR2, height) \
                and not check_dots(word_img[:, SEGP[0]:SEGP[1]])):
                SR_idx += 2
                continue
            # SEGN est un trait sans points
            if SEGN[0] != -1 \
                and (check_stroke(no_dots_copy, no_dots_copy[:, SEGN[0]:SEGN[1]], baseline_idx, MFV, SEGN_SR1, SEGN_SR2, height) \
                and not check_dots(word_img[:, SEGN[0]:SEGN[1]])):
                valid_separation_regions.append(SR)
                SR_idx += 3
                continue
            # SEGN course avec points et course SEGNN sans points
            if SEGN[0] != -1\
                and (check_stroke(no_dots_copy, no_dots_copy[:, SEGN[0]:SEGN[1]], baseline_idx, MFV, SEGN_SR1, SEGN_SR2, height) \
                and check_dots(word_img[:, SEGN[0]:SEGN[1]])) \
                and ((SEGNN[0] != -1 \
                and (check_stroke(no_dots_copy, no_dots_copy[:, SEGNN[0]:SEGNN[1]], baseline_idx, MFV, SEGNN_SR1, SEGNN_SR2, height) \
                and not check_dots(word_img[:, SEGNN[0]:SEGNN[1]]))) or (len(SRL)-1-SR_idx == 2) or (len(SRL)-1-SR_idx == 3)):
                    valid_separation_regions.append(SR)
                    SR_idx += 3
                    continue
            # SEGN n'est pas un trait ou un trait avec des points
            if SEGN[0] != -1 \
                and ((not check_stroke(no_dots_copy, no_dots_copy[:, SEGN[0]:SEGN[1]], baseline_idx, MFV, SEGN_SR1, SEGN_SR2, height)) \
                or (check_stroke(no_dots_copy, no_dots_copy[:, SEGN[0]:SEGN[1]], baseline_idx, MFV, SEGN_SR1, SEGN_SR2, height) \
                and check_dots(word_img[:, SEGN[0]:SEGN[1]]))):
                    ######   a reviser
                    valid_separation_regions.append(SR)
                    #######
                    SR_idx += 1
                    continue
            SR_idx += 1
            continue
        if VP[cut_idx] <= MFV:
            valid_separation_regions.append(SR)
            SR_idx += 1
            continue
        SR_idx += 1
    return valid_separation_regions


def extract_char(img, valid_SR):
    # l'image binaire doit être (0, 255) pour être sauvegardée sur le disque et non (0, 1)
    img = img * 255
    h, w = img.shape
    next_cut = w
    char_imgs = []
    for SR in valid_SR:
        char_imgs.append(img[:, SR[1]:next_cut])
        next_cut = SR[1]
    char_imgs.append(img[:, 0:next_cut])
    return char_imgs


def segment(line, word_img):
    binary_word = word_img/255
    no_dots_copy = remove_dots(binary_word)
    VP = projection(no_dots_copy, 'vertical')
    binary_word = fill(binary_word, VP)
    no_dots_copy = remove_dots(binary_word)
    baseline_idx, MFV = baseline_detection(remove_dots(line))
    MTI = horizontal_transitions(no_dots_copy, baseline_idx)
    SRL, wrong = cut_points(binary_word, VP, MFV, MTI, baseline_idx)
    if wrong:
        MTI -= 1
        SRL.clear()
        SRL, wrong = cut_points(binary_word, VP, MFV, MTI, baseline_idx)
    HP = projection(remove_dots(line),'horizontal')
    top_pixel = -1
    for i, proj in enumerate(HP):
        if proj != 0:
            top_pixel = i
            break
    height = baseline_idx-top_pixel
    valid = filter_regions(word_img, no_dots_copy, SRL, VP, baseline_idx, MTI, MFV, height,top_line=-1)
    chars = extract_char(binary_word, valid)
    return chars



if __name__ == "__main__":
    img = cv.imread('Figures/test1.jpg')
    lines = line_horizontal_projection(img)
    line = lines[1]
    words = word_vertical_projection(line)
    word = words[2]
    chars = segment(line, word)
    for idx, c in enumerate(chars):
        save_image(c, 'caracteres', f'caractere{idx}')
    
    
    
    
    
    
    
    
    
# cv.imshow('line',word)
# cv.waitKey(0)

                    