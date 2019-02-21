from __future__ import division

import numpy as np
from skimage.measure import block_reduce
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

from collections import Counter
from collections import defaultdict


# -*- coding: utf-8 -*-
"""
Contains all custom functions.

"""

def generate_queries_and_refs(images, labels, label_binarizer=None):
    ''' Generates a query set, reference DB, and ground truth values.

        # Arguments:
            label_binarizer: an sklearn.preprocessing.LabelBinarizer fitted on
                    the original labels. This is required when the number of original
                    labels differs from the number of labels passed to this function, 
                    e.g.) when generating from a subset of the original dataset, and 
                    there are labels left out.

        # Returns: (query_imgs, reference_imgs, ground_truth_matrix)
            ground_truth_matrix: a binary matrix
    '''
    label_image_dict = defaultdict(list)

    label_counts = Counter(labels)
    
    print('The 10 most common labels:')
    print(label_counts.most_common(10))

    if label_binarizer == None:
        label_binarizer = LabelBinarizer(labels)

    for i in range(len(labels)):
        if label_counts[labels[i]] >= 2: # Discard classes with less than 2 images
            label_image_dict[labels[i]].append(images[i])

    query_imgs = []
    query_labels = []
    reference_imgs = []
    reference_labels = []

    for label in label_image_dict:
        query_imgs.append(label_image_dict[label][0])
        encoded_label = label_binarizer.transform([label])[0]
        query_labels.append(encoded_label)
        for ref_img in label_image_dict[label][1:]:
            reference_imgs.append(ref_img)
            reference_labels.append(encoded_label)

    query_imgs = np.asarray(query_imgs)
    query_labels = np.asarray(query_labels)
    reference_imgs = np.asarray(reference_imgs)
    reference_labels = np.asarray(reference_labels)

    ground_truth_matrix = np.dot(query_labels, reference_labels.T)

    return query_imgs, reference_imgs, ground_truth_matrix
    
    
def global_max_pool_2d(v):
    '''
        # Arguments: 
            v: 4D tensor (batch_size, width, height, channels)
        
        # Output: a 2D tensor
    '''
    v_reduced = block_reduce(v, 
            block_size=(1, v.shape[1], v.shape[2], 1), func=np.max)
    v_reduced = v_reduced.squeeze((1,2))
    return v_reduced


def global_sum_pool_2d(v):
    v_reduced = block_reduce(v, 
            block_size=(1, v.shape[1], v.shape[2], 1), func=np.sum)
    return v_reduced


def pca_whiten(m, n_components=None):
    pca = PCA(n_components=n_components, whiten=True)
    whitened = pca.fit_transform(m)
    return whitened

def l2_normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norm, where=norm!=0)  # only divide nonzeros else 1
'''
def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
'''    

def calculate_mac(feature_vecs, pca=None):
    ''' Maximum Activations of Convolutions
        i.e.) a spatial max-pool

        # Arguments:
            feature_vecs: (batch_size, width, height, num_channels)

        # Returns:
            mac_vector (batch_size, num_channels)
    ''' 
    result = global_max_pool_2d(feature_vecs) # Outputs (batch_size, num_channels)
    result = l2_normalize(result)
    
    if pca:
        result = pca.transform(result)
        result = l2_normalize(result)

    return result
    
    
def calculate_rmac(conv_maps, L=3, pca=None):
    ''' Regional Maximum Activation of Convolutions
        
        # Arguments:
            conv_maps: (batch_size, width, height, num_channels)
            L: the number of levels (different sizes) of square regions 
            to pool from.
            
        # Returns:
            rmac_vector (batch_size, num_channels)
    '''
    rmac_regions = get_rmac_regions(conv_maps.shape[1], conv_maps.shape[2], L)
    
    mac_list = []
        
    for region in rmac_regions:
        width_start = region[0]
        width_end = width_start + region[2]
        height_start = region[1]
        height_end = height_start + region[2]
        
        sliced = conv_maps[:, 
                           width_start:width_end,
                           height_start:height_end,
                           :]
                           
        mac = calculate_mac(sliced, pca=pca)
        mac_list.append(mac)
        
    mac_list = np.asarray(mac_list) # (num_regions, batch_size, channels)
    summed_mac_list = np.sum(mac_list, axis=0) # (batch_size, channels)
    summed_mac_list = l2_normalize(summed_mac_list)

    return summed_mac_list
    

def get_rmac_regions(W, H, L):

    ovr = 0.4 # desired overlap of neighboring regions
    steps = np.array([2, 3, 4, 5, 6, 7], dtype=np.float) # possible regions for the long dimension

    w = min(W,H)

    b = (max(H,W) - w)/(steps-1)
    idx = np.argmin(abs(((w ** 2 - w*b)/w ** 2)-ovr)) # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd, Hd = 0, 0
    if H < W:
        Wd = idx + 1
    elif H > W:
        Hd = idx + 1

    regions = []

    for l in range(1,L+1):

        wl = np.floor(2*w/(l+1))
        wl2 = np.floor(wl/2 - 1)

        b = (W - wl) / (l + Wd - 1)
        if np.isnan(b): # for the first level
            b = 0
        cenW = np.floor(wl2 + np.arange(0,l+Wd)*b) - wl2 # center coordinates

        b = (H-wl)/(l+Hd-1)
        if np.isnan(b): # for the first level
            b = 0
        cenH = np.floor(wl2 + np.arange(0,l+Hd)*b) - wl2 # center coordinates

        for i_ in cenH:
            for j_ in cenW:
                # R = np.array([i_, j_, wl, wl], dtype=np.int)
                R = np.array([j_, i_, wl, wl], dtype=np.int)
                if not min(R[2:]):
                    continue

                regions.append(R)

    regions = np.asarray(regions)
    return regions
    
    
def expand_query(query_vecs, reference_vecs, indices):
    pass
    
    
    