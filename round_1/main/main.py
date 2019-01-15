# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import argparse
import pickle

import nsml
import numpy as np

from skimage.measure import block_reduce
from sklearn.decomposition import PCA

from nsml import DATASET_PATH
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from data_loader import train_data_loader

from keras.applications import MobileNet, vgg16, nasnet, mobilenet_v2, resnet50
from keras.models import Model


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, db):

        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,322

        queries, query_img, references, reference_img = preprocess(queries, db)

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))

        queries = np.asarray(queries)
        #print(queries) ; 
        #print("===========queries shape===========")
        #print(queries.shape)
        query_img = np.asarray(query_img)
        #print(query_img) ; 
        #print("===========query_img shape===========")
        #print(query_img.shape)
        references = np.asarray(references)
        #print(reference) ; 
        #print("===========reference shape===========")
        #print(references.shape)
        reference_img = np.asarray(reference_img)
        #print(reference_img) ; 
        #print("===========reference shape===========")
        #print(reference_img.shape)

        query_img = query_img.astype('float32')
        query_img /= 255 # To normalize to [0, 1]
        reference_img = reference_img.astype('float32')
        reference_img /= 255

        # An image is converted to a feature vector,
        # which is the last layer after running an input through the network (excluding the softmax).
        get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-3].output])

        print('inference start')

        # inference
        query_vecs = get_feature_layer([query_img, 0])[0]
        #print(query_vecs) ; 
        #print("===========query_vecs shape===========")
        #print(query_vecs.shape)
        # caching db output, db inference
        db_output = './db_infer.pkl'
        if os.path.exists(db_output):
            with open(db_output, 'rb') as f:
                reference_vecs = pickle.load(f)
        else:
            reference_vecs = get_feature_layer([reference_img, 0])[0]
            with open(db_output, 'wb') as f:
                pickle.dump(reference_vecs, f)
        query_vecs = global_max_pool_2d(query_vecs)
        query_vecs = query_vecs.reshape(query_vecs.shape[0],-1) # Flattens 1×1 components  
        query_len = query_vecs.shape[0] 
        reference_vecs = global_max_pool_2d(reference_vecs)
        reference_vecs = reference_vecs.reshape(reference_vecs.shape[0],-1)
        reference_len = reference_vecs.shape[0]
        combined = np.concatenate((query_vecs, reference_vecs), axis = 0)
        
        # l2 normalization & pca whitening
        combined = l2_normalize(combined)
        combined_whitened = pca_whiten(combined)
        combined_final = l2_normalize(combined_whitened)
        
        query_vecs = combined[:query_len,]
        reference_vecs = combined[query_len:,]
        #query_vecs = l2_normalize(query_vecs)
        #reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        # which is a similarity metric between images / vectors
        sim_matrix = np.dot(query_vecs, reference_vecs.T)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            sim_list = zip(references, sim_matrix[i].tolist())
            sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)

            ranked_list = [k.split('/')[-1].split('.')[0] for (k, v) in sorted_sim_list]  # ranked list

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def global_max_pool_2d(v):
    v_reduced = block_reduce(v, block_size=(1,v.shape[1],v.shape[2],1), func=np.max)
    return v_reduced

def global_sum_pool_2d(v):
    v_reduced = block_reduce(v, block_size=(1,v.shape[1],v.shape[2],1), func=np.sum)
    return v_reduced  

def pca_whiten(m):
    pca = PCA(whiten=True)
    whitened = pca.fit_transform(m)
    return whitened

# data preprocess
# resizes all the images (query and reference)
# queries, db are paths; _img appended are the processed image files
def preprocess(queries, db):
    query_img = []
    reference_img = []
    img_size = (224, 224)

    for img_path in queries:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        query_img.append(img)

    for img_path in db:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        reference_img.append(img)

    return queries, query_img, db, reference_img


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch_size', type=int, default=128)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    nb_epoch = config.epochs
    batch_size = config.batch_size
    num_classes = 1000
    input_shape = (224, 224, 3)  # input image shape

    # Pretrained model
    #base_model = MobileNet(weights='imagenet', input_shape=input_shape,include_top=False, pooling='avg')
        
    base_model = resnet50.ResNet50(weights='imagenet', input_shape=input_shape, include_top=False, pooling='avg')
    base_model.summary()

    #x = base_model.output
    x = base_model.get_layer(name='activation_44').output
    x = Flatten()(x)
    #x = GlobalAveragePooling2D()(x)
    
    #x = Dropout(0.5)(x)
    #x = LeakyReLU(alpha=0.3)(x)
    #x = Flatten()(x)
    #x = Dense(3000, activation='relu')(x)
    #x = GlobalMaxPooling2D()(x)
    #x = Dense(2000, activation='relu')(x)
    #x = Dense(1000, activation='relu')(x)
    preds = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=preds)

    for layer in model.layers[:-1]:
        layer.trainable = False # Don't train initial pretrained weights

    model.summary()

    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate RMSprop optimizer """
        opt = keras.optimizers.rmsprop(lr=0.00045, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        """ Load data """
        print('dataset path', DATASET_PATH)
        output_path = ['./img_list.pkl', './label_list.pkl']
        train_dataset_path = DATASET_PATH + '/train/train_data'

        if nsml.IS_ON_NSML:
            # Caching file
            nsml.cache(train_data_loader, data_path=train_dataset_path, img_size=input_shape[:2],
                       output_path=output_path)
        else:
            # local에서 실험할경우 dataset의 local-path 를 입력해주세요.
            train_data_loader(train_dataset_path, input_shape[:2], output_path=output_path)

        with open(output_path[0], 'rb') as img_f:
            img_list = pickle.load(img_f)
        with open(output_path[1], 'rb') as label_f:
            label_list = pickle.load(label_f)

        x_train = np.asarray(img_list)
        labels = np.asarray(label_list)
        y_train = keras.utils.to_categorical(labels, num_classes=num_classes)
        x_train = x_train.astype('float32')
        x_train /= 255
        print(len(labels), 'train samples')

        """ Callback """
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        for epoch in range(nb_epoch):
            res = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            initial_epoch=epoch,
                            epochs=epoch + 1,
                            callbacks=[reduce_lr],
                            verbose=1,
                            shuffle=True)
            print(res.history)
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
            nsml.save(epoch)
