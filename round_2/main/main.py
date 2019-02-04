# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pickle
import time

import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import MobileNet
from keras.applications.vgg16 import VGG16
# from keras.applications import VGG19
from utils import *
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score
from data_loader import train_data_loader
from sklearn.decomposition import PCA
# from annoy import AnnoyIndex
from sklearn.neighbors import BallTree


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, _):
        test_path = DATASET_PATH + '/test/test_data'
        db = [os.path.join(test_path, 'reference', path) for path in os.listdir(os.path.join(test_path, 'reference'))]
        
        queries = [v.split('/')[-1].split('.')[0] for v in queries]
        db = [v.split('/')[-1].split('.')[0] for v in db]
        queries.sort()
        db.sort()

        queries, query_vecs, references, reference_vecs = get_feature(model, queries, db)
        
        print('type queries', type(queries)) # list of filenames
        print('queries[0]', queries[0])
        print('type references', type(references)) # list of filanames
        print('references[0]', references[0])
        print()
        print('len queries', len(queries))
        print('query_vecs.shape', query_vecs.shape)
        print('len references', len(references))
        print('reference_vecs.shape', reference_vecs.shape)

        num_queries = query_vecs.shape[0]
        num_references = reference_vecs.shape[0]
        # MAC
        combined_vecs = np.concatenate((query_vecs, reference_vecs), axis=0)
    
        # Calculate MACs in order to fit PCA weights
        combined_macs = calculate_mac(combined_vecs)
        query_vecs = combined_macs[:num_queries]
        reference_vecs = combined_macs[num_queries:]

        # Calculate cosine similarity
        # sim_matrix = np.dot(query_vecs, reference_vecs.T)
        # indices = np.argsort(sim_matrix, axis=1)
        # indices = np.flip(indices, axis=1)
        
        tree = BallTree(reference_vecs, metric='euclidean')              
        indices = tree.query(query_vecs, k=1000, return_distance=False)
        
        # Query expansion
        k_nearest = reference_vecs[indices[:,:5]] # (192, 5)
        print('k_nearest.shape', k_nearest.shape)
        query_vecs = np.expand_dims(query_vecs, axis=1)
        print('query_vecs.shape', query_vecs.shape)
        
        k_nearest = np.concatenate((k_nearest, query_vecs), axis=1)
        print('k_nearest.shape', k_nearest.shape)
        
        query_vecs = np.sum(k_nearest, axis=1)
        query_vecs = l2_normalize(query_vecs)
        print('k_nearest.shape', k_nearest.shape)
        
        # Re-query
        indices = tree.query(query_vecs, k=1000, return_distance=False)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in indices[i]]
            print('len ranked_list', len(ranked_list))
            if len(ranked_list) >= 1000:
                ranked_list = ranked_list[:1000]
            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)
    

def infer_with_validation(queries, db, ground_truth, model):
    print('------- Start inference with validation set -------')
    print('queries.shape', queries.shape)
    print('db.shape', db.shape)
    print('ground_truth.shape', ground_truth.shape)
    
    layer_name = 'conv_pw_11_bn'
    print('layer_name', layer_name)
    intermediate_layer_model = Model(inputs=model.input, 
            outputs=model.get_layer(layer_name).output)
    
    query_vecs = intermediate_layer_model.predict(queries, verbose=1)
    reference_vecs = intermediate_layer_model.predict(db, verbose=1)
    print('query_vecs.shape', query_vecs.shape)
    print('reference_vecs.shape', reference_vecs.shape)
    
    num_queries = query_vecs.shape[0]
    num_references = reference_vecs.shape[0]
    
    # print('--- PCA ---')
    combined_vecs = np.concatenate((query_vecs, reference_vecs), axis=0)
    
    # Calculate MACs in order to fit PCA weights
    combined_macs = calculate_mac(combined_vecs)
    # pca = PCA(n_components=256, whiten=True)
    # pca = pca.fit(combined_macs)
    # combined_macs = pca.transform(combined_macs)
    
    combined_rmacs = calculate_rmac(combined_vecs, L=3)
    
    mac_query_vecs = combined_macs[:num_queries]
    mac_reference_vecs = combined_macs[num_queries:]
    
    rmac_query_vecs = combined_rmacs[:num_queries]
    rmac_reference_vecs = combined_rmacs[num_queries:]
    
    print('mac_query_vecs.shape', mac_query_vecs.shape)
    print('mac_reference_vecs.shape', mac_reference_vecs.shape)
    print('rmac_query_vecs.shape', rmac_query_vecs.shape)
    print('rmac_reference_vecs.shape', rmac_reference_vecs.shape)
    
    # Calculate cosine similarity
    mac_sim_matrix = np.dot(mac_query_vecs, mac_reference_vecs.T)
    rmac_sim_matrix = np.dot(rmac_query_vecs, rmac_reference_vecs.T)
    print('shape of mac_sim_matrix:', mac_sim_matrix.shape)
    print('shape of rmac_sim_matrix:', rmac_sim_matrix.shape)
    
    rmac_avg_precision = average_precision_score(ground_truth, 
            rmac_sim_matrix, average='macro')
    mac_avg_precision = average_precision_score(ground_truth,
            mac_sim_matrix, average='macro')
    
    print('mac_avg_precision', mac_avg_precision)
    print('rmac_avg_precision:', rmac_avg_precision)
    

# data preprocess
def get_feature(model, queries, db):
    img_size = (224, 224)
    test_path = DATASET_PATH + '/test/test_data'

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv_pw_13_bn').output)
    
    test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32')
    query_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['query'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    query_vecs = intermediate_layer_model.predict_generator(query_generator, steps=len(query_generator), verbose=1)

    reference_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['reference'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    reference_vecs = intermediate_layer_model.predict_generator(reference_generator, steps=len(reference_generator),
                                                                verbose=1)
    
    return queries, query_vecs, db, reference_vecs


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=0)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=1383)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (224, 224, 3)  # input image shape

    """ Model """
    # Pretrained model

    #base_model = MobileNet(weights='imagenet', include_top=False)
    #base_model.summary()

    #x = base_model.output
    
    # x = GlobalMaxPooling2D()(x)
    # x = Dense(2000, activation='relu')(x)
    # x = GlobalMaxPooling2D()(x)

    base_model = MobileNet(input_shape=input_shape, weights='imagenet', include_top=False)
    # base_model = VGG16(weights='imagenet', include_top=False)
    # base_model = VGG19(weights='imagenet', include_top=False)
    base_model.summary()

    x = base_model.output
    x = Dense(2000, activation='relu')(x)
    x = GlobalMaxPooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    
    for layer in model.layers[:-8]:
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

        print('dataset path', DATASET_PATH)

        train_datagen = ImageDataGenerator(
            # rescale=1. / 255,
            # shear_range=0.2,
            # zoom_range=0.2,
            # horizontal_flip=True
        )

        train_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        
        nsml.save(0) # Initial save, without any training.
        
        """ Test with a subset of training data """
        print('dataset path', DATASET_PATH)
        output_path = ['./img_list.pkl', './label_list.pkl']
        train_dataset_path = DATASET_PATH + '/train/train_data'
        
        train_data_loader(train_dataset_path, 
                          input_shape[:2], 
                          output_path=output_path,
                          num_samples=5000)
                          
        with open(output_path[0], 'rb') as img_f:
            img_list = pickle.load(img_f)
        with open(output_path[1], 'rb') as label_f:
            label_list = pickle.load(label_f)
            
        x_train = np.asarray(img_list)
        labels = np.asarray(label_list)
        label_binarizer = LabelBinarizer()
        y_train = label_binarizer.fit_transform(labels)
        x_train = x_train.astype('float32')
        x_train /= 255
        print(len(labels), 'validation samples')
        
        query_imgs, reference_imgs, ground_truth = generate_queries_and_refs(
                x_train, labels, label_binarizer)
                
        infer_with_validation(query_imgs, reference_imgs, ground_truth, model)
        

        """ Callback """
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr],
                                      verbose=1,
                                      shuffle=True)
            t2 = time.time()
            print(res.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
            nsml.save(epoch+1)
        print('Total training time : %.1f' % (time.time() - t0))
