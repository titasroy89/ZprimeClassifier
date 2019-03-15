import numpy as np
from numpy import inf
import keras
import matplotlib
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras import metrics, regularizers
import pickle
from copy import deepcopy
import os
from functions import *

def PredictExternal(parameters, inputfolder, outputfolder, filepostfix):
    print 'Making predictions now'
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    fraction = get_fraction(parameters)

    # Get inputs
    model = keras.models.load_model(outputfolder+'/model.h5')
    model_best = keras.models.load_model(outputfolder+'/model_best.h5')
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)

    signal_identifiers = ['RSGluon_All', 'RSGluon_M1000', 'RSGluon_M2000', 'RSGluon_M3000', 'RSGluon_M4000', 'RSGluon_M5000', 'RSGluon_M6000']

    # Do the predictions
    print 'Now that the model is trained, we\'re going to predict the labels of all 3 sets. '
    print 'predicting for training set'

    pred_train = model.predict(input_train)
    np.save(outputfolder+'/prediction_train.npy'  , pred_train)
    for cl in range(len(parameters['classes'])):
        print 'predicting for training set, class ' + str(cl)
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save(outputfolder+'/prediction_train_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for test set'
    print input_test.shape
    print labels_test.shape
    pred_test = model.predict(input_test)
    print pred_test.shape
    print labels_test.shape
    np.save(outputfolder+'/prediction_test.npy'  , pred_test)
    for cl in range(len(parameters['classes'])):
        print 'predicting for test set, class ' + str(cl)
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save(outputfolder+'/prediction_test_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for val set'
    pred_val = model.predict(input_val)
    np.save(outputfolder+'/prediction_val.npy'  , pred_val)
    for cl in range(len(parameters['classes'])):
        print 'predicting for val set, class ' + str(cl)
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save(outputfolder+'/prediction_val_class'+str(cl)+'.npy'  , tmp)

    # Do predictions with best model instead of last
    print 'predicting for training set, best model'
    pred_train = model_best.predict(input_train)
    np.save(outputfolder+'/prediction_train_best.npy'  , pred_train)
    for cl in range(len(parameters['classes'])):
        print 'predicting for training set, best model, class ' + str(cl)
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save(outputfolder+'/prediction_train_class'+str(cl)+'_best.npy'  , tmp)
    print 'predicting for test set, best model'
    pred_test = model_best.predict(input_test)
    np.save(outputfolder+'/prediction_test_best.npy'  , pred_test)
    for cl in range(len(parameters['classes'])):
        print 'predicting for test set, best model, class ' + str(cl)
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save(outputfolder+'/prediction_test_class'+str(cl)+'_best.npy'  , tmp)
    print 'predicting for val set, best model'
    pred_val = model_best.predict(input_val)
    np.save(outputfolder+'/prediction_val_best.npy'  , pred_val)
    for cl in range(len(parameters['classes'])):
        print 'predicting for val set, best model, class ' + str(cl)
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save(outputfolder+'/prediction_val_class'+str(cl)+'_best.npy'  , tmp)

    print 'predicting for signals'
    for i in range(len(signal_identifiers)):
        pred_signal= model.predict(signals[i])
        np.save(outputfolder+'/prediction_'+signal_identifiers[i]+'.npy'  , pred_signal)
        pred_signal = model_best.predict(signals[i])
        np.save(outputfolder+'/prediction_'+signal_identifiers[i]+'_best.npy'  , pred_signal)



def PredictExternalOnPredictions(parameters, inputfolder, inputfolder_predictions, outputfolder, filepostfix):
    print 'Making predictions now'
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    fraction = get_fraction(parameters)

    # Get inputs
    model = keras.models.load_model(outputfolder+'/model.h5')
    model_best = keras.models.load_model(outputfolder+'/model_best.h5')
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights = load_data(parameters, inputfolder=inputfolder, filepostfix='')

    input_train, input_test, input_val, signals = load_predictions(outputfolder=inputfolder_predictions, filepostfix=filepostfix)
    # pred_train, pred_test, pred_val, pred_signals = load_predictions(outputfolder=inputfolder_predictions, filepostfix=filepostfix)
    # input_train = np.concatenate((input_train, pred_train), axis=1)
    # input_test = np.concatenate((input_test, pred_test), axis=1)
    # input_val = np.concatenate((input_val, pred_val), axis=1)
    # for i in signals.keys():
    #     signals[i] = np.concatenate((signals[i], pred_signals[i]), axis=1)

    signal_identifiers = ['RSGluon_All', 'RSGluon_M1000', 'RSGluon_M2000', 'RSGluon_M3000', 'RSGluon_M4000', 'RSGluon_M5000', 'RSGluon_M6000']

    # Do the predictions
    print 'Now that the model is trained, we\'re going to predict the labels of all 3 sets. '
    print 'predicting for training set'

    pred_train = model.predict(input_train)
    np.save(outputfolder+'/prediction_train.npy'  , pred_train)
    for cl in range(len(parameters['classes'])):
        print 'predicting for training set, class ' + str(cl)
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save(outputfolder+'/prediction_train_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for test set'
    pred_test = model.predict(input_test)
    np.save(outputfolder+'/prediction_test.npy'  , pred_test)
    for cl in range(len(parameters['classes'])):
        print 'predicting for test set, class ' + str(cl)
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save(outputfolder+'/prediction_test_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for val set'
    pred_val = model.predict(input_val)
    np.save(outputfolder+'/prediction_val.npy'  , pred_val)
    for cl in range(len(parameters['classes'])):
        print 'predicting for val set, class ' + str(cl)
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save(outputfolder+'/prediction_val_class'+str(cl)+'.npy'  , tmp)

    # Do predictions with best model instead of last
    print 'predicting for training set, best model'
    pred_train = model_best.predict(input_train)
    np.save(outputfolder+'/prediction_train_best.npy'  , pred_train)
    for cl in range(len(parameters['classes'])):
        print 'predicting for training set, best model, class ' + str(cl)
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save(outputfolder+'/prediction_train_class'+str(cl)+'_best.npy'  , tmp)
    print 'predicting for test set, best model'
    pred_test = model_best.predict(input_test)
    np.save(outputfolder+'/prediction_test_best.npy'  , pred_test)
    for cl in range(len(parameters['classes'])):
        print 'predicting for test set, best model, class ' + str(cl)
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save(outputfolder+'/prediction_test_class'+str(cl)+'_best.npy'  , tmp)
    print 'predicting for val set, best model'
    pred_val = model_best.predict(input_val)
    np.save(outputfolder+'/prediction_val_best.npy'  , pred_val)
    for cl in range(len(parameters['classes'])):
        print 'predicting for val set, best model, class ' + str(cl)
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save(outputfolder+'/prediction_val_class'+str(cl)+'_best.npy'  , tmp)

    print 'predicting for signals'
    for i in range(len(signal_identifiers)):
        pred_signal= model.predict(signals[i])
        np.save(outputfolder+'/prediction_'+signal_identifiers[i]+'.npy'  , pred_signal)
        pred_signal = model_best.predict(signals[i])
        np.save(outputfolder+'/prediction_'+signal_identifiers[i]+'_best.npy'  , pred_signal)
