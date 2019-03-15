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
from PredictExternal import *

def TrainSecondNetwork(parameters, parameters2, use_best_model=False):

    # Get parameters
    layers=parameters2['layers']
    batch_size=parameters2['batchsize']
    regmethod=parameters2['regmethod']
    regrate=parameters2['regrate']
    batchnorm=parameters2['batchnorm']
    epochs=parameters2['epochs']
    learningrate = parameters2['learningrate']
    runonfraction = parameters2['runonfraction']
    fraction = get_fraction(parameters2)
    eqweight=parameters2['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    tag2 = dict_to_str(parameters2)
    classtag2 = get_classes_tag(parameters2)
    train_new_model = True
    try:
        model = keras.models.load_model('output/'+tag+'/SecondModels/output/'+tag2+'/model.h5')
        train_new_model = False
    except:
        pass
    if train_new_model: print 'Couldn\'t find the model "model_%s", a new one will be trained!' % ('output/'+tag+'/SecondModels/output/'+tag2+'/model.h5')
    else:
        print 'Found the model, not training a new one, go on to next function.'
        return
        # print 'Found model, but I will retrain it!'
    if not os.path.isdir('output/'+tag+'/SecondModels/output/'+tag2): os.makedirs('output/'+tag+'/SecondModels/output/'+tag2)


    postfix = '_fail'
    if use_best_model: postfix += '_best'
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights = load_data(parameters, inputfolder='output/'+tag+'/cut', filepostfix=postfix)

    # Define the network
    model = Sequential()
    kernel_regularizer = None
    if regmethod == 'L1':
        kernel_regularizer=regularizers.l1(regrate)
    elif regmethod == 'L2':
        kernel_regularizer=regularizers.l2(regrate)


    print 'Number of input variables: %i' % (input_train.shape[1])
    model.add(Dense(layers[0], input_shape=(input_train.shape[1],), kernel_regularizer=kernel_regularizer))
    if regmethod == 'dropout': model.add(Dropout(regrate))
    if batchnorm: model.add(BatchNormalization())

    for i in layers[1:len(layers)+1]:
        model.add(Dense(i, activation='relu', kernel_regularizer=kernel_regularizer))
        if batchnorm: model.add(BatchNormalization())
        if regmethod == 'dropout': model.add(Dropout(regrate))

    model.add(Dense(labels_train.shape[1], activation='softmax', kernel_regularizer=kernel_regularizer))
    print 'Number of output classes: %i' % (labels_train.shape[1])


    # Train the network
    opt = keras.optimizers.Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    mymetrics = [metrics.categorical_accuracy]
    # mymetrics = [metrics.categorical_accuracy, metrics.mean_squared_error, metrics.categorical_crossentropy, metrics.kullback_leibler_divergence]
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=mymetrics)
    print model.summary()

    period = epochs / 5
    checkpointer = ModelCheckpoint(filepath='output/'+tag+'/SecondModels/output/'+tag2+'/model_epoch{epoch:02d}.h5', verbose=1, save_best_only=False, period=period)
    checkpoint_bestmodel = ModelCheckpoint(filepath='output/'+tag+'/SecondModels/output/'+tag2+'/model_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=20, verbose=0, mode='min', baseline=None, restore_best_weights=True)
    weights_train, weights_test = sample_weights_train, sample_weights_test
    if not eqweight:
        weights_train, weights_test = eventweights_train, eventweights_test
    # model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), callbacks=[checkpointer, checkpoint_bestmodel, earlystopping], verbose=1)
    model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), callbacks=[checkpointer, checkpoint_bestmodel], verbose=1)


    model.save('output/'+tag+'/SecondModels/output/'+tag2+'/model.h5')
    with open('output/'+tag+'/SecondModels/output/'+tag2+'/model_history.pkl', 'w') as f:
        pickle.dump(model.history.history, f)


    PredictExternal(parameters, inputfolder='output/'+tag+'/cut', filepostfix=postfix, outputfolder='output/'+tag+'/SecondModels/output/'+tag2)
