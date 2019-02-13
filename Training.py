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
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import metrics, regularizers
import pickle
from copy import deepcopy
import os
from functions import *

def TrainNetwork(parameters):

    # Get parameters
    layers=parameters['layers']
    batch_size=parameters['batchsize']
    regmethod=parameters['regmethod']
    regrate=parameters['regrate']
    batchnorm=parameters['batchnorm']
    epochs=parameters['epochs']
    learningrate = parameters['learningrate']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight=parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    train_new_model = True
    try:
        model = keras.models.load_model('output/'+tag+'/model.h5')
        train_new_model = False
    except:
        pass
    if train_new_model: print 'Couldn\'t find the model "model_%s", a new one will be trained!' % (tag)
    else:
        print 'Found the model, not training a new one, go on to next function.'
        return

    if not os.path.isdir('output/' + tag): os.makedirs('output/'+tag)

    # Get inputs
    input_train = np.load('input/'+classtag+'/input_'+fraction+'_train.npy')
    input_test = np.load('input/'+classtag+'/input_'+fraction+'_test.npy')
    input_val = np.load('input/'+classtag+'/input_'+fraction+'_val.npy')
    labels_train = np.load('input/'+classtag+'/labels_'+fraction+'_train.npy')
    labels_test = np.load('input/'+classtag+'/labels_'+fraction+'_test.npy')
    labels_val = np.load('input/'+classtag+'/labels_'+fraction+'_val.npy')
    with open('input/'+classtag+'/sample_weights_'+fraction+'_train.pkl', 'r') as f:
        sample_weights_train = pickle.load(f)
    with open('input/'+classtag+'/eventweights_'+fraction+'_train.pkl', 'r') as f:
        eventweights_train = pickle.load(f)
    with open('input/'+classtag+'/sample_weights_'+fraction+'_test.pkl', 'r') as f:
        sample_weights_test = pickle.load(f)
    with open('input/'+classtag+'/eventweights_'+fraction+'_test.pkl', 'r') as f:
        eventweights_test = pickle.load(f)
    with open('input/'+classtag+'/sample_weights_'+fraction+'_val.pkl', 'r') as f:
        sample_weights_val = pickle.load(f)
    with open('input/'+classtag+'/eventweights_'+fraction+'_val.pkl', 'r') as f:
        eventweights_val = pickle.load(f)


    # Define the network
    model = Sequential()
    kernel_regularizer = None
    if regmethod == 'L1':
        kernel_regularizer=regularizers.l1(regrate)
    elif regmethod == 'L2':
        kernel_regularizer=regularizers.l2(regrate)


    print 'Number of input variables: %i' % (input_train.shape[1])
    model.add(Dense(layers[0], input_shape=(input_train.shape[1],), kernel_regularizer=kernel_regularizer))
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
    checkpointer = ModelCheckpoint(filepath='output/'+tag+'/model_epoch{epoch:02d}.h5', verbose=1, save_best_only=False, period=period)
    weights_train, weights_test = sample_weights_train, sample_weights_test
    if not eqweight:
        weights_train, weights_test = eventweights_train, eventweights_test
    model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), callbacks=[checkpointer], verbose=1)


    model.save('output/'+tag+'/model.h5')
    with open('output/'+tag+'/model_history.pkl', 'w') as f:
        pickle.dump(model.history.history, f)



    # Do the predictions
    print 'Now that the model is trained, we\'re going to predict the labels of all 3 sets. '
    print 'predicting for training set'
    pred_train = model.predict(input_train)
    np.save('output/'+tag+'/prediction_train.npy'  , pred_train)
    for cl in range(len(parameters['classes'])):
        print 'predicting for training set, class ' + str(cl)
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save('output/'+tag+'/prediction_train_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for test set'
    pred_test = model.predict(input_test)
    np.save('output/'+tag+'/prediction_test.npy'  , pred_test)
    for cl in range(len(parameters['classes'])):
        print 'predicting for test set, class ' + str(cl)
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save('output/'+tag+'/prediction_test_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for val set'
    pred_val = model.predict(input_val)
    np.save('output/'+tag+'/prediction_val.npy'  , pred_val)
    for cl in range(len(parameters['classes'])):
        print 'predicting for val set, class ' + str(cl)
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save('output/'+tag+'/prediction_val_class'+str(cl)+'.npy'  , tmp)


def TrainOnFails(parameters):

    # Get parameters
    layers=parameters['layers']
    batch_size=parameters['batchsize']
    dropoutrate=parameters['dropoutrate']
    epochs=parameters['epochs']
    runonfullsample=parameters['runonfullsample']
    eqweight=parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)

    # Load model and its history
    model = keras.models.load_model('output/'+tag+'/model.h5')
    # with open('output/'+tag+'/model_history.pkl', 'r') as f:
    #     model_history = pickle.load(f)
    # model.history.history = model_history

    # Get the predictions and weights
    pred_train = np.load('output/'+tag+'/prediction_train.npy')
    pred_test = np.load('output/'+tag+'/prediction_test.npy')
    pred_val = np.load('output/'+tag+'/prediction_val.npy')
    if runonfullsample:
        input_train = np.load('input/'+classtag+'/input_full_train.npy')
        input_test = np.load('input/'+classtag+'/input_full_test.npy')
        input_val = np.load('input/'+classtag+'/input_full_val.npy')
        labels_train = np.load('input/'+classtag+'/labels_full_train.npy')
        labels_test = np.load('input/'+classtag+'/labels_full_test.npy')
        labels_val = np.load('input/'+classtag+'/labels_full_val.npy')
        with open('input/'+classtag+'/sample_weights_full_train.pkl', 'r') as f:
            sample_weights_train = pickle.load(f)
        with open('input/'+classtag+'/eventweights_full_train.pkl', 'r') as f:
            eventweights_train = pickle.load(f)
        with open('input/'+classtag+'/sample_weights_full_test.pkl', 'r') as f:
            sample_weights_test = pickle.load(f)
        with open('input/'+classtag+'/eventweights_full_test.pkl', 'r') as f:
            eventweights_test = pickle.load(f)
        with open('input/'+classtag+'/sample_weights_full_val.pkl', 'r') as f:
            sample_weights_val = pickle.load(f)
        with open('input/'+classtag+'/eventweights_full_val.pkl', 'r') as f:
            eventweights_val = pickle.load(f)
    else:
        input_train = np.load('input/'+classtag+'/input_part_train.npy')
        input_test = np.load('input/'+classtag+'/input_part_test.npy')
        input_val = np.load('input/'+classtag+'/input_part_val.npy')
        labels_train = np.load('input/'+classtag+'/labels_part_train.npy')
        labels_test = np.load('input/'+classtag+'/labels_part_test.npy')
        labels_val = np.load('input/'+classtag+'/labels_part_val.npy')
        with open('input/'+classtag+'/sample_weights_part_train.pkl', 'r') as f:
            sample_weights_train = pickle.load(f)
        with open('input/'+classtag+'/eventweights_part_train.pkl', 'r') as f:
            eventweights_train = pickle.load(f)
        with open('input/'+classtag+'/sample_weights_part_test.pkl', 'r') as f:
            sample_weights_test = pickle.load(f)
        with open('input/'+classtag+'/eventweights_part_test.pkl', 'r') as f:
            eventweights_test = pickle.load(f)
        with open('input/'+classtag+'/sample_weights_part_val.pkl', 'r') as f:
            sample_weights_val = pickle.load(f)
        with open('input/'+classtag+'/eventweights_part_val.pkl', 'r') as f:
            eventweights_val = pickle.load(f)

    newfolder = True
    try:
        # model = keras.models.load_model('output/model_full.h5')
        model = keras.models.load_model('output/'+tag+'_retrained/model.h5')
        newfolder = False
    except:
        pass
    if newfolder:
        os.makedirs('output/'+tag+'_retrained')


    # Find wrong predictions (i.e. true class doesn't have largest value)
    mask_train = get_indices_wrong_predictions(labels_train, pred_train)
    weights_train, weights_test = sample_weights_train, sample_weights_test
    if not eqweight:
        weights_train, weights_test = eventweights_train, eventweights_test


    input_train_wrong = input_train[mask_train]
    labels_train_wrong = labels_train[mask_train]
    weights_train_wrong = weights_train[mask_train]
    additional_weight_wrong = np.zeros(len(weights_train))
    additional_weight_wrong[mask_train] = 5.
    increased_weights_train_wrong = weights_train * additional_weight_wrong


    nepochs_wrong = int(epochs/1)
    # model.fit(input_train_wrong, labels_train_wrong, sample_weight=weights_train_wrong, batch_size=batch_size, epochs=nepochs_wrong, validation_data=(input_test, labels_test, weights_test))
    model.fit(input_train, labels_train, sample_weight=increased_weights_train_wrong, batch_size=batch_size, epochs=nepochs_wrong, validation_data=(input_test, labels_test, weights_test))

    # Train again on full sample
    model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, validation_data=(input_test, labels_test, weights_test))

    for i in range(4):
        # Find wrong predictions (i.e. true class doesn't have largest value)
        mask_train = get_indices_wrong_predictions(labels_train, model.predict(input_train))
        weights_train, weights_test = sample_weights_train, sample_weights_test
        if not eqweight:
            weights_train, weights_test = eventweights_train, eventweights_test


        input_train_wrong = input_train[mask_train]
        labels_train_wrong = labels_train[mask_train]
        weights_train_wrong = weights_train[mask_train]
        additional_weight_wrong = np.zeros(len(weights_train))
        additional_weight_wrong[mask_train] = 5.
        increased_weights_train_wrong = weights_train * additional_weight_wrong

        print 'unweighted number of wrongly classified training events: %i' % (input_train_wrong.shape[0])


        nepochs_wrong = int(epochs/1)
        # model.fit(input_train_wrong, labels_train_wrong, sample_weight=weights_train_wrong, batch_size=batch_size, epochs=nepochs_wrong, validation_data=(input_test, labels_test, weights_test))
        model.fit(input_train, labels_train, sample_weight=increased_weights_train_wrong, batch_size=batch_size, epochs=nepochs_wrong, validation_data=(input_test, labels_test, weights_test))

        # Train again on full sample
        model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, validation_data=(input_test, labels_test, weights_test))


    model.save('output/'+tag+'_retrained/model.h5')
    with open('output/'+tag+'_retrained/model_history.pkl', 'w') as f:
        pickle.dump(model.history.history, f)
