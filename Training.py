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
from keras import metrics
import pickle
import os
from functions import *

def TrainNetwork(parameters):

    # Get parameters
    layers=parameters['layers']
    batch_size=parameters['batchsize']
    dropoutrate=parameters['dropoutrate']
    epochs=parameters['epochs']
    runonfullsample=parameters['runonfullsample']
    equallyweighted=parameters['equallyweighted']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    train_new_model = True
    try:
        # model = keras.models.load_model('output/model_full.h5')
        model = keras.models.load_model('output/'+tag+'/model.h5')
        train_new_model = False
    except:
        pass
    if train_new_model: print 'Couldn\'t find the model "model_%s", a new one will be trained!' % (tag)
    else:
        print 'Found the model, not training a new one, go on to next function.'
        return

    # Get inputs
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


    # Define the network
    model = Sequential()

    print 'Number of input variables: %i' % (input_train.shape[1])
    model.add(Dense(layers[0], input_shape=(input_train.shape[1],), activation='relu'))
    for i in layers[1:len(layers)+1]:
        model.add(Dense(i, activation='relu'))
        # model.add(BatchNormalization())
        model.add(Dropout(dropoutrate))
    model.add(Dense(labels_train.shape[1], activation='softmax'))
    print 'Number of output classes: %i' % (labels_train.shape[1])

    # Train the network
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=[metrics.categorical_accuracy])
    print model.summary()
    if equallyweighted:
        model.fit(input_train, labels_train, sample_weight=sample_weights_train, batch_size=batch_size, epochs=epochs, validation_data=(input_test, labels_test, sample_weights_test))
    else:
        model.fit(input_train, labels_train, sample_weight=eventweights_train, batch_size=batch_size, epochs=epochs, validation_data=(input_test, labels_test, eventweights_test))

    os.makedirs('output/'+tag)
    model.save('output/'+tag+'/model.h5')
    with open('output/'+tag+'/model_history.pkl', 'w') as f:
        pickle.dump(model.history.history, f)



    # Do the predictions
    print 'Now that the model is trained, we\'re going to predict the labels of all 3 sets. '
    print 'predicting for training set'
    pred_train = model.predict(input_train)
    np.save('output/'+tag+'/prediction_train.npy'  , pred_train)
    for cl in range(len(parameters['sampleclasses'])):
        print 'predicting for training set, class ' + str(cl)
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save('output/'+tag+'/prediction_train_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for test set'
    pred_test = model.predict(input_test)
    np.save('output/'+tag+'/prediction_test.npy'  , pred_test)
    for cl in range(len(parameters['sampleclasses'])):
        print 'predicting for test set, class ' + str(cl)
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save('output/'+tag+'/prediction_test_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for val set'
    pred_val = model.predict(input_val)
    np.save('output/'+tag+'/prediction_val.npy'  , pred_val)
    for cl in range(len(parameters['sampleclasses'])):
        print 'predicting for val set, class ' + str(cl)
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save('output/'+tag+'/prediction_val_class'+str(cl)+'.npy'  , tmp)
