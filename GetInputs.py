import numpy as np
from numpy import inf
import keras
import matplotlib
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import pickle
import os
from functions import *

def GetInputs(parameters):

    # Get parameters
    sample_classes = parameters['sampleclasses']
    runonfullsample = parameters['runonfullsample']
    equallyweighted = parameters['equallyweighted']
    classtag = get_classes_tag(parameters)

    if os.path.isdir('input/' + classtag):
        return
    else:
        os.makedirs('input/' + classtag)

    # Get samples as input
    # testcomment
    # sample_classes    = {0: ['TTbar', 'WJets', 'ST', 'DYJets'], 1: ['QCD_Mu']}
    maxfiles_per_sample = {'TTbar': -1, 'WJets': -1, 'ST': -1, 'DYJets': -1, 'RSGluon': -1, 'QCD_Mu': -1}
    if not runonfullsample:
        maxfiles_per_sample = {'TTbar': 2, 'WJets': 2, 'ST': 2, 'DYJets': 2, 'RSGluon': 2, 'QCD_Mu': 2}

    # Find initial file for each class
    inputfiles = os.listdir('input/MLInput')

    # list of numpy.array, containing the inputs for all classes. Will have len() = number of classes = len(sample_classes)
    all_inputs = {}
    all_labels = {}
    all_eventweights = {}
    for cl in sample_classes.keys():
        first = True

        # Get list of input files for this class, it's a list of lists --> one list per sample belonging to this class
        lists_of_inputfiles = []
        for i in range(len(sample_classes[cl])):
            tmp = []
            sample = sample_classes[cl][i]
            idx = 0
            for j in range(len(inputfiles)):
                if sample_classes[cl][i]+'_' in inputfiles[j] and not 'Weights_' in inputfiles[j] and '.npy' in inputfiles[j] and (idx<maxfiles_per_sample[sample] or maxfiles_per_sample[sample]<0):
                    tmp.append(inputfiles[j])
                    idx += 1
            lists_of_inputfiles.append(tmp)
        print lists_of_inputfiles

        # Read files for this class
        for i in range(len(lists_of_inputfiles)):
            print '\nNow starting with sample %s' % (sample_classes[cl][i])
            for j in range(len(lists_of_inputfiles[i])):
                print 'At file no. %i out of %i.' % (j+1, len(lists_of_inputfiles[i]))
                if first:
                    thisinput = np.load('input/MLInput/' + lists_of_inputfiles[i][j])
                    thiseventweight = np.load('input/MLInput/Weights_' + lists_of_inputfiles[i][j])
                    first = False
                else:
                    thisinput = np.concatenate((thisinput, np.load('input/MLInput/' + lists_of_inputfiles[i][j])))
                    thiseventweight = np.concatenate((thiseventweight, np.load('input/MLInput/Weights_' + lists_of_inputfiles[i][j])))
        thisinput = thisinput.astype(np.float32)
        thiseventweight = thiseventweight.astype(np.float32)
        all_inputs[cl] = thisinput
        all_eventweights[cl] = thiseventweight

        # Fill the class i with label i
        thislabel = np.empty(thisinput.shape[0])
        thislabel.fill(cl)
        thislabel = thislabel.astype(np.int8)
        all_labels[cl] = thislabel


    if len(all_inputs) != len(sample_classes) or len(all_labels) != len(sample_classes) or len(all_labels) != len(all_eventweights):
        raise ValueError('Number of input classes or labels or eventweights read in does not match number of classes defined in GetInputs().')


    # Here we're making sure to loop through all classes in the numeric order to avoid confusing the labels of inputs -- dict might be unordered, but the input matrix has to be ordered! Thanks god the class names correspond to the list indices from 0 to nclasses-1
    label_concatenated = np.concatenate((tuple([all_labels[i] for i in range(len(all_labels))])))
    input_total = np.concatenate((tuple([all_inputs[i] for i in range(len(all_inputs))])))
    eventweight_total = np.concatenate((tuple([all_eventweights[i] for i in range(len(all_eventweights))])))

    # Now create matrix with labels, it's zero everywhere, only the column corresponding to the class the example belongs to has ones
    labels_total = np.zeros((label_concatenated.shape[0], len(sample_classes)))
    for i in range(label_concatenated.shape[0]):
        label = label_concatenated[i]
        labels_total[i,label] = 1
    labels_total = labels_total.astype(np.int8)

    class_weights = {}
    class_weights_tmp = class_weight.compute_class_weight('balanced', np.unique(np.array(range(len(sample_classes)))), label_concatenated)
    for i in range(len(sample_classes)):
        class_weights[i] = class_weights_tmp[i]

    # Treat inf entries
    input_total[input_total == inf] = 999999.
    input_total[input_total == -inf] = -999999.
    input_total[np.isnan(input_total)] = 0.

    input_train, input_test, labels_train, labels_test, eventweight_train, eventweight_test = train_test_split(input_total, labels_total, eventweight_total, random_state=42, train_size=0.67)
    input_val, input_test, labels_val, labels_test, eventweight_val, eventweight_test = train_test_split(input_test, labels_test, eventweight_test, random_state=42, train_size=0.5)

    eventweight_train = np.asarray(eventweight_train).ravel()
    eventweight_test = np.asarray(eventweight_test).ravel()
    eventweight_val = np.asarray(eventweight_val).ravel()

    # Scale features
    scaler = preprocessing.StandardScaler().fit(input_train)
    scaler.transform(input_train)
    scaler.transform(input_test)
    scaler.transform(input_val)


    # Build sample weights (1 weight per training example) from class weights and labels
    classes_train = []
    classes_test = []
    classes_val = []
    for i in range(len(labels_train[:,0])):
        #loop over training examples i
        for j in range(len(labels_train[i,:])):
            #loop over possible classes j
            if labels_train[i,j] == 1: classes_train.append(j)
    for i in range(len(labels_test[:,0])):
        #loop over training examples i
        for j in range(len(labels_test[i,:])):
            #loop over possible classes j
            if labels_test[i,j] == 1: classes_test.append(j)
    for i in range(len(labels_val[:,0])):
        #loop over training examples i
        for j in range(len(labels_val[i,:])):
            #loop over possible classes j
            if labels_val[i,j] == 1: classes_val.append(j)
    sample_weights_train = np.array([class_weights[classes_train[i]] for i in range(len(classes_train))])
    sample_weights_test = np.array([class_weights[classes_test[i]] for i in range(len(classes_test))])
    sample_weights_val = np.array([class_weights[classes_val[i]] for i in range(len(classes_val))])
    print class_weights
    print 'after scaling, we have the following weighted number of class0 events: %f, class1: %f' % (len(all_inputs[0])*class_weights[0], len(all_inputs[1])*class_weights[1])

    classtag = get_classes_tag(parameters)


    with open('input/MLInput/variable_names.pkl', 'r') as f:
        variable_names = pickle.load(f)
    with open('input/'+classtag+'/variable_names.pkl', 'w') as f:
        pickle.dump(variable_names, f)
    if runonfullsample:
        np.save('input/'+classtag+'/input_full_train.npy'  , input_train)
        np.save('input/'+classtag+'/input_full_test.npy'   , input_test)
        np.save('input/'+classtag+'/input_full_val.npy'    , input_val)
        np.save('input/'+classtag+'/labels_full_train.npy' , labels_train)
        np.save('input/'+classtag+'/labels_full_test.npy'  , labels_test)
        np.save('input/'+classtag+'/labels_full_val.npy'   , labels_val)
        with open('input/'+classtag+'/sample_weights_full_train.pkl', 'w') as f:
            pickle.dump(sample_weights_train, f)
        with open('input/'+classtag+'/eventweights_full_train.pkl', 'w') as f:
            pickle.dump(eventweight_train, f)
        with open('input/'+classtag+'/sample_weights_full_test.pkl', 'w') as f:
            pickle.dump(sample_weights_test, f)
        with open('input/'+classtag+'/eventweights_full_test.pkl', 'w') as f:
            pickle.dump(eventweight_test, f)
        with open('input/'+classtag+'/sample_weights_full_val.pkl', 'w') as f:
            pickle.dump(sample_weights_val, f)
        with open('input/'+classtag+'/eventweights_full_val.pkl', 'w') as f:
            pickle.dump(eventweight_val, f)
    else:
        np.save('input/'+classtag+'/input_part_train.npy'  , input_train)
        np.save('input/'+classtag+'/input_part_test.npy'   , input_test)
        np.save('input/'+classtag+'/input_part_val.npy'    , input_val)
        np.save('input/'+classtag+'/labels_part_train.npy' , labels_train)
        np.save('input/'+classtag+'/labels_part_test.npy'  , labels_test)
        np.save('input/'+classtag+'/labels_part_val.npy'   , labels_val)
        with open('input/'+classtag+'/sample_weights_part_train.pkl', 'w') as f:
            pickle.dump(sample_weights_train, f)
        with open('input/'+classtag+'/eventweights_part_train.pkl', 'w') as f:
            pickle.dump(eventweight_train, f)
        with open('input/'+classtag+'/sample_weights_part_test.pkl', 'w') as f:
            pickle.dump(sample_weights_test, f)
        with open('input/'+classtag+'/eventweights_part_test.pkl', 'w') as f:
            pickle.dump(eventweight_test, f)
        with open('input/'+classtag+'/sample_weights_part_val.pkl', 'w') as f:
            pickle.dump(sample_weights_val, f)
        with open('input/'+classtag+'/eventweights_part_val.pkl', 'w') as f:
            pickle.dump(eventweight_val, f)
