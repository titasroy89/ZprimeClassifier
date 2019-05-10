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

def GetInputsReduced(parameters):

    # Get parameters
    classes = parameters['classes']
    eqweight = parameters['eqweight']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    classtag = get_classes_tag(parameters)

    if os.path.isdir('input_reduced/' + classtag):
        if os.path.isfile('input_reduced/' + classtag + '/input_' + fraction + '_val.npy'):
            # print 'These inputfiles already exist, go on to next function.'
            # return
            pass
        else:
            pass
    else:
        os.makedirs('input_reduced/' + classtag)

    maxfiles_per_sample = {'TTbar': -1, 'WJets': -1, 'ST': -1, 'DYJets': -1, 'RSGluon': -1, 'QCD_Mu': -1}

    # Find initial file for each class
    inputfiles = os.listdir('input_reduced/MLInput_Reduced')

    # list of numpy.array, containing the inputs for all classes. Will have len() = number of classes = len(classes)
    all_inputs = {}
    all_labels = {}
    all_eventweights = {}
    for cl in classes.keys():
        first = True

        # Get list of input files for this class, it's a list of lists --> one list per sample belonging to this class
        lists_of_inputfiles = []
        for i in range(len(classes[cl])):
            tmp = []
            sample = classes[cl][i]
            idx = 0
            for j in range(len(inputfiles)):
                if classes[cl][i]+'_' in inputfiles[j] and not 'Weights_' in inputfiles[j] and '.npy' in inputfiles[j] and (idx<maxfiles_per_sample[sample] or maxfiles_per_sample[sample]<0):
                    tmp.append(inputfiles[j])
                    idx += 1
            lists_of_inputfiles.append(tmp)
        print lists_of_inputfiles

        # Read files for this class
        for i in range(len(lists_of_inputfiles)):
            print '\nNow starting with sample %s' % (classes[cl][i])
            for j in range(len(lists_of_inputfiles[i])):
                print 'At file no. %i out of %i.' % (j+1, len(lists_of_inputfiles[i]))
                if first:
                    thisinput = np.load('input_reduced/MLInput_Reduced/' + lists_of_inputfiles[i][j])
                    thiseventweight = np.load('input_reduced/MLInput_Reduced/Weights_' + lists_of_inputfiles[i][j])
                    first = False
                else:
                    thisinput = np.concatenate((thisinput, np.load('input_reduced/MLInput_Reduced/' + lists_of_inputfiles[i][j])))
                    thiseventweight = np.concatenate((thiseventweight, np.load('input_reduced/MLInput_Reduced/Weights_' + lists_of_inputfiles[i][j])))
        # thisinput = thisinput.astype(np.float32)
        # thiseventweight = thiseventweight.astype(np.float32)
        all_inputs[cl] = thisinput
        all_eventweights[cl] = thiseventweight

        # Fill the class i with label i
        thislabel = np.empty(thisinput.shape[0])
        thislabel.fill(cl)
        thislabel = thislabel.astype(np.int8)
        all_labels[cl] = thislabel

    # now read in signal
    signal_masses = [1000, 2000, 3000, 4000, 5000, 6000]
    signal_identifiers = ['RSGluon_All']
    for mass in signal_masses:
        signal_identifiers.append('RSGluon_M' + str(mass))
    all_signals = {}
    all_signal_eventweights = {}
    lists_of_inputfiles_sig = []
    for i in range(len(signal_identifiers)):
        tmp = []
        sample = signal_identifiers[i]
        idx = 0
        for j in range(len(inputfiles)):
            if signal_identifiers[i]+'_' in inputfiles[j] and not 'Weights_' in inputfiles[j] and '.npy' in inputfiles[j]:
                tmp.append(inputfiles[j])
                idx += 1
        lists_of_inputfiles_sig.append(tmp)
    print lists_of_inputfiles_sig

    # Read files for this class
    for i in range(len(lists_of_inputfiles_sig)):
        print '\nNow starting with sample %s' % (signal_identifiers[i])
        first = True
        for j in range(len(lists_of_inputfiles_sig[i])):
            print 'At file no. %i out of %i.' % (j+1, len(lists_of_inputfiles_sig[i]))
            if first:
                thisinput = np.load('input_reduced/MLInput_Reduced/' + lists_of_inputfiles_sig[i][j])
                thiseventweight = np.load('input_reduced/MLInput_Reduced/Weights_' + lists_of_inputfiles_sig[i][j])
                first = False
            else:
                thisinput = np.concatenate((thisinput, np.load('input_reduced/MLInput_Reduced/' + lists_of_inputfiles_sig[i][j])))
                thiseventweight = np.concatenate((thiseventweight, np.load('input_reduced/MLInput_Reduced/Weights_' + lists_of_inputfiles_sig[i][j])))
        # thisinput = thisinput.astype(np.float32)
        # thiseventweight = thiseventweight.astype(np.float32)
        all_signals[i] = thisinput
        all_signal_eventweights[i] = thiseventweight


    if len(all_inputs) != len(classes) or len(all_labels) != len(classes) or len(all_labels) != len(all_eventweights):
        raise ValueError('Number of input classes or labels or eventweights read in does not match number of classes defined in GetInputs().')


    # Here we're making sure to loop through all classes in the numeric order to avoid confusing the labels of inputs -- dict might be unordered, but the input matrix has to be ordered! Thanks god the class names correspond to the list indices from 0 to nclasses-1
    label_concatenated = np.concatenate((tuple([all_labels[i] for i in range(len(all_labels))])))
    input_total = np.concatenate((tuple([all_inputs[i] for i in range(len(all_inputs))])))
    eventweight_total = np.concatenate((tuple([all_eventweights[i] for i in range(len(all_eventweights))])))
    # signal_total = np.concatenate((tuple([all_signals[i] for i in range(len(all_signals))])))
    # signal_eventweight_total = np.concatenate((tuple([all_signal_eventweights[i] for i in range(len(all_signal_eventweights))])))

    # Now create matrix with labels, it's zero everywhere, only the column corresponding to the class the example belongs to has ones
    labels_total = np.zeros((label_concatenated.shape[0], len(classes)))
    for i in range(label_concatenated.shape[0]):
        label = label_concatenated[i]
        labels_total[i,label] = 1
    labels_total = labels_total.astype(np.int8)

    # Treat inf entries
    input_total[input_total == inf]    = 999999.
    input_total[input_total == -inf]   = -999999.
    input_total[np.isnan(input_total)] = 0.
    # signal_total[signal_total == inf]    = 999999.
    # signal_total[signal_total == -inf]   = -999999.
    # signal_total[np.isnan(signal_total)] = 0.

    print input_total[labels_total[:,2]==1][0]

    shuffle = np.random.permutation(np.size(input_total, axis=0))
    input_total       = input_total[shuffle]
    labels_total      = labels_total[shuffle]
    eventweight_total = eventweight_total[shuffle]
    label_concatenated = label_concatenated[shuffle]
    for i in all_signals.keys():
        shuffle_signal = np.random.permutation(np.size(all_signals[i], axis=0))
        all_signals[i]       = all_signals[i][shuffle_signal]
        all_signal_eventweights[i] = all_signal_eventweights[i][shuffle_signal]

    # Cut off some events if not running on full sample
    # percentage = 0.01
    percentage = runonfraction
    frac_train = 0.666 * percentage
    frac_test  = 0.167 * percentage
    frac_val   = 0.167 * percentage
    sumweights = np.sum(eventweight_total, axis=0)
    print 'frac_train/test/val -- train+test, train+test+val: ', frac_train, frac_test, frac_val, frac_train + frac_test, frac_train + frac_test + frac_val
    print 'shape of all inputs: ', input_total.shape
    print 'shape and sum of event weights: ', eventweight_total.shape, sumweights
    cutoffweighted_train = float(sumweights)*float(frac_train)
    cutoffweighted_test  = float(sumweights)*float(frac_train + frac_test)
    cutoffweighted_val   = float(sumweights)*float(frac_train + frac_test + frac_val)

    currentsum = 0.
    takeupto_train = 0
    takeupto_test  = 0
    takeupto_val   = 0
    sumweights_classes = {}
    # initialize this dict
    for i in range(labels_total.shape[1]):
        sumweights_classes[i] = 0.

    for i in range(len(eventweight_total)):
        currentsum += eventweight_total[i,0]
        # if i%1000000 == 0: print i, currentsum
        if currentsum >= cutoffweighted_train and takeupto_train == 0:
            takeupto_train = i+1
        if currentsum >= cutoffweighted_test  and takeupto_test  == 0:
            takeupto_test = i+1
        if currentsum >= cutoffweighted_val  and takeupto_val  == 0:
            takeupto_val = i+1

        #find out which class this event belongs to
        thisclass = label_concatenated[i]
        sumweights_classes[thisclass] += eventweight_total[i,0]

    input_train = input_total[:takeupto_train]
    labels_train = labels_total[:takeupto_train]
    eventweight_train = eventweight_total[:takeupto_train]
    input_test = input_total[takeupto_train:takeupto_test]
    labels_test = labels_total[takeupto_train:takeupto_test]
    eventweight_test = eventweight_total[takeupto_train:takeupto_test]
    input_val = input_total[takeupto_test:takeupto_val]
    labels_val = labels_total[takeupto_test:takeupto_val]
    eventweight_val = eventweight_total[takeupto_test:takeupto_val]
    print 'shapes of inputs (train, test, val): ', input_train.shape, input_test.shape, input_val.shape

    # Calculate class weights such, that after weighting by class_weight all classes have the same number of weighted events, where all events are ALSO weighted by eventweight --> total weight = class_weight * eventweight
    class_weights = {}
    # scale each class to the one with the smallest sum of weights
    minsum = sumweights_classes[0]
    for i in range(len(sumweights_classes)):
        if sumweights_classes[i] < minsum: minsum = sumweights_classes[i]
        # print sumweights_classes[i]

    for i in range(len(sumweights_classes)):
        weight = 1
        if sumweights_classes[i] != 0: weight = minsum/sumweights_classes[i]
        class_weights[i] = weight
        # print weight

    sample_weights_train_list = []
    sample_weights_test_list = []
    sample_weights_val_list = []
    for i in range(len(labels_train[:,0])):
        #loop over training examples i
        for j in range(len(labels_train[i,:])):
            #loop over possible classes j
            if labels_train[i,j] == 1:
                thisweight = class_weights[j] * eventweight_train[i]
                sample_weights_train_list.append(thisweight)
    for i in range(len(labels_test[:,0])):
        for j in range(len(labels_test[i,:])):
            if labels_test[i,j] == 1:
                thisweight = class_weights[j] * eventweight_test[i]
                sample_weights_test_list.append(thisweight)
    for i in range(len(labels_val[:,0])):
        for j in range(len(labels_val[i,:])):
            if labels_val[i,j] == 1:
                thisweight = class_weights[j] * eventweight_val[i]
                sample_weights_val_list.append(thisweight)


    # Test: sum val-sampleweights for each class, should be the same value for all classes
    sums = {0:0., 1:0., 2:0., 3:0., 4:0.}
    for i in range(len(labels_val[:,0])):
        #loop over training examples i
        for j in range(len(labels_val[i,:])):
            #loop over possible classes j
            if labels_val[i,j] == 1:
                sums[j] += sample_weights_val_list[i]

    sample_weights_train = np.asarray(sample_weights_train_list).ravel()
    sample_weights_test = np.asarray(sample_weights_test_list).ravel()
    sample_weights_val = np.asarray(sample_weights_val_list).ravel()

    eventweight_train = np.asarray(eventweight_train).ravel()
    eventweight_test  = np.asarray(eventweight_test).ravel()
    eventweight_val   = np.asarray(eventweight_val).ravel()
    for i in all_signal_eventweights.keys():
        all_signal_eventweights[i] = np.asarray(all_signal_eventweights[i]).ravel()

    # Scale features
    scaler = preprocessing.StandardScaler()
    scaler.mean_ = np.mean(input_train, axis=0)
    scaler.scale_ = np.std(input_train, axis=0)
    input_train = deepcopy(scaler.transform(input_train))
    input_test = deepcopy(scaler.transform(input_test))
    input_val = deepcopy(scaler.transform(input_val))
    for i in all_signals.keys():
        all_signals[i] = deepcopy(scaler.transform(all_signals[i]))


    classtag = get_classes_tag(parameters)

    with open('input_reduced/MLInput_Reduced/variable_names.pkl', 'r') as f:
        variable_names = pickle.load(f)

    # Write out scaler info
    with open('input_reduced/'+classtag+'/NormInfo.txt', 'w') as f:
        for i in range(scaler.mean_.shape[0]):
            var = variable_names[i]
            mean = scaler.mean_[i]
            scale = scaler.scale_[i]
            line = var + ' StandardScaler ' + str(mean) + ' ' + str(scale) + '\n'
            f.write(line)


    with open('input_reduced/'+classtag+'/variable_names.pkl', 'w') as f:
        pickle.dump(variable_names, f)
    np.save('input_reduced/'+classtag+'/input_'+fraction+'_train.npy'  , input_train)
    np.save('input_reduced/'+classtag+'/input_'+fraction+'_test.npy'   , input_test)
    np.save('input_reduced/'+classtag+'/input_'+fraction+'_val.npy'    , input_val)
    np.save('input_reduced/'+classtag+'/labels_'+fraction+'_train.npy' , labels_train)
    np.save('input_reduced/'+classtag+'/labels_'+fraction+'_test.npy'  , labels_test)
    np.save('input_reduced/'+classtag+'/labels_'+fraction+'_val.npy'   , labels_val)

    np.save('input_reduced/'+classtag+'/sample_weights_'+fraction+'_train.npy', sample_weights_train)
    np.save('input_reduced/'+classtag+'/eventweights_'+fraction+'_train.npy', eventweight_train)
    np.save('input_reduced/'+classtag+'/sample_weights_'+fraction+'_test.npy', sample_weights_test)
    np.save('input_reduced/'+classtag+'/eventweights_'+fraction+'_test.npy', eventweight_test)
    np.save('input_reduced/'+classtag+'/sample_weights_'+fraction+'_val.npy', sample_weights_val)
    np.save('input_reduced/'+classtag+'/eventweights_'+fraction+'_val.npy', eventweight_val)



    for i in all_signals.keys():
        np.save('input_reduced/'+classtag+'/'+signal_identifiers[i]+'.npy', all_signals[i])
        np.save('input_reduced/'+classtag+'/'+signal_identifiers[i]+'_eventweight.npy', all_signal_eventweights[i])
