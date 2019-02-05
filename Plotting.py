import numpy as np
from numpy import inf
import keras
import matplotlib
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

from ROOT import TCanvas, TFile, TH1F, TH2F, gROOT, kRed, kBlue, kGreen, kMagenta, kCyan, gStyle
from ROOT import gErrorIgnoreLevel, kInfo, kWarning, kError

import math
import pickle
import sys
import os
from functions import *

def PlotPerformance(parameters, plot_distributions=False):
    print 'Now plotting the performance'
    gErrorIgnoreLevel = kWarning

    # Get parameters
    runonfullsample = parameters['runonfullsample']
    equallyweighted = parameters['equallyweighted']
    # classes = parameters['sampleclasses']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    colorstr = ['r', 'b', 'g', 'm', 'c']
    rootcolors = {'r': kRed, 'b': kBlue+1, 'g': kGreen+1, 'm': kMagenta, 'c': kCyan}

    # Get model and its history
    model = keras.models.load_model('output/'+tag+'/model.h5')
    with open('output/'+tag+'/model_history.pkl', 'r') as f:
        model_history = pickle.load(f)

    if runonfullsample:
        # Get inputs
        input_train = np.load('input/'+classtag+'/input_full_train.npy')
        input_test = np.load('input/'+classtag+'/input_full_test.npy')
        input_val = np.load('input/'+classtag+'/input_full_val.npy')
        labels_train = np.load('input/'+classtag+'/labels_full_train.npy')
        labels_test = np.load('input/'+classtag+'/labels_full_test.npy')
        labels_val = np.load('input/'+classtag+'/labels_full_val.npy')
        with open('input/'+classtag+'/sample_weights_full_train.pkl', 'r') as f:
            sample_weights_train = pickle.load(f)
        with open('input/'+classtag+'/sample_weights_full_test.pkl', 'r') as f:
            sample_weights_test = pickle.load(f)
        with open('input/'+classtag+'/sample_weights_full_val.pkl', 'r') as f:
            sample_weights_val = pickle.load(f)
        with open('input/'+classtag+'/eventweights_full_train.pkl', 'r') as f:
            eventweights_train = pickle.load(f)
        with open('input/'+classtag+'/eventweights_full_test.pkl', 'r') as f:
            eventweights_test = pickle.load(f)
        with open('input/'+classtag+'/eventweights_full_val.pkl', 'r') as f:
            eventweights_val = pickle.load(f)
    else:
        # Get inputs
        input_train = np.load('input/'+classtag+'/input_part_train.npy')
        input_test = np.load('input/'+classtag+'/input_part_test.npy')
        input_val = np.load('input/'+classtag+'/input_part_val.npy')
        labels_train = np.load('input/'+classtag+'/labels_part_train.npy')
        labels_test = np.load('input/'+classtag+'/labels_part_test.npy')
        labels_val = np.load('input/'+classtag+'/labels_part_val.npy')
        with open('input/'+classtag+'/sample_weights_part_train.pkl', 'r') as f:
            sample_weights_train = pickle.load(f)
        with open('input/'+classtag+'/sample_weights_part_test.pkl', 'r') as f:
            sample_weights_test = pickle.load(f)
        with open('input/'+classtag+'/sample_weights_part_val.pkl', 'r') as f:
            sample_weights_val = pickle.load(f)
        with open('input/'+classtag+'/eventweights_part_train.pkl', 'r') as f:
            eventweights_train = pickle.load(f)
        with open('input/'+classtag+'/eventweights_part_test.pkl', 'r') as f:
            eventweights_test = pickle.load(f)
        with open('input/'+classtag+'/eventweights_part_val.pkl', 'r') as f:
            eventweights_val = pickle.load(f)

    # Load model prediction
    pred_train = np.load('output/'+tag+'/prediction_train.npy')
    pred_test = np.load('output/'+tag+'/prediction_test.npy')
    pred_val = np.load('output/'+tag+'/prediction_val.npy')


    FalsePositiveRates, TruePositiveRates, Thresholds, aucs = get_fpr_tpr_thr_auc(parameters=parameters, pred_val=pred_val, labels_val=labels_val, weights_val=sample_weights_val)
    FalsePositiveRates_lumi, TruePositiveRates_lumi, Thresholds_lumi, aucs_lumi = get_fpr_tpr_thr_auc(parameters=parameters, pred_val=pred_val, labels_val=labels_val, weights_val=eventweights_val)

    if not os.path.isdir('Plots/' + tag): os.makedirs('Plots/' + tag)





    # log_model_performance(parameters=parameters, model_history=model_history, aucs=aucs)
    plot_roc(parameters=parameters, FalsePositiveRates=FalsePositiveRates, TruePositiveRates=TruePositiveRates, aucs=aucs, colorstr=colorstr, outnametag='equallyweighted')
    plot_roc(parameters=parameters, FalsePositiveRates=FalsePositiveRates_lumi, TruePositiveRates=TruePositiveRates_lumi, aucs=aucs_lumi, colorstr=colorstr, outnametag='lumiweighted')
    # plot_loss(parameters=parameters, model_history=model_history)
    # plot_accuracy(parameters=parameters, model_history=model_history)
    # plot_model(model, show_shapes=True, to_file='Plots/'+tag+'/Model.pdf')
    # plot_weights(parameters=parameters, model=model)
    # plot_confusion_matrices(parameters=parameters, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val)
    pred_trains, weights_trains, normweights_trains, lumiweights_trains, pred_vals, weights_vals, normweights_vals, lumiweights_vals = get_data_dictionaries(parameters=parameters, eventweights_train=eventweights_train, sample_weights_train=sample_weights_train, pred_train=pred_train, labels_train=labels_train, eventweights_val=eventweights_val, sample_weights_val=sample_weights_val, pred_val=pred_val, labels_val=labels_val)
    if not plot_distributions: return
    # plot_outputs_2d(parameters=parameters, pred_vals=pred_vals, lumiweights_vals=lumiweights_vals, colorstr=colorstr, rootcolors=rootcolors)
    plot_outputs_1d(parameters=parameters, colorstr=colorstr, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals)



def PlotInputs(parameters):

    # Get parameters
    runonfullsample = parameters['runonfullsample']
    classtag = get_classes_tag(parameters)

    if not os.path.isdir('Plots/InputDistributions/' + classtag):
        os.makedirs('Plots/InputDistributions/' + classtag)

    # Get inputs
    if runonfullsample:
        input_train = np.load('input/'+classtag+'/input_full_train.npy')
        input_test = np.load('input/'+classtag+'/input_full_test.npy')
        input_val = np.load('input/'+classtag+'/input_full_val.npy')
        labels_test = np.load('input/'+classtag+'/labels_full_test.npy')
        labels_train = np.load('input/'+classtag+'/labels_full_train.npy')
        labels_val = np.load('input/'+classtag+'/labels_full_val.npy')
        with open('input/'+classtag+'/sample_weights_full_train.pkl', 'r') as f:
            sample_weights_train = pickle.load(f)
        with open('input/'+classtag+'/sample_weights_full_test.pkl', 'r') as f:
            sample_weights_test = pickle.load(f)
        with open('input/'+classtag+'/sample_weights_full_val.pkl', 'r') as f:
            sample_weights_val = pickle.load(f)
        with open('input/'+classtag+'/variable_names.pkl', 'r') as f:
            variable_names = pickle.load(f)
    else:
        input_train = np.load('input/'+classtag+'/input_part_train.npy')
        input_test = np.load('input/'+classtag+'/input_part_test.npy')
        input_val = np.load('input/'+classtag+'/input_part_val.npy')
        labels_test = np.load('input/'+classtag+'/labels_part_test.npy')
        labels_train = np.load('input/'+classtag+'/labels_part_train.npy')
        labels_val = np.load('input/'+classtag+'/labels_part_val.npy')
        with open('input/'+classtag+'/sample_weights_part_train.pkl', 'r') as f:
            sample_weights_train = pickle.load(f)
        with open('input/'+classtag+'/sample_weights_part_test.pkl', 'r') as f:
            sample_weights_test = pickle.load(f)
        with open('input/'+classtag+'/sample_weights_part_val.pkl', 'r') as f:
            sample_weights_val = pickle.load(f)
        with open('input/'+classtag+'/variable_names.pkl', 'r') as f:
            variable_names = pickle.load(f)

    # Divide into classes
    input_train_classes = {}
    input_test_classes = {}
    input_val_classes = {}
    weights_train_classes = {}
    weights_test_classes = {}
    weights_val_classes = {}
    for i in range(labels_train.shape[1]):
        input_train_classes[i] = input_train[labels_train[:,i] == 1]
        input_test_classes[i] = input_test[labels_test[:,i] == 1]
        input_val_classes[i] = input_val[labels_val[:,i] == 1]
        weights_train_classes[i] = sample_weights_train[labels_train[:,i] == 1]
        weights_test_classes[i] = sample_weights_test[labels_test[:,i] == 1]
        weights_val_classes[i] = sample_weights_val[labels_val[:,i] == 1]

    # Create class-title dictionary
    classes = parameters['sampleclasses']
    classtitles = {}
    for key in classes.keys():
        list = classes[key]
        title = ''
        for i in range(len(list)):
            title = title + list[i]
            if i < len(list)-1:
                title = title + '+'
        classtitles[key] = title

    matplotlib.style.use('default')
    # print input_train_classes
    nbins = 50
    idx = 0
    for varname in variable_names:
        xmax = max([max(input_train_classes[i][:,idx]) for i in range(len(input_train_classes))])
        xmin = min([min(input_train_classes[i][:,idx]) for i in range(len(input_train_classes))])
        if xmax == xmin: xmax = xmin + 1.
        xmin = min([0,xmin])
        binwidth = (xmax - xmin) / float(nbins)
        bins = np.arange(xmin, xmax + binwidth, binwidth)

        plt.clf()
        fig = plt.figure()
        for i in range(len(input_train_classes)):
            mycolor = 'C'+str(i)
            plt.hist(input_train_classes[i][:,idx], weights=weights_train_classes[i], bins=bins, histtype='step', label='Training sample, '+classtitles[i])
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel(varname)
        plt.ylabel('Number of events / bin')
        if runonfullsample: fig.savefig('Plots/InputDistributions/' + classtag+  '/' + varname + '_full.pdf')
        else: fig.savefig('Plots/InputDistributions/' + classtag+  '/' + varname + '_part.pdf')
        idx += 1

        sys.stdout.write( '{0:d} of {1:d} plots done.\r'.format(idx, len(variable_names)))
        if not i == len(variable_names): sys.stdout.flush()
        plt.close()
