#!/usr/bin/env python

import numpy as np
from numpy import inf
import keras
import matplotlib
matplotlib.use('Agg')
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
import math
import pickle
from Training import *
from Plotting import *
from GetInputs import *
from RankNetworks import *
from functions import *
from PredictExternal import *
from TrainSecondNetwork import *


jobid = int(sys.argv[1]) # = 1,2,3,4,5,6...

values_layers = [[256, 256], [512, 512], [512, 512, 512], [1024, 1024]]
values_batchsize = [8192, 131072]
values_regrate = [0.00000, 0.0100, 0.0500, 0.2000, 0.5000]
values_epochs = [500]
values_runonfraction = [1.00]
values_eqweight = [False]

idx_layers          = (jobid-1)  % len(values_layers)
idx_batchsize       = ((jobid-1) / (len(values_layers))) % len(values_batchsize)
idx_regrate         = ((jobid-1) / (len(values_layers)) / (len(values_batchsize))) % len(values_regrate)
idx_epochs          = ((jobid-1) / (len(values_layers)) / (len(values_batchsize)) / (len(values_regrate))) % len(values_epochs)
idx_runonfraction   = ((jobid-1) / (len(values_layers)) / (len(values_batchsize)) / (len(values_regrate)) / (len(values_epochs))) % len(values_runonfraction)
idx_eqweight        = ((jobid-1) / (len(values_layers)) / (len(values_batchsize)) / (len(values_regrate)) / (len(values_epochs)) / (len(values_runonfraction))) % len(values_eqweight)

parameters = {'layers':[512, 512],
              'batchsize': 8192,
              'classes':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['DYJets'], 3:['WJets'], 4:['ST']},
              # 'classes':{0: ['QCD_Mu'], 1: ['TTbar', 'DYJets', 'WJets', 'ST']},
              # 'classes':{0: ['TTbar'], 1:['DYJets'], 2:['WJets'], 3:['ST']},
              # 'classes':{0: ['TTbar'], 1:['ST']},
              # 'classes':{0: ['TTbar'], 1: ['DYJets', 'WJets', 'ST']},
              'regmethod': 'dropout',
              'regrate':0.20000,
              'batchnorm': True,
              'epochs':500,
              'learningrate': 0.00100,
              'runonfraction': 1.00,
              'eqweight':False}



parameters2 = {'layers':values_layers[idx_layers],
              'batchsize': values_batchsize[idx_batchsize],
              'classes':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['DYJets'], 3:['WJets'], 4:['ST']},
              'regmethod': 'dropout',
              'regrate':values_regrate[idx_regrate],
              'batchnorm': True,
              'epochs':values_epochs[idx_epochs],
              'learningrate': 0.00100,
              'runonfraction':values_runonfraction[idx_runonfraction],
              'eqweight':values_eqweight[idx_eqweight]}


tag = dict_to_str(parameters)
tag2 = dict_to_str(parameters2)
classtag = get_classes_tag(parameters)
classtag2 = get_classes_tag(parameters2)




# GetInputs(parameters)
# PlotInputs(parameters)

# TrainNetwork(parameters)
# PredictExternal(parameters, inputfolder='input/'+classtag, outputfolder='output/'+tag, filepostfix='')
# # PlotPerformance(parameters, inputfolder='input/'+classtag, outputfolder='output/'+tag, filepostfix='', plot_distributions=True, use_best_model=False, usesignal=4)
# PlotPerformance(parameters, inputfolder='input/'+classtag, outputfolder='output/'+tag, filepostfix='', plotfolder='Plots/'+tag, plot_distributions=True, use_best_model=True, usesignal=4)


TrainSecondNetwork(parameters, parameters2, use_best_model=True)
PredictExternal(parameters, inputfolder='output/'+tag+'/cut', filepostfix='_fail_best', outputfolder='output/'+tag+'/SecondModels/output/'+tag2)
PlotPerformance(parameters, inputfolder='output/'+tag+'/cut', outputfolder='output/'+tag+'/SecondModels/output/'+tag2, filepostfix='_fail_best', plotfolder='output/'+tag+'/SecondModels/Plots/'+tag2, plot_distributions=True, use_best_model=True, usesignal=4)


# RankNetworks(parameters)
