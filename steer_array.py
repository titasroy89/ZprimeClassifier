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


jobid = int(sys.argv[1]) # = 1,2,3,4,5,6...

values_layers = [[256, 256]]
values_batchsize = [8192]
values_regrate = [0.00100]
values_runonfraction = [1.00, 0.05, 0.01]
values_eqweight = [False]

idx_layers          = (jobid-1)  % len(values_layers)
idx_batchsize       = ((jobid-1) / (len(values_layers))) % len(values_batchsize)
idx_regrate         = ((jobid-1) / (len(values_layers)) / (len(values_batchsize))) % len(values_regrate)
idx_runonfraction   = ((jobid-1) / (len(values_layers)) / (len(values_batchsize)) / (len(values_regrate))) % len(values_runonfraction)
idx_eqweight = ((jobid-1) / (len(values_layers)) / (len(values_batchsize)) / (len(values_regrate)) / (len(values_runonfraction))) % len(values_eqweight)

# parameters = {'layers':values_layers[idx_layers],
#               'batchsize': values_batchsize[idx_batchsize],
#               'classes':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['DYJets'], 3:['WJets'], 4:['ST']},
#               'regrate':values_regrate[idx_regrate],
#               'epochs':20,
#               'runonfullsample':True,
#               'eqweight':values_eqweight[idx_eqweight]}
parameters = {'layers':values_layers[idx_layers],
              'batchsize': values_batchsize[idx_batchsize],
              'classes':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['DYJets'], 3:['WJets'], 4:['ST']},
              'regmethod': 'dropout',
              'regrate':values_regrate[idx_regrate],
              'batchnorm': True,
              'epochs':300,
              'learningrate': 0.00100,
              'runonfraction':values_runonfraction[idx_runonfraction],
              'eqweight':values_eqweight[idx_eqweight]}


GetInputs(parameters)
# PlotInputs(parameters)

# TrainNetwork(parameters)
# # TrainOnFails(parameters)
# PlotPerformance(parameters, True)

# RankNetworks(parameters)
