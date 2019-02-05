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
import sys
import pickle
from Training import *
from Plotting import *
from GetInputs import *
from functions import *

jobid = int(sys.argv[1]) # = 1,2,3,4,5,6...

values_layers = [[128, 128, 32], [256, 256, 32], [64, 64, 16], [64, 64, 64, 16], [128, 128, 128, 32], [256, 256, 256, 32]]
values_batchsize = [512, 2048, 8192]
values_dropoutrate = [0.01, 0.03, 0.1, 0.3]
values_equallyweighted = [True, False]

idx_layers          = (jobid-1)%len(values_layers)
idx_batchsize       = ((jobid-1) / (len(values_layers))) % len(values_batchsize)
idx_dropoutrate     = ((jobid-1) / (len(values_layers)) / (len(values_batchsize))) % len(values_dropoutrate)
idx_equallyweighted = ((jobid-1) / (len(values_layers)) / (len(values_batchsize)) / (len(values_dropoutrate))) % len(values_equallyweighted)

parameters = {'layers':values_layers[idx_layers],
              'batchsize': values_batchsize[idx_batchsize],
              'sampleclasses':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['DYJets'], 3:['WJets'], 4:['ST']},
              'dropoutrate':values_dropoutrate[idx_dropoutrate],
              'epochs':20,
              'runonfullsample':True,
              'equallyweighted':values_equallyweighted[idx_equallyweighted]}


# GetInputs(parameters)
# PlotInputs(parameters)


# TrainNetwork(parameters)
PlotPerformance(parameters, True)
