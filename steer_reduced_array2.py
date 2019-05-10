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
from TrainingReduced import *
from Plotting import *
from GetInputsReduced import *
from RankNetworks import *
from PredictExternal import *
from functions import *
from ExportModel import *


jobid = int(sys.argv[1]) # = 1,2,3,4,5,6...


values_layers = [[512, 512]]
values_batchsize = [131072]
values_regrate = [0.2500, 0.3000, 0.3500, 0.4000, 0.4500]
values_epochs = [500]
values_learningrate = [0.001]
values_runonfraction = [1.00]
values_eqweight = [False]

idx_layers          = (jobid-1)  % len(values_layers)
idx_batchsize       = ((jobid-1) / (len(values_layers))) % len(values_batchsize)
idx_regrate         = ((jobid-1) / (len(values_layers)) / (len(values_batchsize))) % len(values_regrate)
idx_epochs          = ((jobid-1) / (len(values_layers)) / (len(values_batchsize)) / (len(values_regrate))) % len(values_epochs)
idx_learningrate    = ((jobid-1) / (len(values_layers)) / (len(values_batchsize)) / (len(values_regrate)) / (len(values_epochs))) % len(values_learningrate)
idx_runonfraction   = ((jobid-1) / (len(values_layers)) / (len(values_batchsize)) / (len(values_regrate)) / (len(values_epochs)) / (len(values_learningrate))) % len(values_runonfraction)
idx_eqweight        = ((jobid-1) / (len(values_layers)) / (len(values_batchsize)) / (len(values_regrate)) / (len(values_epochs)) / (len(values_learningrate)) / (len(values_runonfraction))) % len(values_eqweight)




parameters = {'layers':values_layers[idx_layers],
              'batchsize': values_batchsize[idx_batchsize],
              'classes':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['DYJets'], 3:['WJets'], 4:['ST']},
              'regmethod': 'dropout',
              'regrate':values_regrate[idx_regrate],
              'batchnorm': False,
              'epochs':values_epochs[idx_epochs],
              'learningrate': values_learningrate[idx_learningrate],
              'runonfraction':values_runonfraction[idx_runonfraction],
              'eqweight':values_eqweight[idx_eqweight]}


tag = dict_to_str(parameters)
classtag = get_classes_tag(parameters)



# # First network
# # =============
TrainNetworkReduced(parameters)
# PredictExternal(parameters, inputfolder='input/'+classtag, outputfolder='output/'+tag, filepostfix='')
PlotPerformance(parameters, inputfolder='input_reduced/'+classtag, outputfolder='output_reduced/'+tag, filepostfix='', plotfolder='Plots_reduced/'+tag, use_best_model=True, usesignals=[2,4])
