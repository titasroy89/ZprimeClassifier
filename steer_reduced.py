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





parameters = {
              'layers':[512, 512],
              'batchsize': 131072,
              'classes':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['DYJets'], 3:['WJets'], 4:['ST']},
              'regmethod': 'dropout',
              'regrate':0.35000,
              'batchnorm': False,
              'epochs':500,
              'learningrate': 0.00100,
              'runonfraction': 1.00,
              'eqweight':False}

tag = dict_to_str(parameters)
classtag = get_classes_tag(parameters)


# # Get all the inputs
# # ==================
# GetInputsReduced(parameters)
# PlotInputs(parameters, inputfolder='input_reduced/'+classtag, filepostfix='', plotfolder='Plots_reduced/InputDistributions/' + classtag)

# # First network
# # =============
# TrainNetworkReduced(parameters)
PredictExternal(parameters, inputfolder='input_reduced/'+classtag, outputfolder='output_reduced/'+tag, filepostfix='')
PlotPerformance(parameters, inputfolder='input_reduced/'+classtag, outputfolder='output_reduced/'+tag, filepostfix='', plotfolder='Plots_reduced/'+tag, use_best_model=True, usesignals=[2,4])
# ExportModel(parameters, inputfolder='input_reduced/', outputfolder='output_reduced/', use_best_model=True)
# RankNetworksReduced(outputfolder='output_reduced/')
