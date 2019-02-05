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


# parameters = {'layers':[128, 128, 16],
#               'batchsize': 8192,
#               'sampleclasses':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['DYJets'], 3:['WJets'], 4:['ST']},
#               'dropoutrate':0.01,
#               'epochs':10,
#               'runonfullsample':True,
#               'equallyweighted':True}
parameters = {'layers':[256, 256, 256, 32],
              'batchsize': 8192,
              'sampleclasses':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['DYJets'], 3:['WJets'], 4:['ST']},
              'dropoutrate':0.01,
              'epochs':20,
              'runonfullsample':True,
              'equallyweighted':True}


# GetInputs(parameters)
# PlotInputs(parameters)


# TrainNetwork(parameters)
PlotPerformance(parameters, True)

# RankNetworks(parameters)
