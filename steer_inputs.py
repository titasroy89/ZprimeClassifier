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
from PredictExternal import *
from functions import *
from TrainModelOnPredictions import *
from TrainSecondNetwork import *
from TrainThirdNetwork import *
from ExportModel import *

# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

variations = ['NOMINAL','JEC_up','JEC_down','JER_up','JER_down']
merged_str = 'Merged'
parameters = {
    'layers':[128, 128],
    'batchsize': 131072,
    'classes':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['WJets'], 3:['DYJets','ST']},
    'regmethod': 'dropout',
    'regrate':0.60000,
    'batchnorm': False,
    'epochs':700,
    'learningrate': 0.00050,
    'runonfraction': 0.20,
    'eqweight':False,
    'preprocess': 'MinMaxScaler',
    'sigma': 0.5, #sigma for Gaussian prior (BNN only)
    'inputdir': 'input/2017_Moriond19JEC_RightLumiweights_forml_morevar_Puppi/',#general path to inputs with systematic variations
    #        'systvar': variations[ivars],
    'inputsubdir': '/MLInput_Reduced/', #path to input files: inputdir + systvar + inputsubdir
    'prepreprocess': 'RAW' #for inputs with systematics don't do preprocessing before merging all inputs on one,     #FixME: add prepreprocessing in case one does not need to merge inputs
}

tag = dict_to_str(parameters)
classtag = get_classes_tag(parameters)

for ivars in range(len(variations)):
    merged_str = merged_str+'__'+variations[ivars]
    parameters['systvar'] = variations[ivars]
    # # # # # # # Get all the inputs
    # # # # # # # # # ==================
    #inputfolder = parameters['inputdir']+parameters['systvar']+parameters['inputsubdir']+parameters['prepreprocess']+'/'+ classtag
    # GetInputs(parameters)
    #PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder='Plots/'+parameters['prepreprocess']+'/InputDistributions/'+parameters['systvar']+'/' + classtag)
    
# #FixMe: preprocessing and storage for merge outputs
# MixInputs(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, variations=variations, filepostfix='')
# SplitInputs(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')
# FitPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')
# ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='train')
# ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='test')
# ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='val')
# ApplySignalPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')

PlotInputs(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='', plotfolder='Plots/' +parameters['preprocess']+'/InputDistributions/'+merged_str+'/' + classtag)
