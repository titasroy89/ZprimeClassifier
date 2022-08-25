#!/usr/bin/env python

import numpy as np
from numpy import inf
import keras
import random
import tensorflow as tf
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
#from TrainModelOnPredictions import *
#from TrainSecondNetwork import *
#from TrainThirdNetwork import *
#from ExportModel import *

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

#variations = ['NOMINAL','JEC_up','JEC_down','JER_up','JER_down']
#variations = ['JEC_up','JEC_down']
#variations = ['JER_up','JER_down']
#variations = ['JEC_up']
#variations = ['JEC_down']
variations = ['NOMINAL']
merged_str = 'Merged'
channel = 'muon'
parameters = {
    'layers':[512,512], #512
    'batchsize': 13107,#131072
    #'classes':{0: ['QCD'], 1:['TTbar_Semi_1','TTbar_Semi_2','TTbar_Semi_3','TTbar_Semi_4','TTbar_Other','ST','WJets','DY','Diboson']},
    'classes':{0: ['QCD'], 1:['TTbar_All','ST','WJets','DYJets','Diboson']},
    'regmethod': 'dropout',
    'regrate':0.50,
    'batchnorm': False,
    'epochs':500, #300
    'learningrate': 0.0005,
    'decay_steps':1.0,
    'decay_rate': 0.0025,
    'runonfraction': 1.0,
    'eqweight':False,
    'preprocess': 'MinMaxScaler',
    'sigma': 1.0, #sigma for Gaussian prior (BNN only)
    'inputdir': '../../MLInputs_mu_UL/',
    'inputsubdir': 'MLInput/', #path to input files: inputdir + systvar + inputsubdir
    'prepreprocess': 'RAW' #for inputs with systematics don't do preprocessing before merging all inputs on one,     #FixME: add prepreprocessing in case one does not need to merge inputs
}

tag = dict_to_str(parameters)
print "tag is:",tag
classtag = get_classes_tag(parameters)
print "classtag is:",classtag

########## GetInputs ########
for ivars in range(len(variations)):
     merged_str = merged_str+'__'+variations[ivars]
     parameters['systvar'] = variations[ivars]
     # # # # # # Get all the inputs
     # # # # # # # # # ==================
     inputfolder = parameters['inputdir']+parameters['inputsubdir']+parameters['systvar']+'/'+parameters['prepreprocess']+'/'+ classtag
     GetInputs(parameters)
     print "I am here"
     PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder='Plots/'+parameters['prepreprocess']+'/InputDistributions/'+parameters['systvar']+'/' + classtag)

MixInputs(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, variations=variations, filepostfix='')
print "done with MixInputs"
SplitInputs(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')
print "done with SplitInputs"
FitPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='train')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='test')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='val')

inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag
outputfolder='output/'+parameters['preprocess']+'/'+merged_str+'/' + classtag+'/DNN_'+channel+'_'+tag
plotfolder = 'Plots/'+parameters['preprocess']
#PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder=plotfolder+'/InputDistributions/'+merged_str+'_'+channel+'/' + classtag)

########
tf.keras.utils.set_random_seed(0)
# # DNN 
TrainNetwork(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+channel)
print outputfolder
PredictExternal(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+channel, filepostfix='')

print "outputfolder:", 'output/'+parameters['preprocess']+'/DNN_'+tag
PlotPerformance(parameters, inputfolder=parameters['inputdir']+'/'+parameters['preprocess']+'/'+merged_str+'/'+classtag,outputfolder='output/'+parameters['preprocess']+'/DNN_'+channel,filepostfix='', plotfolder='Plots/'+parameters['preprocess']+'/DNN_'+channel, use_best_model=True, usesignals=[2,4])


