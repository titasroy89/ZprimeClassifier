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




parameters = {
#              'layers':[512, 512],
              'layers':[128, 128],
        #      'layers':[256, 256],
              'batchsize': 131072,
             #  'batchsize': 16384,
           #   'batchsize': 2526814,
              #'batchsize': 128,
              # 'classes':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['DYJets'], 3:['WJets'], 4:['ST']},
              #'classes':{0: ['QCD_Mu'], 1: ['TTbar', 'ST'], 2:['DYJets', 'WJets'], 3:['RSGluon_All']},
              'classes':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['WJets'], 3:['DYJets','ST']},
              'regmethod': 'dropout',
              'regrate':0.60000,
              'batchnorm': False,
#              'epochs':700,
              'epochs':400,
#              'epochs':15,
#              'epochs':7000,
              #'epochs':500,
#               'epochs':200,
#               'epochs':50,
#              'epochs':2,
              'learningrate': 0.00050,
#              'learningrate': 0.050,
              'runonfraction': 0.99,
#              'runonfraction': 0.20,
#               'runonfraction': 0.50,
#              'runonfraction': 0.10,
              'eqweight':False,
#              'preprocess': 'StandardScaler',
#              'preprocess': 'QuantileTransformerUniform',
               'preprocess': 'MinMaxScaler',
#               'sigma': 0.25, #sigma for Gaussian prior (BNN only)
#               'sigma': 0.05, #sigma for Gaussian prior (BNN only),
               'sigma': 1.00, #sigma for Gaussian prior (BNN only),
               'systvar': 'NOMINAL',
               'inputdir': 'input/2017_Moriond19JEC_RightLumiweights_forml_morevar_Puppi/',#general path to inputs with systematic variations
    #        'systvar': variations[ivars],
               'inputsubdir': '/MLInput_Reduced/', #path to input files: inputdir + systvar + inputsubdir
               'prepreprocess': 'RAW' #for inputs with systematics don't do preprocessing before merging all inputs on one,     #FixME: add prepreprocessing in case one does not need to merge inputs
}

# ##TEST
# parameters = {
#               'layers':[512, 512],
#               'batchsize': 131072,
#               # 'classes':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['DYJets'], 3:['WJets'], 4:['ST']},
#               'classes':{0: ['QCD_Mu'], 1: ['TTbar', 'ST'], 2:['DYJets', 'WJets'], 3:['RSGluon_All']},
#               'regmethod': 'dropout',
#               'regrate':0.60000,
#               'batchnorm': False,
#               #'epochs':700,
#               'epochs':7,
#               'learningrate': 0.00050,
#               'runonfraction': 1.00,
#               'eqweight':False}



parameters_onpredictions ={'layers':[20, 20],
                            'batchsize': 8192,
                            'classes':parameters['classes'],
                            'regmethod': 'dropout',
                            'regrate':0.01000,
                            'batchnorm': True,
                            'epochs':499,
                            'learningrate': 0.00150,
                            'runonfraction': parameters['runonfraction'],
                            'eqweight':False,
                            'preprocess': parameters['preprocess'],
                            'sigma': 1.0, #sigma for Gaussian prior (BNN only)
}

tag = dict_to_str(parameters)
classtag = get_classes_tag(parameters)
tag_onpredictions = dict_to_str(parameters_onpredictions)
classtag_onpredictions = get_classes_tag(parameters_onpredictions)

#inputfolder='input/'+parameters['preprocess']+'/'+classtag #Default
#outputfolder='output/'+parameters['preprocess']+'/BNN_'+tag #Default


inputfolder='input/2017_Moriond19JEC_RightLumiweights_forml_morevar_Puppi/NOMINAL/MLInput_Reduced/'+parameters['preprocess']+'/'+classtag #NOMINAL
outputfolder='output/NOMINAL/'+parameters['preprocess']+'/BNN_'+tag #NOMINAL
plotfolder = 'Plots/'+parameters['preprocess']
inputfolder_raw='input/2017_Moriond19JEC_RightLumiweights_forml_morevar_Puppi/NOMINAL/MLInput_Reduced/'+parameters['prepreprocess']+'/'+classtag #NOMINAL
outputfolder_raw_BNN='output/NOMINAL/'+parameters['prepreprocess']+'/BNN_'+tag #NOMINAL
outputfolder_raw_DNN='output/NOMINAL/'+parameters['prepreprocess']+'/DNN_'+tag #NOMINAL
plotfolder_raw = 'Plots/'+parameters['prepreprocess']

# # # # # # # Get all the inputs
# # # # # # # # # ==================
#GetInputs(parameters)
#PlotInputs(parameters, inputfolder='input/'+parameters['preprocess']+'/'+classtag, filepostfix='', plotfolder='Plots/InputDistributions/' +parameters['preprocess']+'/' + classtag)
#PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder='Plots/InputDistributions/' +parameters['preprocess']+'/' + classtag)

# # # # # # # #BNN #TEST network with RAW (not pre-processed inputs)
# PlotInputs(parameters, inputfolder=inputfolder_raw, filepostfix='', plotfolder=plotfolder_raw+'/InputDistributions/'+parameters['systvar']+'/' + classtag)
# TrainBayesianNetwork(parameters, inputfolder=inputfolder_raw, outputfolder=outputfolder_raw_BNN)
# PredictExternalBayesianNetwork(parameters, inputfolder=inputfolder_raw, outputfolder=outputfolder_raw_BNN, filepostfix='',nsamples=50)
# #Plot validation results and store model for usage in UHH2, etc
PlotBayesianPerformance(parameters, inputfolder=inputfolder_raw, outputfolder=outputfolder_raw_BNN, filepostfix='', plotfolder=plotfolder_raw+'/BNN_'+tag, use_best_model=False, usesignals=[2,4])


# # # # # # # # # # ### # # # #DNN
# TrainNetwork(parameters, inputfolder=inputfolder_raw, outputfolder=outputfolder_raw_DNN)
# PredictExternal(parameters, inputfolder=inputfolder_raw, outputfolder=outputfolder_raw_DNN, filepostfix='')
#PlotPerformance(parameters, inputfolder=inputfolder_raw, outputfolder=outputfolder_raw_DNN, filepostfix='', plotfolder=plotfolder_raw+'/DNN_'+tag, use_best_model=True, usesignals=[2,4])

# # # # # # # # # # ### # # # #DNN
# TrainNetwork(parameters, inputfolder='input/'+parameters['preprocess']+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag)
# PredictExternal(parameters, inputfolder='input/'+parameters['preprocess']+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='')
# PlotPerformance(parameters, inputfolder='input/'+parameters['preprocess']+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='', plotfolder='Plots/'+parameters['preprocess']+'/DNN_'+tag, use_best_model=True, usesignals=[2,4])


# # # # # # #BNN
#TrainBayesianNetwork(parameters, inputfolder=inputfolder, outputfolder=outputfolder)
#PredictExternalBayesianNetwork(parameters, inputfolder=inputfolder, outputfolder=outputfolder, filepostfix='',nsamples=10)
#Plot validation results and store model for usage in UHH2, etc
#PlotBayesianPerformance(parameters, inputfolder=inputfolder, outputfolder=outputfolder, filepostfix='', plotfolder='Plots/'+parameters['preprocess']+'/BNN_'+tag, use_best_model=False, usesignals=[2,4])

# # DNN debug
#TrainDeepNetwork(parameters)
#PredictExternalDeepNetwork(parameters, inputfolder='input/'+classtag, outputfolder='output/DNN_'+tag, filepostfix='')
#PlotDeepPerformance(parameters, inputfolder='input/'+classtag, outputfolder='output/DNN_'+tag, filepostfix='', plotfolder='Plots/DNN_'+tag, use_best_model=False, usesignals=[2,4])


# #==================

# # # First network
# # # =============
## TrainNetwork(parameters)
# PlotPerformance(parameters, inputfolder='input/'+classtag, outputfolder='output/'+tag, filepostfix='', plotfolder='Plots/'+tag, use_best_model=True, usesignals=[2,4])
# # PredictExternal(parameters, inputfolder='input/'+classtag, outputfolder='output/'+tag, filepostfix='')
# # PlotPerformance(parameters, inputfolder='input/'+classtag, outputfolder='output/'+tag, filepostfix='', use_best_model=False, usesignals=[2,4])

# # PlotInputs(parameters, inputfolder='output/'+tag+'/cut', filepostfix='_pass_best', plotfolder='Plots/'+tag+'/InputDistributions/pass')
# # PlotInputs(parameters, inputfolder='output/'+tag+'/cut', filepostfix='_fail_best', plotfolder='Plots/'+tag+'/InputDistributions/fail')
# # ExportModel(parameters, inputfolder='input/', outputfolder='output/', use_best_model=True)
# # RankNetworks(outputfolder='output/')

# # # Network trained on outputs
# # # ==========================
# # TrainModelOnPredictions(parameters, parameters_onpredictions, use_best_model=True)
# # PredictExternalOnPredictions(parameters, inputfolder='input/'+classtag, inputfolder_predictions='output/'+tag, filepostfix='_best', outputfolder='output/'+tag+'/ModelsOnPredictions/output/'+tag_onpredictions)
# # PlotPerformance(parameters_onpredictions, inputfolder='input/'+classtag, outputfolder='output/'+tag+'/ModelsOnPredictions/output/'+tag_onpredictions, filepostfix='', plotfolder='output/'+tag+'/ModelsOnPredictions/Plots/'+tag_onpredictions, use_best_model=True, usesignals=[2,4])

# # # Second network
# # # ==============
# # TrainSecondNetwork(parameters, parameters2, use_best_model=True)
# # PredictExternal(parameters, inputfolder='output/'+tag+'/cut', filepostfix='_fail_best', outputfolder='output/'+tag+'/SecondModels/output/'+tag2)
# # PlotPerformance(parameters2, inputfolder='output/'+tag+'/cut', outputfolder='output/'+tag+'/SecondModels/output/'+tag2, filepostfix='_fail_best', plotfolder='output/'+tag+'/SecondModels/Plots/'+tag2, use_best_model=True, usesignals=[2,4])
# # RankNetworks(outputfolder='output/'+tag+'/SecondModels/output/')

# # # Third network
# # # ==============
# # TrainThirdNetwork(parameters, parameters2, parameters3, use_best_model=True)
# # PredictExternal(parameters, inputfolder='output/'+tag+'/SecondModels/output/'+tag2+'/cut', filepostfix='_fail_best', outputfolder='output/'+tag+'/SecondModels/output/'+tag2+'/ThirdModels/output/'+tag3)
# # PlotPerformance(parameters3, inputfolder='output/'+tag+'/SecondModels/output/'+tag2+'/cut', outputfolder='output/'+tag+'/SecondModels/output/'+tag2+'/ThirdModels/output/'+tag3, filepostfix='_fail_best', plotfolder='output/'+tag+'/SecondModels/output/'+tag2+'/ThirdModels/Plots/'+tag3, use_best_model=True, usesignals=[2,4])
# # RankNetworks(outputfolder='output/'+tag+'/SecondModels/output/'+tag2+'/ThirdModels/output/')]



# parameters2 ={'layers':[512, 512],
#               'batchsize': 8192,
#               'classes':parameters['classes'],
#               'regmethod': 'dropout',
#               'regrate':0.20000,
#               'batchnorm': True,
#               'epochs':500,
#               'learningrate': 0.00100,
#               'runonfraction': parameters['runonfraction'],
#               'eqweight':False}

# parameters3 ={'layers':[512, 512],
#               'batchsize': 8192,
#               'classes':parameters['classes'],
#               'regmethod': 'dropout',
#               'regrate':0.20000,
#               'batchnorm': True,
#               'epochs':500,
#               'learningrate': 0.00100,
#               'runonfraction': parameters['runonfraction'],
#               'eqweight':False}

# tag2 = dict_to_str(parameters2)
# classtag2 = get_classes_tag(parameters2)
# tag3= dict_to_str(parameters3)
# classtag3 = get_classes_tag(parameters3)
