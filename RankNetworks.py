import numpy as np
from numpy import inf
import keras
import matplotlib
import math
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
from keras import metrics
import pickle
import os
from functions import *

def RankNetworks(parameters):

    # Get all the model folders
    modellist = os.listdir('output')

    # Loop over them, open the correct file
    modelresults_lumiweighted = {}
    modelresults_equallyweighted = {}
    for tag in modellist:
        if not os.path.isdir('output/' + tag): continue
        infile = open('output/'+tag+'/ModelPerformance.txt','r')
        lines = infile.readlines()
        for line in lines:
            if 'Tag:' == line[0:4]:
                tag = line.split(' ')[1]
            elif '    ' == line[0:4]:
                line = line.replace(' ','')
                vallosses = line.split('--')[1]
                vallossmin = vallosses.split(',')[0].replace('(','')
                vallossfin = vallosses.split(',')[1].replace(')','')
                # print line
                # print vallossmin, vallossfin
        if 'equallyweighted_False' in tag:
            modelresults_lumiweighted[tag] = [float(vallossmin), float(vallossfin)]
        elif 'equallyweighted_True' in tag:
            modelresults_equallyweighted[tag] = [float(vallossmin), float(vallossfin)]
        else:
            raise ValueError('the part "equallyweighted_(True|False)" is not in the tag.')
        infile.close()

    outfile = open('output/ModelRanking_lumiweighted.txt', 'w')
    idx = 0
    for tag in sorted(modelresults_lumiweighted, key=modelresults_lumiweighted.get, reverse=False):
        text = 'Rank: ' + str(idx) + '\nTag: ' + tag + 'Losses: ' + str(modelresults_lumiweighted[tag]) + '\n\n'
        outfile.write(text)
        idx=idx+1
    outfile.close()
    outfile = open('output/ModelRanking_equallyweighted.txt', 'w')
    idx = 0
    for tag in sorted(modelresults_equallyweighted, key=modelresults_equallyweighted.get, reverse=False):
        text = 'Rank: ' + str(idx) + '\nTag: ' + tag + 'Losses: ' + str(modelresults_equallyweighted[tag]) + '\n\n'
        outfile.write(text)
        idx=idx+1
    outfile.close()
