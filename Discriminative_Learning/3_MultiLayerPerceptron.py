# Import the necessary packages
import numpy as np
import random
from ml_lib import *
import copy
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")


# Returns predictions with the input test data, class labels, hidden layer parameters, output layer parameters and hidden unit count
def getEstimate(attributes,olParam,hlParam,otcmVal,huCount,testingEstimate):
    for i in range(attributes.shape[0]):
        prevEst, otcmMax = 0, 0
        currEst = []
        hlOutput = [logisticFunction(hlParam[j],attributes[i]) for j in range(huCount)]

        for y in range(len(otcmVal)):
            currEst.append(softmaxFunction(olParam,hlOutput,y,otcmVal))
            if(currEst[y] > prevEst):
                otcmMax = y
                prevEst = currEst[y]
        testingEstimate.append(otcmVal[otcmMax])
    return testingEstimate   
