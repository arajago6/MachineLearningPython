# Import the necessary packages
from ml_lib import *
import numpy as np
import copy
from sklearn import preprocessing
from random import randrange
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
    
    
# To get the final parameters by using iterative parameter update
def gradDescentFunction(attributes, outcomes, learningRate, iterCountMax, threshold):
    params = [float(randrange(1,5))/10000 for x in range(attributes.shape[1])]
    for itr in range(iterCountMax):
        paramCorrection=[]
        for intItr in range(attributes.shape[0]):
            paramCorrection.append((logisticFunction(params,attributes[intItr]) - outcomes[intItr]) * attributes[intItr])
        updatedParams = params - (learningRate*sum(paramCorrection))
        llhDiff = logLikelihoodFunction2c(updatedParams,attributes,outcomes) - logLikelihoodFunction2c(params,attributes,outcomes)
        if llhDiff < threshold:
			break
        params = updatedParams
        
    return params
    
    
# Returns loglikelihood to form update equation and to check whether early stopping criterion has been met   
def logLikelihoodFunction2c(params,attributes,outcomes):
    llHood = []
    for x in range(attributes.shape[0]):
        otcmGuess = logisticFunction(params,attributes[x])
        llHood.append(np.log(otcmGuess) if outcomes[x] == 1 else np.log(1-otcmGuess))
    return sum(llHood)


# Returns predictions with the input test data and parameters
def getEstimate(attributes, params):
    testingEstimate = [1 if logisticFunction(params,attributes[i]) > 0.5 else 0 for i in range(0,attributes.shape[0])]
    return testingEstimate


# Calls for gradient descent function and returns final parameters
def logisticRegressionFunction2c(attributes, outcomes, learningRate, iterCountMax, threshold):
    params = gradDescentFunction(attributes, outcomes, learningRate, iterCountMax, threshold)
    return params
    
    
# This function, splits the input attributes and outcomes into specified folds, gets prediction and metrics for each step 
# of training and testing using own function and inbuilt sklearn function, based on the ownFunction flag
def crossValidate(attributes, outcomes, foldCount, degree, learningRate, iterCountMax, threshold, ownFunction = True):
    presList =[]; recallList = []
    accrList = []; fMeasList = []
    otcmVal = list(set(outcomes))
    featLen = attributes.shape[1]

    zMatrix = getDataMatrix(attributes,featLen,degree)
    zMatrixFolds = getFolds(zMatrix,foldCount)
    otcmFolds = getFolds(outcomes,foldCount)

    testDataList = copy.copy(zMatrixFolds)
    testOtcmList = copy.copy(otcmFolds)

    for itr in range(foldCount):
        trainDataList = []
        trainOtcmList = []
        for intitr in range (foldCount):
            if intitr != itr:
                trainDataList.append(zMatrixFolds[intitr]) 
                trainOtcmList.append(otcmFolds[intitr])
                
        trainDataArr = 	np.array(trainDataList).reshape(-1,zMatrix.shape[1])
        trainOtcmArr =  np.array(trainOtcmList).reshape(-1)
        testDataArr = np.array(testDataList[itr])
        testOtcmArr = np.array(testOtcmList[itr])
