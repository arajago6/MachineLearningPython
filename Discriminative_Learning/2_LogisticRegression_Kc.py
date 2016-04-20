# Import the necessary packages
import numpy as np
import copy
from ml_lib import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from random import randrange
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore")


# To get the final parameters by using iterative parameter update
def gradDescent(attributes, outcomes, learningRate, iterCountMax, threshold):
    otcmVal = list(set(outcomes))
    params = [float(randrange(1,5))/10000 for x in range(len(otcmVal)*attributes.shape[1])]
    params = np.array(params).reshape(len(otcmVal),attributes.shape[1])
    updatedParams = copy.copy(params)
    
    for itr in range(iterCountMax):
        for y in range(len(otcmVal)):
            paramCorrection=[]
            for x in range(attributes.shape[0]):
                paramCorrection.append((softmaxFunction(params,attributes[x],y,otcmVal) - (1 if outcomes[x] == y else 0)) * attributes[x])
            updatedParams[y] = (params[y] - (learningRate*sum(paramCorrection)))
            
        llhDiff = logLikelihoodFunctionKc(updatedParams,attributes,outcomes,otcmVal) - logLikelihoodFunctionKc(params,attributes,outcomes,otcmVal)
        #if llhDiff < threshold:
            #break
        params = [updatedParams[pm] for pm in range(len(params))]
        
    return params
    
    
# Returns loglikelihood to form update equation and to check whether early stopping criterion has been met    
def logLikelihoodFunctionKc(params,attributes,outcomes,otcmVal):
    llHood = []
    actVal = np.zeros([attributes.shape[0],len(otcmVal)])
    for x in range(attributes.shape[0]):
        intLlHood = []
        for y in range(len(otcmVal)):
            actVal[x,y] = softmaxFunction(params, attributes[x], y, otcmVal)
            intLlHood.append((1 if y == outcomes[x] else 0) * np.log(actVal[x,y]))
        llHood.append(sum(intLlHood))
    return sum(llHood)
