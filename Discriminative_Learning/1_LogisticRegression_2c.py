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
