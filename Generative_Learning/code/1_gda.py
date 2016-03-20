# Importing the necessary packages
import numpy as np, copy
from ml_lib import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# This function gets mean of a specific column for the records of given class
def getSpecificMean(attributes, outcomes, dsrdOutcome, colNum=0):
	dsrdOList = ([attributes[i] for i in range(len(outcomes)) if outcomes[i]==dsrdOutcome])
	return np.mean(np.array(dsrdOList)[:,colNum])


# This function gets variance of a specific column for the records of given class
def getSpecificVar(attributes, outcomes, dsrdOutcome, colNum=0):
	dsrdOList = ([attributes[i] for i in range(len(outcomes)) if outcomes[i]==dsrdOutcome])
	return np.std(np.array(dsrdOList)[:,colNum])


# This function calculates the membership
def memberFn(x,sMean,sVar,pClassProb):
	return (-np.log(sVar)-((0.5)*((x-sMean)**2/sVar**2))+np.log(pClassProb))
