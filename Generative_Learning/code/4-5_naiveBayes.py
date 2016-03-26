# Importing the necessary packages
import numpy as np, pylab as pl, math, operator, collections, copy
from ml_lib import *
import warnings
warnings.filterwarnings("ignore")

# Defining needed global variables
sConst = 1.0; # Laplace Smoothing constant
foldCount = 10; docLength = 5574
otcmVal = ['spam','nospam']


# This function, splits the input attributes and outcomes into specified folds, gets prediction and metrics for each step 
# of training and testing, based on the bernoulli flag. If set, the flag makes the feature distribution as Bernoulli. 
def crossValidate(attributes, outcomes, foldCount, bernoulli=False):
	tup = (0,0,0,0,0); featLen = 1; 
	global otcmVal

	attrFolds = getFolds(attributes,foldCount)
	otcmFolds = getFolds(outcomes,foldCount)

	testDataList = copy.copy(attrFolds)
	testOtcmList = copy.copy(otcmFolds)
	
	for itr in range(foldCount):
		trainDataList = []
		trainOtcmList = []
		for intitr in range (foldCount):
			if intitr != itr:
				trainDataList.append(attrFolds[intitr]) 
				trainOtcmList.append(otcmFolds[intitr])
