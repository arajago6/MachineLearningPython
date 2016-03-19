import numpy as np
import pandas as pd
import copy
from ml_lib import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def getSpecificMean(attributes, outcomes, dsrdOutcome, colNum=0):
	dsrdOList = ([attributes[i] for i in range(len(outcomes)) if outcomes[i]==dsrdOutcome])
	return np.mean(np.array(dsrdOList)[:,colNum])


def getSpecificVar(attributes, outcomes, dsrdOutcome, colNum=0):
	dsrdOList = ([attributes[i] for i in range(len(outcomes)) if outcomes[i]==dsrdOutcome])
	return np.std(np.array(dsrdOList)[:,colNum])
