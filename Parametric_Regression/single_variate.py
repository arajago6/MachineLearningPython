import numpy as np
import copy
from ml_lib import *
from sklearn.linear_model import LinearRegression


def getDataMatrix(attributes, degree):
	zMatrix = np.ones((len(attributes),degree+1),np.float)
	if degree == 1:
		zMatrix[:,1] = attributes
	else:
		for itr in range(1,degree+1):
			zMatrix[:,itr] = np.array(attributes)**itr	
	return zMatrix
