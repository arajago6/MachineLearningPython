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


def solveLinearModel(dataArr, outcomes):
	attArr = dataArr[:,1]
	attSum = sum(attArr)
	aMatrix = [[len(attArr),attSum], [attSum, sum(attArr**2)]]
    	bMatrix = [[sum(outcomes)], [sum(attArr*outcomes)]]
	return np.linalg.solve(np.array(aMatrix),np.array(bMatrix))


def solvePolyModel(dataArr,outcomes):
    return np.linalg.solve((np.dot(dataArr.transpose(),dataArr)),
                           (np.dot(dataArr.transpose(),outcomes)))	


def crossValidate(attributes, outcomes, foldCount, degree, ownFunction = True):
	thetaValList = []
    	trainingErrorList =[]
    	testingErrorList = []

	zMatrix = getDataMatrix(attributes,degree)
	zMatrixFolds = getFolds(zMatrix,foldCount)
	otcmFolds = getFolds(outcomes,foldCount)

	testDataList = copy.copy(zMatrixFolds)
	testOtcmList = copy.copy(otcmFolds)
