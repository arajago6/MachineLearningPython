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

	for itr in range(foldCount):
		trainDataList = []
		trainOtcmList = []
		for intitr in range (foldCount):
			if intitr != itr:
				trainDataList.append(zMatrixFolds[intitr]) 
				trainOtcmList.append(otcmFolds[intitr])

		trainDataArr = 	np.array(trainDataList).reshape(-1,degree+1)
		trainOtcmArr =  np.array(trainOtcmList).reshape(-1)
		testDataArr = np.array(testDataList[itr])
		testOtcmArr = np.array(testOtcmList[itr])
		
		if ownFunction:
			if degree == 1:
				thetaVal = solveLinearModel(trainDataArr,trainOtcmArr)	
			else:	
				thetaVal = solvePolyModel(trainDataArr,trainOtcmArr)
			trainingEstimate = estimate(thetaVal,trainDataArr) 
			testingEstimate = estimate(thetaVal,testDataArr)			
		else:			
			lR = LinearRegression()
			lR.fit(trainDataArr,trainOtcmArr)
			thetaVal = lR.coef_
			trainingEstimate = lR.predict(trainDataArr) 
			testingEstimate = lR.predict(testDataArr)

		trainingError = meanSqrError(trainingEstimate, trainOtcmArr)
		testingError = meanSqrError(testingEstimate,testOtcmArr)
	
		thetaValList.append(thetaVal)
        	trainingErrorList.append(trainingError)
        	testingErrorList.append(testingError)
	
	return zMatrix, thetaValList, trainingErrorList, testingErrorList
