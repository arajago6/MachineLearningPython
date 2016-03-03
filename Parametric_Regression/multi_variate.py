import numpy as np
import copy
from ml_lib import *
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist

# Map data to higher dimensional space
def mapHD(inptData,degree):
    return PolynomialFeatures(degree).fit_transform(inptData)


def solve(inptData,outcomes):
    return np.dot(np.linalg.pinv(inptData),outcomes)


def crossValidate(zMatrix, outcomes, foldCount):
	thetaValList = []
    	trainingErrorList =[]
    	testingErrorList = []
	featSize = zMatrix.shape[1]

	zMatrixFolds = np.asarray(getFolds(zMatrix,foldCount))
	otcmFolds = np.asarray(getFolds(outcomes,foldCount))

	testDataList = copy.copy(zMatrixFolds)
	testOtcmList = copy.copy(otcmFolds)

	for itr in range(foldCount):
		trainDataList = []
		trainOtcmList = []
		for intitr in range (foldCount):
			if intitr != itr:
				trainDataList.append(zMatrixFolds[intitr]) 
				trainOtcmList.append(otcmFolds[intitr])

		trainDataArr = 	np.array(trainDataList).reshape(-1,featSize)
		trainOtcmArr =  np.array(trainOtcmList).reshape(-1,1)
		testDataArr = np.array(testDataList[itr])
		testOtcmArr = np.array(testOtcmList[itr])

		thetaVal  = solve(trainDataArr,trainOtcmArr)
		trainingEstimate = estimate(thetaVal,trainDataArr) 
		testingEstimate = estimate(thetaVal,testDataArr)

		trainingError = meanSqrError(trainingEstimate, trainOtcmArr)
		testingError = meanSqrError(testingEstimate,testOtcmArr)
	
		thetaValList.append(thetaVal)
        	trainingErrorList.append(trainingError)
        	testingErrorList.append(testingError)
	
	return zMatrix, thetaValList, trainingErrorList, testingErrorList

