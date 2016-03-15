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


def gradDescent(zMatrix, otcm, guessedTheta, lRate, iterNum):
    thetaVal = np.ones(len(zMatrix[0])); thetaVal.fill(guessedTheta)
    for i in range(iterNum):
        estd = estimate(thetaVal,zMatrix)
        totThetaVal = np.zeros(len(zMatrix[0]))
        for i in range(len(estd)):
            totThetaVal = totThetaVal + ((estd[i] - otcm[i])*zMatrix[i])
        thetaVal = thetaVal - lRate*totThetaVal
    return thetaVal


def iterSolution(zMatrix, otcm, guessedTheta, lRate, iterNum,ztMatrix):
    return estimate(np.array(gradDescent(zMatrix, otcm, guessedTheta, lRate, iterNum)),ztMatrix)


def explicitSolution(zMatrix, otcm,ztMatrix):
    return estimate(solve(zMatrix,otcm),ztMatrix)


def gaussianFit(ip,otcm,sigma):
    return np.linalg.solve(np.exp((-1/2)*(cdist(ip,ip,'sqeuclidean') /sigma**2)),otcm)


def gaussianEstimate(alpha,x,xEx,sigma):
    return np.dot(alpha.transpose(),np.exp((-1/2)*(cdist(x,xEx,'sqeuclidean')/sigma**2)))


if __name__ == "__main__":

	#dataFiles = ["mvar-set1.dat", "mvar-set2.dat", "mvar-set3.dat", "mvar-set4.dat"]
	dataFiles = ["mvar-set1.dat"]
	attList = []; otcmList = []; zz = 0
	dataInHD = {}; thetaDict = {}; estimateOnHD = {}; meanError = {}
	for iterator in range(0,len(dataFiles)):
		attributes = []; outcomes = []
		attributes, outcomes = readData(dataFiles, iterator, attributes, outcomes)
