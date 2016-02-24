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


if __name__ == "__main__":

	#dataFiles = ["svar-set1.dat", "svar-set2.dat", "svar-set3.dat", "svar-set4.dat"]
	dataFiles = ["svar-set2.dat"]
	attList = []; otcmList = []; foldCount = 10
	for iterator in range(0,len(dataFiles)):
		attributes = []; outcomes = []
		attributes, outcomes = readData(dataFiles, iterator, attributes, outcomes)
		attList.append(attributes); otcmList.append(outcomes); 

		# View data
		drawPlot([attributes,outcomes],'Plot of '+dataFiles[iterator])
		
		# Fit linear model to data using own function
		zMatrix, thetaValues,trainingErrors,testingErrors = crossValidate(attributes,outcomes,foldCount,1,ownFunction = True)
		zMatArr = np.array(zMatrix); otcmArr = np.array(outcomes)
		print("Linear Model - Own Function - 10 fold cross validation")
		print("Theta Values")
		print np.array(thetaValues).reshape(-1,2)
		print("Training Error\t\tTesting Error")
		for itr in range(foldCount):
    			print "%f\t\t%f" %(trainingErrors[itr][0],testingErrors[itr][0])
		print "Mean Errors: Training: %f --- Testing: %f\n\n" %(np.mean(trainingErrors),np.mean(testingErrors))
		pltInput = [attributes,outcomes,attributes,estimate(solveLinearModel(zMatArr,otcmArr),zMatArr)]
		drawPlot(pltInput,'Own Function - Linear Model for '+dataFiles[iterator])

		# Fit linear model to data using pre-built python function
		zMatrix, thetaValues,trainingErrors,testingErrors = crossValidate(attributes,otcmList[0],foldCount,1,ownFunction = False)
		zMatArr = np.array(zMatrix); otcmArr = np.array(outcomes)
		print("Linear Model - Pre-built Python Function - 10 fold cross validation")
		print("Theta Values")
		print np.array(thetaValues)
		print("Training Error\t\tTesting Error")
		for itr in range(foldCount):
    			print "%f\t\t%f" %(np.array(trainingErrors[itr]),np.array(testingErrors[itr]))
		print "Mean Errors: Training: %f --- Testing: %f\n\n" %(np.mean(trainingErrors),np.mean(testingErrors))
		lR = LinearRegression(); lR.fit(zMatArr,otcmArr);
		pltInput = [attributes,outcomes,attributes,lR.predict(zMatArr)]
		drawPlot(pltInput,'Pre-built Python Function - Linear Model for '+dataFiles[iterator])

		# Fit degree 2 polynomial model to a subset of data
		zMatrix, thetaValues,trainingErrors,testingErrors = crossValidate(attributes[:50],otcmList[0][:50],foldCount,2,ownFunction = True)
		zMatArr = np.array(zMatrix); otcmArr = np.array(otcmList[0][:50])
		print("Degree 2 Polynomial Model - Own Function - 10 fold cross validation")
		print("Theta Values")
		print np.array(thetaValues)
		print("Training Error\t\tTesting Error")
		for itr in range(foldCount):
    			print "%f\t\t%f" %(np.array(trainingErrors[itr]),np.array(testingErrors[itr]))
		print "Mean Errors: Training: %f --- Testing: %f\n\n" %(np.mean(trainingErrors),np.mean(testingErrors))
		pltInput = [attributes[:50],outcomes[:50],attributes[:50],estimate(solvePolyModel(zMatArr,otcmArr),zMatArr)]
		drawPlot(pltInput,'Own Function - Degree 2 Polynomial Model for '+dataFiles[iterator])
