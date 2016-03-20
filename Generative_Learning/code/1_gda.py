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


# This function based on the membership value, decides on the final outcome prediction
def discrimFn(mem1, mem2, outcome):
	return outcome[0] if mem1>mem2 else outcome[1]


# This function returns a list of predictions for input records, using discrimFn
def gda1D(x,sMean,sVar,otcmVal,pClassProb):
	otcmGuess = []
	for i in range(len(x)):
		otcmGuess.append(discrimFn(memberFn(x[i],sMean[0],sVar[0],pClassProb[0]),memberFn(x[i],sMean[1],sVar[1],pClassProb[1]),otcmVal))
	return otcmGuess


# This function, splits the input attributes and outcomes into specified folds, gets prediction and metrics for each step 
# of training and testing using own function and inbuilt sklearn function, based on the ownFunction flag
def crossValidate(attributes, outcomes, foldCount, ownFunction=True):
    	presList =[]; recallList = []
	accrList = []; fMeasList = []
	testingEstimate = []
	featLen = 1; otcmVal = list(set(outcomes))

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

		trainDataArr = 	np.array(trainDataList).reshape(-1,featLen)
		trainOtcmArr =  np.array(trainOtcmList).reshape(-1)
		testDataArr = np.array(testDataList[itr]).reshape(-1,featLen)
		testOtcmArr = np.array(testOtcmList[itr]).reshape(-1)
		
		if ownFunction:
			# Predicting outcomes using own gda1D function
			muVal = [getSpecificMean(trainDataArr,trainOtcmArr,1),getSpecificMean(trainDataArr,trainOtcmArr,2)] 
			varVal = [getSpecificVar(trainDataArr,trainOtcmArr,1),getSpecificVar(trainDataArr,trainOtcmArr,2)] 
			pClassProb = [len([trainOtcmArr[i] for i in range(len(trainOtcmArr)) if trainOtcmArr[i]==1])/float(len(trainOtcmArr)),
						len([trainOtcmArr[i] for i in range(len(trainOtcmArr)) if trainOtcmArr[i]==2])/float(len(trainOtcmArr))												]
			testingEstimate = gda1D(testDataArr,muVal,varVal,otcmVal,pClassProb)
		else:		
			# Predicting outcomes using inbuilt predict function
			clf = LinearDiscriminantAnalysis()
			clf.fit(trainDataArr,trainOtcmArr)
			trainingEstimate = clf.predict(trainDataArr) 
			testingEstimate = clf.predict(testDataArr)

		metric = getMetrics(testOtcmArr,testingEstimate,otcmVal)
		accrList.append(metric[0])
		presList.append(metric[1])
		recallList.append(metric[2])
		fMeasList.append(metric[3])
		
	return accrList, presList, recallList, fMeasList
