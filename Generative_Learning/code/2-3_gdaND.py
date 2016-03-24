# Importing the necessary packages
import numpy as np, copy, operator
from ml_lib import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import warnings
warnings.filterwarnings("ignore")


# This function calculates the membership
def gdaNDMemberFn(x,sMean,sSig,pClassProb):
	return(-np.log(np.linalg.det(sSig))- 0.5*(np.dot(np.dot(np.transpose(x-sMean),np.linalg.inv(sSig)),(x-sMean))) + np.log(pClassProb))


# This function gets parameters of the feature distribution
def getParams(trainDataArr,trainOtcmArr,otcmVal,featLen):
	muVal = {}; prior = {}; sig = {}; 
	for each in otcmVal:
		muVal[each] = np.array([getSpecificMean(trainDataArr,trainOtcmArr,each,j) for j in range(featLen)])
		prior[each] = [len([trainOtcmArr[i] for i in range(len(trainOtcmArr)) if trainOtcmArr[i]==each])/float(len(trainOtcmArr))]
		sig[each] = np.array(np.cov([trainDataArr[i] for i in range(len(trainOtcmArr)) if trainOtcmArr[i]==each],rowvar=0))
	return dict({"muVal":muVal,"prior":prior,"sig":sig})


# This function returns the list of predictions for the input data
def gdaNDEstimate(testDataArr,params,otcmVal):
	memValue = {}
	testingEstimate=[]
	for i in range(len(testDataArr)):
		for each in otcmVal:
			mean = np.array(params["muVal"][each])
			prior = np.array(params["prior"][each])
			sig = np.array(params["sig"][each])
			memValue[each] = gdaNDMemberFn(testDataArr[i].reshape(4,-1),mean.reshape(4,-1),sig,prior)
		testingEstimate.append(max(memValue.iteritems(), key=operator.itemgetter(1))[0])
	return testingEstimate
	

# This function, splits the input attributes and outcomes into specified folds, gets prediction and metrics for each step 
# of training and testing using own function and inbuilt sklearn function, based on the ownFunction flag
def crossValidate(attributes, outcomes, foldCount, ownFunction=True):
    	presList =[]; recallList = []
	accrList = []; fMeasList = []
	aucList = []
	testingEstimate = []

	otcmVal = list(set(outcomes))
	params = {}; featLen = 4; 

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
