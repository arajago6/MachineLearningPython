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
