# Import the necessary packages
import numpy as np
import copy
from ml_lib import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from random import randrange
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore")


# To get the final parameters by using iterative parameter update
def gradDescent(attributes, outcomes, learningRate, iterCountMax, threshold):
    otcmVal = list(set(outcomes))
    params = [float(randrange(1,5))/10000 for x in range(len(otcmVal)*attributes.shape[1])]
    params = np.array(params).reshape(len(otcmVal),attributes.shape[1])
    updatedParams = copy.copy(params)
    
    for itr in range(iterCountMax):
        for y in range(len(otcmVal)):
            paramCorrection=[]
            for x in range(attributes.shape[0]):
                paramCorrection.append((softmaxFunction(params,attributes[x],y,otcmVal) - (1 if outcomes[x] == y else 0)) * attributes[x])
            updatedParams[y] = (params[y] - (learningRate*sum(paramCorrection)))
            
        llhDiff = logLikelihoodFunctionKc(updatedParams,attributes,outcomes,otcmVal) - logLikelihoodFunctionKc(params,attributes,outcomes,otcmVal)
        #if llhDiff < threshold:
            #break
        params = [updatedParams[pm] for pm in range(len(params))]
        
    return params
    
    
# Returns loglikelihood to form update equation and to check whether early stopping criterion has been met    
def logLikelihoodFunctionKc(params,attributes,outcomes,otcmVal):
    llHood = []
    actVal = np.zeros([attributes.shape[0],len(otcmVal)])
    for x in range(attributes.shape[0]):
        intLlHood = []
        for y in range(len(otcmVal)):
            actVal[x,y] = softmaxFunction(params, attributes[x], y, otcmVal)
            intLlHood.append((1 if y == outcomes[x] else 0) * np.log(actVal[x,y]))
        llHood.append(sum(intLlHood))
    return sum(llHood)
    
    
# Returns predictions with the input test data, class labels and parameters
def getEstimate(attributes, params, otcmVal, testingEstimate):
    for x in range(attributes.shape[0]):
        prevEst, otcmMax = 0, 0
        for y in range(len(otcmVal)):
            currEst = np.dot(np.transpose(params[y]),attributes[x])
            if(currEst > prevEst):
                otcmMax = y
            prevEst = currEst
        testingEstimate.append(otcmVal[otcmMax])
    return testingEstimate


# Calls for gradient descent function and returns final parameters
def logisticRegressionKc(attributes, outcomes, learningRate, iterCountMax, threshold):
    params = gradDescent(attributes, outcomes, learningRate, iterCountMax, threshold)
    return params
    
    
# This function, splits the input attributes and outcomes into specified folds, gets prediction and metrics for each step 
# of training and testing using own function and inbuilt sklearn function, based on the ownFunction flag
def crossValidate(attributes, outcomes, foldCount, learningRate, iterCountMax, threshold, ownFunction = True):
    thetaValList = []
    trainingErrorList =[]
    testingErrorList = []
    presList =[]; recallList = []
    accrList = []; fMeasList = []
    otcmVal = list(set(outcomes))
    featLen = attributes.shape[1]

    zMatrix = getDataMatrix(attributes,featLen)
    zMatrixFolds = getFolds(attributes,foldCount)
    otcmFolds = getFolds(outcomes,foldCount)

    testDataList = copy.copy(zMatrixFolds)
    testOtcmList = copy.copy(otcmFolds)

    for itr in range(foldCount):
        trainDataList = []
        trainOtcmList = []
        testingEstimate = []
        
        for intitr in range (foldCount):
            if intitr != itr:
                trainDataList.append(zMatrixFolds[intitr]) 
                trainOtcmList.append(otcmFolds[intitr])

        trainDataArr = 	np.array(trainDataList).reshape(-1,featLen)
        trainOtcmArr =  np.array(trainOtcmList).reshape(-1)
        testDataArr = np.array(testDataList[itr])
        testOtcmArr = np.array(testOtcmList[itr])
		
        if ownFunction:
            params = logisticRegressionKc(trainDataArr,trainOtcmArr, learningRate, iterCountMax, threshold)
            testingEstimate = getEstimate(testDataArr,params, otcmVal, testingEstimate)		
        else:			
            lR = LogisticRegression()
            lR.fit(trainDataArr,trainOtcmArr)
            testingEstimate = lR.predict(testDataArr)
            
        metric = getMetrics(testOtcmArr,testingEstimate,otcmVal)
        accrList.append(metric[0])
        presList.append(metric[1])
        recallList.append(metric[2])
        fMeasList.append(metric[3])
	
    return accrList, presList, recallList, fMeasList


# This function tests the above functions, when this file is called directly from terminal
def testScript():
	print "\n---> Started Logistic Regression - Iris dataset - Own function - k class...\n"
	attributes, outcomes = getDataFromFile("../Data/iriskc.data.shuffled")
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
	attributes, outcomes = min_max_scaler.fit_transform(np.array(attributes)), np.array(outcomes)
	#attributes, outcomes = np.array(attributes), np.array(outcomes)

	accrValues, presValues, recallValues, fMeasValues = crossValidate(attributes, outcomes, 10, learningRate=0.01, iterCountMax=750, threshold=0.005, ownFunction=True)
	for itr in range(10):
		print "Fold %d: \tAccuracy: %f\tPrecision: %f\tRecall: %f\tF-Measure: %f" %(itr+1,accrValues[itr],presValues[itr],recallValues[itr],fMeasValues[itr])
	print "\nMean values:\tAccuracy: %f\tPrecision: %f\tRecall: %f\tF-Measure: %f\n" % (np.mean(accrValues),np.mean(presValues),\
																												np.mean(recallValues),np.mean(fMeasValues))
															
																												
	print "---> Started Logistic Regression - Iris dataset - Inbuilt function - k class...\n"
	attributes, outcomes = getDataFromFile("../Data/iriskc.data.shuffled")
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
	attributes, outcomes = min_max_scaler.fit_transform(np.array(attributes)), np.array(outcomes)
	#attributes, outcomes = np.array(attributes), np.array(outcomes)

	accrValues, presValues, recallValues, fMeasValues = crossValidate(attributes, outcomes, 10, learningRate=0.01, iterCountMax=750, threshold=0.005, ownFunction=False)
	for itr in range(10):
		print "Fold %d: \tAccuracy: %f\tPrecision: %f\tRecall: %f\tF-Measure: %f" %(itr+1,accrValues[itr],presValues[itr],recallValues[itr],fMeasValues[itr])
	print "\nMean values:\tAccuracy: %f\tPrecision: %f\tRecall: %f\tF-Measure: %f\n" % (np.mean(accrValues),np.mean(presValues),\
																												np.mean(recallValues),np.mean(fMeasValues))


	print "---> Started Logistic Regression - Digits dataset - Own function - k class...\n"																									
	mnist = datasets.fetch_mldata('MNIST original')
	X, y = mnist.data / 255., mnist.target
	attributes = X[:20000]
	outcomes = y[:20000]
	#print list(set(outcomes))
	accrValues, presValues, recallValues, fMeasValues = crossValidate(attributes, outcomes, 10, learningRate=0.01, iterCountMax=100, threshold=0.005, ownFunction=False)
	for itr in range(10):
		print "Fold %d: \tAccuracy: %f\tPrecision: %f\tRecall: %f\tF-Measure: %f" %(itr+1,accrValues[itr],presValues[itr],recallValues[itr],fMeasValues[itr])
	print "\nMean values:\tAccuracy: %f\tPrecision: %f\tRecall: %f\tF-Measure: %f\n" % (np.mean(accrValues),np.mean(presValues),\
																												np.mean(recallValues),np.mean(fMeasValues))


	print "---> Started Logistic Regression - Digits dataset - Inbuilt function - k class...\n"																									
	mnist = datasets.fetch_mldata('MNIST original')
	X, y = mnist.data / 255., mnist.target
	attributes = X[:20000]
	outcomes = y[:20000]
	#print list(set(outcomes))
	accrValues, presValues, recallValues, fMeasValues = crossValidate(attributes, outcomes, 10, learningRate=0.01, iterCountMax=100, threshold=0.005, ownFunction=False)
	for itr in range(10):
		print "Fold %d: \tAccuracy: %f\tPrecision: %f\tRecall: %f\tF-Measure: %f" %(itr+1,accrValues[itr],presValues[itr],recallValues[itr],fMeasValues[itr])
	print "\nMean values:\tAccuracy: %f\tPrecision: %f\tRecall: %f\tF-Measure: %f\n" % (np.mean(accrValues),np.mean(presValues),\
																												np.mean(recallValues),np.mean(fMeasValues))
	
	
if __name__ == "__main__":
    testScript()

