# Importing the necessary packages
import numpy as np, pylab as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, f1_score, precision_recall_curve


# This function gets mean of a specific column for the records of given class
def getSpecificMean(attributes, outcomes, dsrdOutcome, colNum=0):
	dsrdOList = ([attributes[i] for i in range(len(outcomes)) if outcomes[i]==dsrdOutcome])
	return np.mean(np.array(dsrdOList)[:,colNum])


# This function gets variance of a specific column for the records of given class
def getSpecificVar(attributes, outcomes, dsrdOutcome, colNum=0):
	dsrdOList = ([attributes[i] for i in range(len(outcomes)) if outcomes[i]==dsrdOutcome])
	return np.std(np.array(dsrdOList)[:,colNum])


# This function gets folds of given count from the input list of data
def getFolds(iptData, foldCount):
    iptDataSize = len(iptData)
    elemCount = iptDataSize % foldCount
    otptData = []
    iterator = iter(iptData)
    for i in range(foldCount):
        otptData.append([])
        for j in range(iptDataSize / foldCount):
            otptData[i].append(iterator.next())
        if elemCount:
            otptData[i].append(iterator.next())
            elemCount -= 1
    return otptData


# This function is to read the input file and get list of attributes and outcomes from it
def readData(dataFiles, split=",", nB=False):
	attributes = []; outcomes = []
	fileContent = open(dataFiles,'r')
	for line in fileContent:
		li=line.strip()
		# Ignore blank lines and comments, then split data
		if li != "":
			if not li.startswith("#"):
				spltData = li.split(split)
				lastElem = len(spltData)-1
				if lastElem == 1:
					if nB:
						attributes.append(spltData[1].lower())
						outcomes.append(spltData[0])
					else:
						attributes.append(float(spltData[0]))
						outcomes.append(spltData[1])
				else:
					attributes.append([float(spltData[i]) for i in range(lastElem)])
					outcomes.append(spltData[lastElem])
	return attributes, outcomes	


def getMetrics(testOtcmArr,testingEstimate,otcmVal,showPlot=False,title=None):
	tPos = 0; tNeg = 0; fPos = 0; fNeg = 0
	finOtcmEstmate=[]; outcomes=[]

	# Calculate confusion matrix elements, then calculate metrics
	for fitr in range(len(testOtcmArr)):			
		tEst = testingEstimate[fitr]
		if  tEst == testOtcmArr[fitr]:
			if tEst == otcmVal[0]:
				tPos += 1 
			else:
				tNeg += 1
		else:
			if tEst == otcmVal[0]:
				fPos += 1 
			else:
				fNeg += 1
	accrVal = float(tPos+tNeg)/(tPos+tNeg+fNeg+fPos)
	presVal = float(tPos)/(tPos+fPos) if tPos+fPos != 0 else 0.0
	recallVal = float(tPos)/(tPos+fNeg) if tPos+fNeg != 0 else 0.0
	fMeasVal = 2*float(presVal*recallVal)/(presVal+recallVal) if presVal+recallVal != 0 else 0.0
	
	#accrVal = accuracy_score(testOtcmArr,testingEstimate)
   	#presVal = precision_score(testOtcmArr,testingEstimate)
    #recallVal = recall_score(testOtcmArr,testingEstimate)
    #fMeasVal= f1_score(testOtcmArr,testingEstimate)
	
    for i in range(0,len(testOtcmArr)):
        outcomes+=[1 if testOtcmArr[i] == otcmVal[0] else 0]
        finOtcmEstmate+=[1 if testingEstimate[i] == otcmVal[0] else 0]
	precision, recall, threshold = precision_recall_curve(outcomes,finOtcmEstmate)
	areaUnderPrc = auc(precision, recall)

	# If showPlot flag is set, display the precision-recall curve
    if(showPlot):
		pl.clf()
		pl.plot(recall, precision, label='P-R curve for Fold 1')
		pl.plot([0, 1], [0, 1], 'k--')

		pl.xlim([0.0, 1.0]); pl.ylim([0.0, 1.0])
		pl.xlabel('Recall'); pl.ylabel('Precision')

		pl.title('%s - P-R curve for Fold 1' % title); pl.legend(loc="lower right")
		pl.show()

	return accrVal, presVal, recallVal, fMeasVal, areaUnderPrc

