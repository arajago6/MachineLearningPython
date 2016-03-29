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
