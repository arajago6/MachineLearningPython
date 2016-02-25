import numpy as np
import matplotlib.pyplot as plt

def drawPlot(inptData,plotTitle):
	plt.scatter(inptData[0], inptData[1], color='black')
	plt.title(plotTitle)
	if len(inptData)>2:
		plt.scatter(inptData[2], inptData[3], color='r')
	plt.show()


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


def estimate(thetaVal,dataArr):
	return np.dot(dataArr,thetaVal)


def meanSqrError(estimated,actual):
	#estmAvg = np.mean(actual); denom = (sum(actual)-estmAvg)**2
	#return sum((estimated-actual)**2)/denom 
	return sum([(yh-y)**2 for y, yh in zip(estimated,actual)])/len(estimated) 
