import numpy as np
import matplotlib.pyplot as plt

def drawPlot(inptData,plotTitle):
	plt.scatter(inptData[0], inptData[1], color='black')
	plt.title(plotTitle)
	if len(inptData)>2:
		plt.scatter(inptData[2], inptData[3], color='r')
	plt.show()
