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
