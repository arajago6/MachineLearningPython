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
