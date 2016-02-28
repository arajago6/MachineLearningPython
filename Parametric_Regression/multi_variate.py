import numpy as np
import copy
from ml_lib import *
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist

# Map data to higher dimensional space
def mapHD(inptData,degree):
    return PolynomialFeatures(degree).fit_transform(inptData)
