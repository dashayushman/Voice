from datasource import OneHmmModel as hmmmod
from hmmlearn.hmm import GaussianHMM
import numpy as np
from utils import feature_extractor as fe
import pickle
import os
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#set these parameters for doing a grid search for svm
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

def generateModel(trainingData,labels,n_states = 3):

    models = {}

    for label in labels:
        if label in trainingData:
            trs = trainingData.get(label)

            hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000).fit(emg, emgl)



            objModel = hmmmod.OneHmmModel(hmm_model)
            models[label] = objModel
    return models
    #print(labels)

def dumpModels(filePath,model):
    try:
        with open(filePath, 'wb') as f:
            pickle.dump(model, f)
            return True
    except IOError as e:
        return False

def loadModels(filePath):
    if os.path.isfile(filePath):
        with open(filePath, 'rb') as f:
            models = pickle.load(f)
            return models
    else:
        return None