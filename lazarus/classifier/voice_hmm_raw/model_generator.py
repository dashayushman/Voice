from datasource import OneHmmModel as hmmmod
from hmmlearn.hmm import GaussianHMM
import pickle
import os

def generateModel(trainingData,labels,n_states = 3):

    if(type(n_states) is list):
        model_list = []
        for s in n_states:
            models = {}
            for i,label in enumerate(labels):
                print(str((i/len(labels))*100) + '% comleted')
                if int(label) in trainingData:
                    trs = trainingData.get(int(label))
                    data = trs['data']
                    l_data = trs['datal']
                    hmm_model = GaussianHMM(n_components=s, covariance_type="full", n_iter=1000).fit(data,l_data)
                    objModel = hmmmod.OneHmmModel(hmm_model)
                    models[label] = objModel
                #print('generated model for label ', label)
            print('finished!!! generating HMM for '+ str(s)+ ' states')
            model_list.append(models)
        return model_list
    models = {}
    for label in labels:
        if label in trainingData:
            trs = trainingData.get(label)
            data = trs['data']
            l_data = trs['datal']
            hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000).fit(data, l_data)
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