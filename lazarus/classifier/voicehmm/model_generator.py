from datasource import HmmModel as hmmmod
from hmmlearn.hmm import GaussianHMM
import numpy as np
from utils import feature_extractor as fe
import pickle
import os

def generateModels(trainingData,labels):
    models = {}
    modelLabels = []
    for label in labels:
        if label in trainingData:
            trs = trainingData.get(label)

            #emg model
            emg = trs['emg']
            emgl = trs['emgl']
            hmm_emg = GaussianHMM(n_components=10, covariance_type="diag", n_iter=1000).fit(emg,emgl)

            # acc model
            acc = trs['acc']
            accl = trs['accl']
            hmm_acc = GaussianHMM(n_components=10, covariance_type="diag", n_iter=1000).fit(acc, accl)

            # gyr model
            gyr = trs['gyr']
            gyrl = trs['gyrl']
            hmm_gyr = GaussianHMM(n_components=10, covariance_type="diag", n_iter=1000).fit(gyr, gyrl)

            # ori model
            ori = trs['ori']
            oril = trs['oril']
            hmm_ori = GaussianHMM(n_components=10, covariance_type="diag", n_iter=1000).fit(ori, oril)

            objModel = hmmmod.HmmModel(label,hmm_emg,hmm_acc,hmm_gyr,hmm_ori)
            models[label] = objModel
            modelLabels.append(label)

    return models,modelLabels
    #print(labels)

def getLabel(data,models,modelLabels):
    prob_vector = []
    emg_t, acc_t, gyr_t, ori_t = data.getRawData()
    for modelLabel in modelLabels:
        model = models.get(modelLabel)
        m_emg, m_acc, m_gyr, m_ori = model.getModels()
        p_emg, _ = m_emg.decode(emg_t)
        p_acc, _ = m_acc.decode(acc_t)
        p_gyr, _ = m_gyr.decode(gyr_t)
        p_ori, _ = m_ori.decode(ori_t)
        aggr_prob = np.sum(fe.max_abs_scaler.fit_transform(np.array([p_emg, p_acc, p_gyr, p_ori])))
        prob_vector.append(aggr_prob)

    maxProbIndex = prob_vector.index(max(prob_vector))
    return modelLabels[maxProbIndex]

def dumpModels(filePath,models):
    try:
        with open(filePath, 'wb') as f:
            pickle.dump(models, f)
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