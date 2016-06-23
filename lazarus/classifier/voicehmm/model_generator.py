from datasource import HmmModel as hmmmod
from hmmlearn.hmm import GaussianHMM

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

    return models,modelLabels
    #print(labels)

