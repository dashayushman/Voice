from sklearn.cross_validation import StratifiedKFold
from lazarus.utils import dataprep as dp
from lazarus.utils import utility as util
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import os
import time
from lazarus.datasource import vds
from lazarus.datasource.data import DataSet

def evaluateAccuracy(test,data,target,models,labelsdict,labels):
    y_true = []
    y_pred = []
    for idx in test:
        ti = data[idx]
        #test_label = str(target[idx])
        y_true.append(target[idx])
        prob_vector = []
        con_data = ti.getConsolidatedDataMatrix()

        for modelLabel in labels:
            model = models.get(modelLabel)
            m_hmm = model.getModel()
            p_log,_ = m_hmm.decode(con_data)
            prob_vector.append(p_log)

        maxProbIndex = prob_vector.index(max(prob_vector))
        pred_label = int(labels[maxProbIndex])
        y_pred.append(pred_label)
    clf_rpt = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    acc_scr = accuracy_score(y_true, y_pred)

    return clf_rpt,cm,acc_scr

def evaluate(n_folds,
             rootDir,
             reportDir,
             modelGenerator,
             sampling_rate=50,
             prnt=True,
             filewrt=False,
             scaler=None):
    '''
        Method to run an evaluation of Hidden Markov Model with raw signal data from training and validation set given the parameters required
        :param n_folds:             (int)                       : Number of folds for k-fold cross-validation
        :param n_states_l:          (list)                      : List of number of states to train the HMM with and evaluate the system on
        :param rootDir:             (string)                    : The root directory of the Ground Truth
        :param reportDir:           (string)                    : directory path to save the classification report
        :param modelGenerator:      (model_generator object)    : to generate an SVM models and evaluate them
        :param sampling_rate:       (int)                       : Sampling rate that we want to resample the data to
        :param prnt:                (boolean)                   : Flag to print the classification report in the console
        :param filewrt:             (boolean)                   : Flag to write the classification into a file
        :param cacherefresh:        (boolean)                   : Refresh the cache
        :param scaler:              (scalar object)             : To scale the sensor data
        :return:
    '''
    fileContent = []
    accuracies = []


    kfolds = dp.read_data_sets(rootDir,
                                scaler,
                                n_folds)
    for i,(train, test) in enumerate(kfolds):
         print('\n\n\n')
         print('running evaluation of fold ' + str(i))
         modelGenerator.generateModel(train, test)
         #print('Error rate: ' + error)
    #modelGenerator.generateModel(kfolds, gtarget)