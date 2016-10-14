#from model_generator import generateModel
from sklearn.cross_validation import StratifiedKFold
from utils import dataprep as dp
from utils import utility as util
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import os
import time

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Training Data\individual"
n_states_l = [8]
n_folds = 10

def evaluateAccuracy(test,data,target,models,labelsdict,labels):
    y_true = []
    y_pred = []
    for idx in test:
        ti = data[idx]
        test_label = str(target[idx])
        y_true.append(target[idx])
        prob_vector = []
        con_data = ti.getConsolidatedFeatureMatrix()

        for modelLabel in labels:
            model = models.get(modelLabel)
            m_hmm = model.getModel()
            p_log = m_hmm.score(con_data)
            prob_vector.append(p_log)

        maxProbIndex = prob_vector.index(max(prob_vector))
        pred_label = int(labels[maxProbIndex])
        y_pred.append(pred_label)
    clf_rpt = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    acc_scr = accuracy_score(y_true, y_pred)
    return clf_rpt,cm,acc_scr

def evaluate(n_folds,
             n_states_l,
             rootDir,
             reportDir,
             cacheDir,
             modelGenerator,
             sampling_rate=50,
             prnt=True,
             filewrt=False,
             cacherefresh=False,
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
    # extract training data from the root directory of the ground truth
    labels, \
    data, \
    target, \
    labelsdict, \
    avg_len_emg, \
    avg_len_acc, \
    user_map, \
    user_list, \
    data_dict, \
    max_length_emg, \
    max_length_others, \
    data_path = dp.getTrainingData(rootDir)

    # scale the training data
    print('scaling data')
    if scaler is not None:
        data = dp.scaleData(data,
                            scaler)  # add a print statement that will indicate which instance is being scaled and how many left. do this for others as well

    # normalize the training instances to a common length to preserve the sampling to be used later for extracting features
    print('normalizing data')
    data = dp.normalizeTrainingData(data, max_length_emg, max_length_others)

    # resample all the training instances to normalize the data vectors
    # resample also calls consolidate data so there is no need to call consolidate raw data again
    print('resample data')
    data = dp.resampleTrainingData(data, sampling_rate, avg_len_acc, emg=False, imu=True)

    # extract features and consolidate features into one single matrix
    if cacherefresh:
        if os.path.isfile(os.path.join(cacheDir, 'featdata.pkl')):
            os.remove(os.path.join(cacheDir, 'featdata.pkl'))
        # if featdata.pkl does not exist then extract features and store it again for the future
        data = dp.extractFeatures(data, None, window=True, rms=False, f_mfcc=True, emg=False, imu=True)
        dp.dumpObject(os.path.join(cacheDir, 'featdata.pkl'), data)
    elif os.path.isfile(os.path.join(cacheDir, 'featdata.pkl')):
        # load the serielized featdata.pkl file to make things faster
        featData = dp.loadObject(os.path.join(cacheDir, 'featdata.pkl'))
        data = featData
    else:
        data = dp.extractFeatures(data, None, window=False, rms=False, f_mfcc=True, emg=False, imu=True)
        dp.dumpObject(os.path.join(cacheDir, 'featdata.pkl'), data)

    # Split the training instances into two sets
    # 1. Training Set
    # 2. Validation Set
    skf = StratifiedKFold(target, n_folds)
    i = 1
    for train, test in skf:
        print('\n\n\n')
        print('running evaluation of fold ' + str(i))
        print('\n')

        # preparing the training data and consolidating it to pass it to the HMM model generator
        print('preparing training data for generating the model')
        trainingData = dp.prepareTrainingDataHmmFeatures(train,target,data)

        # generate HMM models given the training data
        print('generating HMMs. This  might take a long time depending on the size of the training set and the number of states')
        models = modelGenerator.generateModel(trainingData,labels,n_states_l)
        l_states = True
        if type(n_states_l) is list:
            for j,model in enumerate(models):
                print('generating the classification report and adding that to a file')
                clf_rpt, cm, acc_scr = evaluateAccuracy(test, data, target, model, labelsdict, labels)
                fileContent = util.appendClfReportToListHMM(fileContent,clf_rpt, cm, acc_scr,n_states_l[j],i,len(train),len(test),labels)
                accuracies.append(acc_scr)
        else:
            clf_rpt, cm, acc_scr = evaluateAccuracy(test, data, target, models, labelsdict, labels)
            fileContent = util.appendClfReportToListHMM(fileContent, clf_rpt, cm, acc_scr, n_states_l[j], i, len(train),
                                                        len(test),labels)

            accuracies.append(acc_scr)
        i += 1
    fileContent = util.appendHeaderToFcListHMM(fileContent,accuracies,'Hidden Markov Model (With windowed features)')

    str_fc = util.getStrFrmList(fileContent,'')
    html = util.mrkdwn2html(str_fc)
    if prnt:
        print(str_fc)
    if filewrt:
        util.writeToFile(os.path.join(reportDir,'Hmm_Features_'+str(int(time.time())) +'.html'),html)







