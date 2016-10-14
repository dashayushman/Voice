from sklearn.cross_validation import StratifiedKFold,LabelShuffleSplit
from utils import dataprep as dp
from utils import utility as util
import os
import time

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Training Data\individual"
algos = ['gauss']
n_folds = 5

def evaluate(n_folds,
             algos,
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
        Method to run an evaluation of Naive Bayes Classifier on the training and validation data given the parameters required
        :param n_folds:             (int)                       : Number of folds for k-fold cross-validation
        :param algos:               (list)                      : To select gaussian or multinomial
        :param rootDir:             (string)                    : The root directory of the Ground Truth
        :param reportDir:           (string)                    : directory path to save the classification report
        :param cacheDir:            (string)                    : Directory path to look for cached features
        :param modelGenerator:      (model_generator object)    : to generate an SVM models and evaluate them
        :param sampling_rate:       (int)                       : Sampling rate that we want to resample the data to
        :param prnt:                (boolean)                   : Flag to print the classification report in the console
        :param filewrt:             (boolean)                   : Flag to write the classification into a file
        :param cacherefresh:        (boolean)                   : Refresh the cache
        :param scaler:              (scalar object)             : To scale the sensor data
        :return:
    '''

    fileContent = []  # empty list for storing the classification report data which will be used later for writing to a file
    accuracies = []  # list of accuracies in every fold of the crossvalidation

    # extract training data from the root directory of the ground truth
    labels,\
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
        data = dp.extractFeatures(data, None, window=False, rms=False, f_mfcc=True, emg=False, imu=True)
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

    # K-Fold Cross validation
    for train, test in skf:
        print('running evaluation of fold ' + str(i))
        # split training instances into training and validation sets.
        # Do not get confused by the name test. Just trying to use scikit-learn nomenclature
        print('preparing training and validation data for evaluation')
        train_x,train_y,test_x,test_y = dp.prepareTrainingDataSvm(train,test,target,data)

        # generate the best model and classification report by doing a grid search over a list of parameters
        print('generating models and classification report')
        models = modelGenerator.generateModel(train_x,train_y,test_x,test_y,algos)

        for j,model in enumerate(models):
            clf = model['model']
            clf_rpt = model['clf_rpt']
            cm = model['cm']
            as_ = model['as']
            name = model['name']
            # add classification report to a list for structuring reports to write to a file
            fileContent = util.appendClfReportToListNB(fileContent,
                                                        clf_rpt,
                                                        cm,
                                                        as_,
                                                        algos[j],
                                                        i,
                                                        len(train),
                                                        len(test),
                                                        labels)
            accuracies.append(as_)
        i += 1
    fileContent = util.appendHeaderToFcListHMM(fileContent, accuracies, 'Naive Bayes')
    str_fc = util.getStrFrmList(fileContent, '')
    # convert markdown to html
    html = util.mrkdwn2html(str_fc)
    if prnt:
        print(str_fc)
    if filewrt:
        util.writeToFile(os.path.join(reportDir, 'NB_' + str(int(time.time())) + '.html'), html)