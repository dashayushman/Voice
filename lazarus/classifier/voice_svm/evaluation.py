from sklearn.cross_validation import StratifiedKFold,LabelShuffleSplit
from utils import dataprep as dp
from sklearn import preprocessing
from utils import utility as util
import os
import time

def evaluate(n_folds,tp,rootDir,reportDir,cacheDir,modelGenerator,sampling_rate=50,prnt=True,filewrt=False,cacherefresh=False,scaler = None):
    fileContent = []    # empty list for storing the classification report data which will be used later for writing to a file
    accuracies = []     # list of accuracies in every fold of the crossvalidation

    # extract training data from the root directory of the ground truth
    labels, data, target,labelsdict,avg_len_emg,avg_len_acc,user_map,user_list,data_dict,max_length_emg,max_length_others,data_path = dp.getTrainingData(rootDir)

    #scale the training data
    print('scaling data')
    if scaler is not None:
        data = dp.scaleData(data,scaler) #add a print statement that will indicate which instance is being scaled and how many left. do this for others as well

    #normalize the training instances to a common length to preserve the sampling to be used later for extracting features
    print('normalizing data')
    data = dp.normalizeTrainingData(data,max_length_emg,max_length_others)

    # resample all the training instances to normalize the data vectors
    # resample also calls consolidate data so there is no need to call consolidate raw data again
    print('resample data')
    data = dp.resampleTrainingData(data,sampling_rate,avg_len_acc,emg=False,imu=True)


    #extract features and consolidate features into one single matrix
    if cacherefresh:
        if os.path.isfile(os.path.join(cacheDir, 'featdata.pkl')):
            os.remove(os.path.join(cacheDir, 'featdata.pkl'))
        #if featdata.pkl does not exist then extract features and store it again for the future
        data = dp.extractFeatures(data, None, window=False, rms=False, f_mfcc=True,emg=False,imu=True)
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
    table = []
    i = 1

    # K-Fold Cross validation
    for train, test in skf:
        #split training instances into training and validation sets. Do not get confused by the name test. Just trying to use scikit-learn nomenclature
        train_x,train_y,test_x,test_y = dp.prepareTrainingDataSvm(train,test,target,data)

        #generate the best model and classification report by doing a grid search over a list of parameters
        model,clf_rpt,cm,acc_scr,best_params = modelGenerator.generateModel(train_x,train_y,test_x,test_y,tp)

        #add classification report to a list for structuring reports to write to a file
        fileContent = util.appendClfReportToListSvm(fileContent,
                                                   clf_rpt,
                                                   cm,
                                                   acc_scr,
                                                   best_params,
                                                   i,
                                                   len(train),
                                                   len(test),
                                                   labels)
        accuracies.append(acc_scr)
        i += 1
    fileContent = util.appendHeaderToFcListHMM(fileContent, accuracies, 'SVM')

    str_fc = util.getStrFrmList(fileContent, '')

    #convert markdown to html
    html = util.mrkdwn2html(str_fc)
    if prnt:
        print(str_fc)
    if filewrt:
        util.writeToFile(os.path.join(reportDir, 'SVM_' + str(int(time.time())) + '.html'), html)






