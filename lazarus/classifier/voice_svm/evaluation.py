from sklearn.cross_validation import StratifiedKFold,LabelShuffleSplit
from utils import dataprep as dp
from sklearn import preprocessing
from utils import utility as util
import os
import time

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Training Data\individual"
n_folds = 5

scaler = preprocessing.StandardScaler()

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3,1e-4,1e-5,1e-6],
                     'C': [1, 10, 100, 1000,10000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000,10000]}]

def evaluate(n_folds,tp,rootDir,reportDir,cacheDir,modelGenerator,prnt=True,filewrt=False,cacherefresh=False):
    fileContent = []
    accuracies = []

    labels, data, target,labelsdict,avg_len,user_map,user_list,data_dict = dp.getTrainingData(rootDir)

    #resample also calls consolidate data so there is no need to call consolidate raw data again
    data = dp.resampleTrainingData(data,avg_len)

    #extract features and consolidate features into one single matrix
    if cacherefresh:
        os.remove(os.path.join(cacheDir, 'featdata.pkl'))
    # extract features and consolidate features into one single matrix
    featData = dp.loadObject(os.path.join(cacheDir, 'featdata.pkl'))
    if featData is None:
        data = dp.extractFeatures(data, None, window=True, rms=False, f_mfcc=True)
        dp.dumpObject(os.path.join(cacheDir, 'featdata.pkl'), data)
    else:
        data = featData

    skf = StratifiedKFold(target, n_folds)
    #skf = LabelShuffleSplit(target, n_iter=10, test_size=0.3,random_state=0)
    table = []
    i = 1
    for train, test in skf:
        train_x,train_y,test_x,test_y = dp.prepareTrainingDataSvm(train,test,target,data)
        model,clf_rpt,cm,acc_scr,best_params = modelGenerator.generateModel(train_x,train_y,test_x,test_y,tp)
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
    html = util.mrkdwn2html(str_fc)
    if prnt:
        print(str_fc)
    if filewrt:
        util.writeToFile(os.path.join(reportDir, 'SVM_' + str(int(time.time())) + '.html'), html)






