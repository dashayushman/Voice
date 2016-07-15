from sklearn.cross_validation import StratifiedKFold,LabelShuffleSplit
from utils import dataprep as dp
from utils import utility as util
import os
import time

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Training Data\individual"
algos = ['gauss']
n_folds = 5

def evaluate(n_folds,algos,rootDir,reportDir,cacheDir,modelGenerator,prnt=True,filewrt=False,cacherefresh=False):
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
        models = modelGenerator.generateModel(train_x,train_y,test_x,test_y,algos)
        for j,model in enumerate(models):
            clf = model['model']
            clf_rpt = model['clf_rpt']
            cm = model['cm']
            as_ = model['as']
            name = model['name']
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
    html = util.mrkdwn2html(str_fc)
    if prnt:
        print(str_fc)
    if filewrt:
        util.writeToFile(os.path.join(reportDir, 'NB_' + str(int(time.time())) + '.html'), html)