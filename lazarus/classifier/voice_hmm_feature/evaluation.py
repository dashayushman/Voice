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

def evaluate(n_folds,n_states_l,rootDir,reportDir,cacheDir,modelGenerator,prnt=True,filewrt=False,cacherefresh=False):

    fileContent = []
    accuracies = []

    labels, data, target,labelsdict,avg_len,user_map,user_list,data_dict = dp.getTrainingData(rootDir)

    #resample also calls consolidate data so there is no need to call consolidate raw data again
    data = dp.resampleTrainingData(data,avg_len)

    if cacherefresh:
        os.remove(os.path.join(cacheDir,'featdata.pkl'))
    # extract features and consolidate features into one single matrix
    featData = dp.loadObject(os.path.join(cacheDir,'featdata.pkl'))
    if featData is None:
        data = dp.extractFeatures(data,None,window=True,rms=False,f_mfcc=True)
        dp.dumpObject(os.path.join(cacheDir,'featdata.pkl'),data)
    else:
        data = featData

    skf = StratifiedKFold(target, n_folds)
    i = 1
    for train, test in skf:
        trainingData = dp.prepareTrainingDataHmmFeatures(train,target,data)
        models = modelGenerator.generateModel(trainingData,labels,n_states_l)
        l_states = True
        if type(n_states_l) is list:
            for j,model in enumerate(models):
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







