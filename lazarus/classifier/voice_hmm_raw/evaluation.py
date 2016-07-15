from sklearn.cross_validation import StratifiedKFold
from utils import dataprep as dp
from utils import utility as util
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import os
import time

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Training Data\individual"
n_states_l = [8]
n_folds = 5

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

def evaluate(n_folds,n_states_l,rootDir,reportDir,modelGenerator,prnt=True,filewrt=False):
    fileContent = []
    accuracies = []
    labels, data, target,labelsdict,avg_len,user_map,user_list,data_dict = dp.getTrainingData(rootDir)
    data = dp.resampleTrainingData(data,avg_len)
    skf = StratifiedKFold(target, n_folds)
    i = 1
    for train, test in skf:
        trainingData = dp.prepareTrainingDataHmmRaw(train,target,data)
        models = modelGenerator.generateModel(trainingData,labels,n_states_l)
        l_states = True
        if type(n_states_l) is list:
            for j,model in enumerate(models):
                clf_rpt, cm, acc_scr = evaluateAccuracy(test, data, target, model, labelsdict, labels)
                fileContent = util.appendClfReportToListHMM(fileContent, clf_rpt, cm, acc_scr, n_states_l[j], i,
                                                            len(train), len(test),labels)
                accuracies.append(acc_scr)
        else:
            clf_rpt, cm, acc_scr = evaluateAccuracy(test, data, target, models, labelsdict, labels)
            fileContent = util.appendClfReportToListHMM(fileContent, clf_rpt, cm, acc_scr, n_states_l[j], i, len(train),
                                                        len(test),labels)

            accuracies.append(acc_scr)
        i += 1
    fileContent = util.appendHeaderToFcListHMM(fileContent, accuracies,'Hidden Markov Model (With Raw Data)')

    str_fc = util.getStrFrmList(fileContent, '')
    html = util.mrkdwn2html(str_fc)
    if prnt:
        print(str_fc)
    if filewrt:
        util.writeToFile(os.path.join(reportDir, 'Hmm_Raw_' + str(int(time.time())) + '.html'), html)







