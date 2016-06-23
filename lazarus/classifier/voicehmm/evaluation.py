from model_generator import generateModels
from utils import dataprep
from sklearn.cross_validation import StratifiedKFold
from utils import dataprep as dp
import numpy as np
from utils import feature_extractor as fe

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Algos\HMM\Training Data\New"

def evaluateAccuracy(test,data,target,models,modelLabels):
    hit = 0
    for idx in test:
        ti = data[idx]
        test_label = target[idx]
        prob_vector = []
        emg_t, acc_t, gyr_t, ori_t = ti.getRawData()

        for modelLabel in modelLabels:
            model = models.get(modelLabel)
            m_emg,m_acc,m_gyr,m_ori = model.getModels()
            p_emg,_ = m_emg.decode(emg_t)
            p_acc,_ = m_acc.decode(acc_t)
            p_gyr,_ = m_gyr.decode(gyr_t)
            p_ori,_ = m_ori.decode(ori_t)
            aggr_prob = np.sum(fe.max_abs_scaler.fit_transform(np.array([p_emg,p_acc,p_gyr,p_ori])))
            prob_vector.append(aggr_prob)

        maxProbIndex = prob_vector.index(max(prob_vector))
        if test_label == modelLabels[maxProbIndex]:
            hit += 1
    accuracy = hit/len(test)
    return accuracy

if __name__ == "__main__":
    labels, data, target = dataprep.getTrainingData(rootDir)
    skf = StratifiedKFold(target, 5)
    accuracies = []
    for train, test in skf:
        trainingData = dp.prepareTrainingData(train,target,data)
        models,modelLabels = generateModels(trainingData,labels)
        acc = evaluateAccuracy(test,data,target,models,modelLabels)
        accuracies.append(acc)
        #print("%s %s" % (train, test))
        #modelsDict = generateModels()
    print(accuracies)
    print('Total Accuracy in Percentage is:',(np.sum(accuracies)/len(accuracies))*100)





