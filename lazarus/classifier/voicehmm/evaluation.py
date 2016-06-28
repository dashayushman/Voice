from model_generator import generateModels,loadModels,dumpModels
from utils import dataprep
from sklearn.cross_validation import StratifiedKFold,LabelShuffleSplit
from utils import dataprep as dp
import numpy as np
from utils import feature_extractor as fe
import matplotlib.pyplot as plt

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Algos\HMM\Training Data\New"
modelLabels=['0','1','2','3','4','5','6','7','8','9']

def plot_confusion_matrix(cm,labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluateAccuracy(test,data,target,models,modelLabels,labelsdict,labels):
    hit = 0
    conmat = np.zeros((10,10))
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
        if test_label in labelsdict and modelLabels[maxProbIndex] in labelsdict:
            x = labelsdict[test_label]
            y = labelsdict[modelLabels[maxProbIndex]]
            conmat[x,y] = conmat[x,y] + 1
    accuracy = hit/len(test)
    #print(conmat)
    plt.figure()
    plot_confusion_matrix(conmat,labels)
    plt.show()
    return accuracy

if __name__ == "__main__":
    labels, data, target,labelsdict = dataprep.getTrainingData(rootDir)
    #skf = StratifiedKFold(target, 5)
    skf = LabelShuffleSplit(target, n_iter=10, test_size=0.3,random_state=0)
    accuracies = []
    for train, test in skf:
        trainingData = dp.prepareTrainingData(train,target,data)
        models = loadModels('models.pkl')
        if(models == None):
            models,modelLabels = generateModels(trainingData,labels)
            dumpModels('models.pkl',models)
        acc = evaluateAccuracy(test,data,target,models,modelLabels,labelsdict,labels)
        accuracies.append(acc)
        #print ("%s %s" % (train, test))
        #modelsDict = generateModels()

    print(accuracies)
    print('Total Accuracy in Percentage is:',(np.sum(accuracies)/len(accuracies))*100)





