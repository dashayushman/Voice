from model_generator import generateLabeledModel,generateModel
from sklearn.cross_validation import StratifiedKFold,LabelShuffleSplit
from utils import dataprep as dp
import numpy as np
from utils import feature_extractor as fe
import matplotlib.pyplot as plt

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Algos\HMM\Training Data\New"

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

def evaluateAccuracy(test,data,target,models,labelsdict,labels):
    hit = 0
    conmat = np.zeros((10,10))
    for idx in test:
        ti = data[idx]
        test_label = target[idx]
        prob_vector = []
        emg_t, acc_t, gyr_t, ori_t = ti.getRawData()

        for modelLabel in labels:
            model = models.get(modelLabel)
            m_hmm = model.getModel()
            p_log,_ = m_hmm.decode(emg_t)
            prob_vector.append(p_log)

        maxProbIndex = prob_vector.index(max(prob_vector))
        if test_label == labels[maxProbIndex]:
            hit += 1
        if test_label in labelsdict and labels[maxProbIndex] in labelsdict:
            x = labelsdict[test_label]
            y = labelsdict[labels[maxProbIndex]]
            conmat[x,y] = conmat[x,y] + 1
    accuracy = hit/len(test)
    #plt.figure()
    #plot_confusion_matrix(conmat,labels)
    #plt.show()
    return accuracy

if __name__ == "__main__":
    labels, data, target,labelsdict,avg_len = dp.getTrainingData(rootDir)
    data = dp.resampleTrainingData(data,avg_len)
    skf = StratifiedKFold(target, 5)
    #skf = LabelShuffleSplit(target, n_iter=10, test_size=0.3,random_state=0)
    accuracies = []
    for train, test in skf:
        trainingData = dp.prepareTrainingDataHmmRaw(train,target,data)
        model = generateModel(trainingData,labels,3)

        acc = evaluateAccuracy(test,data,target,labels,labelsdict,labels)
        accuracies.append(acc)
        #print ("%s %s" % (train, test))
        #modelsDict = generateModels()

    print(accuracies)
    print('Total Accuracy in Percentage is:',(np.sum(accuracies)/len(accuracies))*100)





