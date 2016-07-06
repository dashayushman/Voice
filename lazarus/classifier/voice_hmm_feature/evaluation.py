from model_generator import generateModel
from sklearn.cross_validation import StratifiedKFold,LabelShuffleSplit
from utils import dataprep as dp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Algos\HMM\Training Data\New"
n_states_l = [3,4,5,6,7,8,9,10]
n_folds = 5

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
        con_data = ti.getConsolidatedFeatureMatrix()

        for modelLabel in labels:
            model = models.get(modelLabel)
            m_hmm = model.getModel()
            p_log,_ = m_hmm.decode(con_data)
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

    #resample also calls consolidate data so there is no need to call consolidate raw data again
    data = dp.resampleTrainingData(data,avg_len)

    #extract features and consolidate features into one single matrix
    data = dp.extractFeatures(data)

    skf = StratifiedKFold(target, n_folds)
    #skf = LabelShuffleSplit(target, n_iter=10, test_size=0.3,random_state=0)
    table = []
    i = 1
    for train, test in skf:
        trainingData = dp.prepareTrainingDataHmmFeatures(train,target,data)
        models = generateModel(trainingData,labels,n_states_l)
        l_states = True
        if type(n_states_l) is list:
            accuracies = []
            for model in models:
                acc = evaluateAccuracy(test, data, target, model, labelsdict, labels)
                accuracies.append(acc)
            accuracies.insert(0,i)
            table.append(accuracies)
            i += 1
        else:
            acc = evaluateAccuracy(test, data, target, models, labelsdict, labels)
            table.append([i,acc])
        #print ("%s %s" % (train, test))
        #modelsDict = generateModels()
    print('Classification Results')
    print('Number of Folds : ',n_folds)
    table = np.array(table)
    if type(n_states_l) is list:
        skip = True
        avg_accs = ['average']
        for ind in np.arange(table.shape[1]):
            if skip:
                skip = False
                continue
            else:
                state_accs = table[:,ind]
                avg = np.average(state_accs)
                avg_accs.append(avg)
        table = np.append(table,[avg_accs],axis=0)
        headers = ['fold index'] + ['n_states=' + str(s) for s in n_states_l]
        print(tabulate(table, headers, tablefmt="simple"))
    else:
        print('Number of states : ', n_folds)
        avg = np.average(table[:,1])
        table = np.append(table,[['average',avg]])
        headers = ['fold index','accuracy']
        print(tabulate(table, headers, tablefmt="fancy_grid"))






