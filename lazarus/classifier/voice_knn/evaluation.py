from model_generator import generateModel
from sklearn.cross_validation import StratifiedKFold,LabelShuffleSplit
from utils import dataprep as dp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Algos\HMM\Training Data\New"
algos = ['gauss','multi']
n_folds = 5

if __name__ == "__main__":
    labels, data, target,labelsdict,avg_len = dp.getTrainingData(rootDir)

    #resample also calls consolidate data so there is no need to call consolidate raw data again
    data = dp.resampleTrainingData(data,avg_len)

    #extract features and consolidate features into one single matrix
    data = dp.extractFeatures(data,False)

    skf = StratifiedKFold(target, n_folds)
    #skf = LabelShuffleSplit(target, n_iter=10, test_size=0.3,random_state=0)
    table = []
    i = 1
    for train, test in skf:
        train_x,train_y,test_x,test_y = dp.prepareTrainingDataSvm(train,test,target,data)
        models = generateModel(train_x,train_y,test_x,test_y)
        for model in models:
            clf = model['model']
            clf_rpt = model['clf_rpt']
            cm = model['cm']
            name = model['name']
            print('for ',name,'classifier')
            print('For fold : ',i)
            print('Number of training instances = ',train_y.shape[0])
            print('Number of testing instances = ', test_y.shape[0])
            print('Classification Report')
            print(clf_rpt)
            print('Confusion Matrix')
            print(cm)
        i += 1






