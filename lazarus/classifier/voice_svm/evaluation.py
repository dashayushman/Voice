from model_generator import generateModel
from sklearn.cross_validation import StratifiedKFold,LabelShuffleSplit
from utils import dataprep as dp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Training Data\New"
n_states_l = [3,4,5,6,7,8,9,10]
n_folds = 10

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,1e-5,1e-6],
                     'C': [1, 10, 100, 1000,10000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000,10000]}]

if __name__ == "__main__":
    labels, data, target,labelsdict,avg_len,user_map,user_list,data_dict = dp.getTrainingData(rootDir)

    #resample also calls consolidate data so there is no need to call consolidate raw data again
    data = dp.resampleTrainingData(data,avg_len)

    #extract features and consolidate features into one single matrix
    featData = dp.loadObject('featdata.pkl')
    if featData is None:
        data = dp.extractFeatures(data, False)
        dp.dumpObject('featdata.pkl', data)
    else:
        data = featData

    skf = StratifiedKFold(target, n_folds)
    #skf = LabelShuffleSplit(target, n_iter=10, test_size=0.3,random_state=0)
    table = []
    i = 1
    for train, test in skf:
        train_x,train_y,test_x,test_y = dp.prepareTrainingDataSvm(train,test,target,data)
        model,clf_rpt,cm = generateModel(train_x,train_y,test_x,test_y,tuned_parameters)
        print('For fold : ',i)
        print('Number of training instances = ',train_y.shape[0])
        print('Number of testing instances = ', test_y.shape[0])
        print('Classification Report')
        print(clf_rpt)

        print('Confusion Matrix')
        print(cm)
        i += 1






