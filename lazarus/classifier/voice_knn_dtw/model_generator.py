
from sklearn.metrics import classification_report
from knn_dtw_class import KnnDtw
import pickle
import os
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB,MultinomialNB


def generateModel(X_train,y_train,X_test,y_test):
    m = KnnDtw(n_neighbors=1, max_warping_window=10)
    m.fit(X_train, y_train)
    y_true, y_pred,y_probab = y_test, m.predict(X_test)
    clf_rpt = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    clf = {}
    clf['model'] = m
    clf['clf_rpt'] = clf_rpt
    clf['cm']=cm
    clf['name'] = 'KnnDtw'

    # Note the problem is too easy: the hyperparameter plateau is too flat and the
    # output model is the same for precision and recall with ties in quality.
    return clf
    #print(labels)

def dumpModels(filePath,model):
    try:
        with open(filePath, 'wb') as f:
            pickle.dump(model, f)
            return True
    except IOError as e:
        return False

def loadModels(filePath):
    if os.path.isfile(filePath):
        with open(filePath, 'rb') as f:
            models = pickle.load(f)
            return models
    else:
        return None