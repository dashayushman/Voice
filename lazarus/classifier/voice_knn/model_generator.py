from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle
import os
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB,MultinomialNB


def generateModel(X_train,y_train,X_test,y_test,algos=['gauss']):
    clfs = []
    for algo in algos:
        if algo == 'gauss':
            nb_clf_gaus = GaussianNB()
            nb_clf_gaus.fit(X_train, y_train)
            y_true, y_pred = y_test, nb_clf_gaus.predict(X_test)
            clf_rpt = classification_report(y_true, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            clf = {}
            clf['model'] = nb_clf_gaus
            clf['clf_rpt'] = clf_rpt
            clf['cm']=cm
            clf['name'] = 'Gaussian Naive Bayes'
            clfs.append(clf)
        if algo == 'multi':
            nb_clf_multi = MultinomialNB()
            nb_clf_multi.fit(X_train, y_train)
            y_true, y_pred = y_test, nb_clf_multi.predict(X_test)
            clf_rpt = classification_report(y_true, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            clf = {}
            clf['model'] = nb_clf_multi
            clf['clf_rpt'] = clf_rpt
            clf['cm'] = cm
            clf['name'] = 'Multinomial Naive Bayes'
            clfs.append(clf)

    # Note the problem is too easy: the hyperparameter plateau is too flat and the
    # output model is the same for precision and recall with ties in quality.
    return clfs
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