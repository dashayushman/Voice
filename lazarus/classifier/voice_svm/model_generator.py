from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def generateModel(X_train,y_train,X_test,y_test,tuned_parameters):
    scores = ['precision']
    clf = None
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        #print()
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_weighted' % score)
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        clf_rpt = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        acc_scr = accuracy_score(y_true, y_pred)


    return clf,clf_rpt,cm,acc_scr,clf.best_params_

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