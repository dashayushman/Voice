from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle
import os
from sklearn.metrics import confusion_matrix

def generateModel(X_train,y_train,X_test,y_test,tuned_parameters):
    scores = ['precision']
    clf = None
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_weighted' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        clf_rpt = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        #print('Confusion matrix, without normalization')

        #print(classification_report(y_true, y_pred))
        #print()


    # Note the problem is too easy: the hyperparameter plateau is too flat and the
    # output model is the same for precision and recall with ties in quality.
    return clf,clf_rpt,cm
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