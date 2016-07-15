from sklearn.metrics import classification_report
import pickle
import os
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def generateModel(X_train,y_train,X_test,y_test,n_neighbours=[3]):
    clfs = []
    for n_neighbour in n_neighbours:
        neigh = KNeighborsClassifier(n_neighbors=n_neighbour)
        neigh.fit(X_train, y_train)
        y_true, y_pred = y_test, neigh.predict(X_test)
        clf_rpt = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        acc_scr = accuracy_score(y_true, y_pred)
        clf = {}
        clf['model'] = neigh
        clf['clf_rpt'] = clf_rpt
        clf['cm']=cm
        clf['as'] = acc_scr
        clf['n_neighbours'] = n_neighbour
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