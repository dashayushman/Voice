from classifier.voice_hmm_feature import evaluation as hmmfeat
from classifier.voice_hmm_feature import model_generator as m_hmmfeat
from classifier.voice_hmm_raw import evaluation as hmmraw
from classifier.voice_hmm_raw import model_generator as m_hmmraw
from classifier.voice_knn import evaluation as knn
from classifier.voice_knn import model_generator as m_knn
from classifier.voice_naive_bayes import evaluation as nb
from classifier.voice_naive_bayes import model_generator as m_nb
from classifier.voice_svm import evaluation as svm
from classifier.voice_svm import model_generator as m_svm
from utils import utility as util
import os

rd = r'C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Training Data\individual'
repDir = r'C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\reports\Clf_Reports'
cacheDir = r'G:\PROJECTS\Voice\lazarus\resources'

svm_tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6],
                     'C': [1, 10, 100, 1000, 10000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000]}]

if __name__ == "__main__":
    util.createDir(repDir)
    print('Evaluating Various classifiers')
    print()
    print('Evaluating HMM with windowed features')
    hmmfeat.evaluate(n_folds=5,
                     n_states_l=[10],
                     rootDir=rd,
                     reportDir=repDir,
                     cacheDir=os.path.join(cacheDir,'hmmfeatures'),
                     modelGenerator = m_hmmfeat,
                     prnt=False,
                     filewrt=True,
                     cacherefresh=False)
    print()
    print('Evaluating HMM with raw data')
    hmmraw.evaluate(n_folds=5,
                     n_states_l=[10],
                     rootDir=rd,
                     reportDir=repDir,
                     modelGenerator=m_hmmraw,
                     prnt=False,
                     filewrt=True)

    print()
    print('Evaluating KNN with global features')
    knn.evaluate(n_folds=5,
                 n_neighbours=[1],
                 rootDir=rd,
                 reportDir=repDir,
                 cacheDir=os.path.join(cacheDir, 'globalfeatures'),
                 modelGenerator=m_knn,
                 prnt=False,
                 filewrt=True,
                 cacherefresh=False)

    print()
    print('Evaluating Naive Bayes with global features')
    nb.evaluate(n_folds=5,
                 algos=['gauss'],
                 rootDir=rd,
                 reportDir=repDir,
                 cacheDir=os.path.join(cacheDir, 'globalfeatures'),
                 modelGenerator=m_nb,
                 prnt=False,
                 filewrt=True,
                 cacherefresh=False)

    print()
    print('Evaluating SVM with global features')
    svm.evaluate(n_folds=5,
                tp=svm_tuned_parameters,
                rootDir=rd,
                reportDir=repDir,
                cacheDir=os.path.join(cacheDir, 'globalfeatures'),
                modelGenerator=m_svm,
                prnt=False,
                filewrt=True,
                cacherefresh=False)
    print()
    print('Done!!!!')
