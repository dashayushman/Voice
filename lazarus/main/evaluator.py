'''
This script evaluates all models on the same data
It consists of methods that call individual modules of different classifiers and evaluates them and dumps their classification report to a file
'''
#from lazarus.classifier.voice_lstm_raw import evaluation as lstmraw
#from lazarus.classifier.voice_lstm_raw import model_generator as m_lstmraw

from lazarus.classifier.voice_lstm_dynamic_length import evaluation as lstmDL
from lazarus.classifier.voice_lstm_dynamic_length import model_generator as m_lstmDL

from lazarus.classifier.voice_cnn_raw import evaluation as cnn
from lazarus.classifier.voice_cnn_raw import model_generator as m_cnn

#from lazarus.classifier.voice_hmm_feature import evaluation as hmmfeat
#from lazarus.classifier.voice_hmm_feature import model_generator as m_hmmfeat
#from lazarus.classifier.voice_hmm_raw import evaluation as hmmraw
#from lazarus.classifier.voice_hmm_raw import model_generator as m_hmmraw
#from lazarus.classifier.voice_knn import evaluation as knn
#from lazarus.classifier.voice_knn import model_generator as m_knn
#from lazarus.classifier.voice_naive_bayes import evaluation as nb
#from lazarus.classifier.voice_naive_bayes import model_generator as m_nb
#from lazarus.classifier.voice_svm import evaluation as svm
#from lazarus.classifier.voice_svm import model_generator as m_svm
from lazarus.utils import utility as util
from sklearn import preprocessing
import os

# The root directory where we have the ground truth (Training and validation data)

#windows
#rd = r'C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Training Data\individual\individual_ayushman'

#linux
rd = r'/home/amit/Desktop/voice/individual_all'


# Directory path to store the classification report

#windows
#repDir = r'C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\reports\Clf_Reports'

#linux
repDir = r'/home/amit/Desktop/voice/resources/data/clfreports'


# Directory to look for cached data (Pre-computed features) to avoid calculating them again and again

#windows
#cacheDir = r'G:\PROJECTS\Voice\lazarus\resources'

#linux
cacheDir = r'/home/amit/Desktop/voice/resources'

# parameters to do a grid search on SVM
svm_tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6],
                     'C': [1, 10, 100, 1000, 10000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000]}]

# scaler used to rescale the data
max_abs_scaler = preprocessing.MaxAbsScaler()

if __name__ == "__main__":
    # Creating a directory for storing the classification report if it does not exist
    util.createDir(repDir)



    print('Evaluating Various classifiers')
    print()
    '''
    # Evaluate CNN with raw data
    print('Evaluating CNN with windowed features')
    cnn.evaluate(n_folds=5,                                         # Number of folds for cross validation
                     rootDir=rd,                                        # Root directory for ground truth
                     reportDir=repDir,                                  # Directory to store the classification report
                     modelGenerator=m_cnn,                          # Reference to the model generator object for creating a lstm model
                     sampling_rate=20,                                  # new sampling rate for normalizing the data
                     prnt=False,                                        # To print the classification Report in the console
                     filewrt=True,                                      # To write the classification report to a file
                     scaler = max_abs_scaler)                           # scaler used for scaling the sensor data
    print()

    '''
    # Evaluate LSTM with CTC with raw data
    print('Evaluating LSTM with CTC with windowed features')
    lstmDL.evaluate(n_folds=5,  # Number of folds for cross validation
                     rootDir=rd,  # Root directory for ground truth
                     reportDir=repDir,  # Directory to store the classification report
                     modelGenerator=m_lstmDL,  # Reference to the model generator object for creating a lstm model
                     sampling_rate=20,  # new sampling rate for normalizing the data
                     prnt=False,  # To print the classification Report in the console
                     filewrt=True,  # To write the classification report to a file
                     scaler=max_abs_scaler)  # scaler used for scaling the sensor data
    print()
    '''

    # Evaluate LSTM with raw data
    print('Evaluating LSTM with windowed features')
    lstmraw.evaluate(n_folds=5,                                         # Number of folds for cross validation
                     rootDir=rd,                                        # Root directory for ground truth
                     reportDir=repDir,                                  # Directory to store the classification report
                     modelGenerator=m_lstmraw,                          # Reference to the model generator object for creating a lstm model
                     sampling_rate=20,                                  # new sampling rate for normalizing the data
                     prnt=False,                                        # To print the classification Report in the console
                     filewrt=True,                                      # To write the classification report to a file
                     scaler = max_abs_scaler)                           # scaler used for scaling the sensor data
    print()

    # Evaluate Hidden Markov Model based Classifier with features extracted using a sliding window
    print('Evaluating HMM with windowed features')
    hmmfeat.evaluate(n_folds=5,                                         # Number of folds for cross validation
                     n_states_l=[18],                                   # Number of states in the HMM
                     rootDir=rd,                                        # Root directory for ground truth
                     reportDir=repDir,                                  # Directory to store the classification report
                     cacheDir=os.path.join(cacheDir,'hmmfeatures'),     # Path to the cache directory
                     modelGenerator = m_hmmfeat,                        # Reference to the model generator object for creating a Hidden Markov Model
                     prnt=False,                                        # To print the classification Report in the console
                     filewrt=True,                                      # To write the classification report to a file
                     cacherefresh=False)                                # to delete and recreate the cache file before running any evaluation
    print()


    # Evaluate Hidden Markov Model based Classifier with raw data
    print('Evaluating HMM with raw data')
    hmmraw.evaluate(n_folds=5,                                          # Number of folds for cross validation
                     n_states_l=[10],                                   # Number of states in the HMM
                     rootDir=rd,                                        # Root directory for ground truth
                     reportDir=repDir,                                  # Directory to store the classification report
                     modelGenerator=m_hmmraw,                           # Reference to the model generator object for creating a Hidden Markov Model
                     sampling_rate=20,                                  # new sampling rate for normalizing the data
                     prnt=False,                                        # To print the classification Report in the console
                     filewrt=True,                                      # To write the classification report to a file
                     scaler = max_abs_scaler)                           # scaler used for scaling the sensor data
    print()


    # Evaluate K- Nearest Neighbors Classifier with global features
    print('Evaluating KNN with global features')
    knn.evaluate(n_folds=5,                                             # Number of folds for cross validation
                 n_neighbours=[1],                                      # Number of neighbors in KNN
                 rootDir=rd,                                            # Root directory for ground truth
                 reportDir=repDir,                                      # Directory to store the classification report
                 cacheDir=os.path.join(cacheDir, 'globalfeatures'),     # Path to the cache directory
                 modelGenerator=m_knn,                                  # Reference to the model generator object for creating a K-Nearest Neighbor Classifier
                 sampling_rate=20,                                      # new sampling rate for normalizing the data
                 prnt=False,                                            # To print the classification Report in the console
                 filewrt=True,                                          # To write the classification report to a file
                 cacherefresh=False,                                    # to delete and recreate the cache file before running any evaluation
                 scaler = max_abs_scaler)                               # scaler used for scaling the sensor data
    print()

    # Evaluate Naive Bayes Classifier with global features
    print('Evaluating Naive Bayes with global features')
    nb.evaluate(n_folds=5,                                              # Number of folds for cross validation
                 algos=['gauss'],                                       # Types of data distribution you want to evaluate the model with
                 rootDir=rd,                                            # Root directory for ground truth
                 reportDir=repDir,                                      # Directory to store the classification report
                 cacheDir=os.path.join(cacheDir, 'globalfeatures'),     # Path to the cache directory
                 modelGenerator=m_nb,                                   # Reference to the model generator object for creating a Naive Bayes Classifier
                 sampling_rate=20,                                      # new sampling rate for normalizing the data
                 prnt=False,                                            # To print the classification Report in the console
                 filewrt=True,                                          # To write the classification report to a file
                 cacherefresh=False,                                    # to delete and recreate the cache file before running any evaluation
                 scaler = max_abs_scaler)                               # scaler used for scaling the sensor data
    print()

    # Evaluate Support vector Machine with global features
    print('Evaluating SVM with global features')
    svm.evaluate(n_folds=5,                                             # Number of folds for cross validation
                  tp=svm_tuned_parameters,                              # Paramerters for doing a grid search to find the best parameters
                  rootDir=rd,                                           # Root directory for ground truth
                  reportDir=repDir,                                     # Directory to store the classification report
                  cacheDir=os.path.join(cacheDir, 'globalfeatures'),    # Path to the cache directory
                  modelGenerator=m_svm,                                 # Reference to the model generator object for creating a Naive Bayes Classifier
                  sampling_rate=20,                                     # new sampling rate for normalizing the data
                  prnt=False,                                           # To print the classification Report in the console
                  filewrt=True,                                         # To write the classification report to a file
                  cacherefresh=False,                                   # to delete and recreate the cache file before running any evaluation
                  scaler=max_abs_scaler)                                # scaler used for scaling the sensor data
    print()
    '''
    print('Done!!!!')

