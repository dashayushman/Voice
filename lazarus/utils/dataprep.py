#Import dependencies
import json
import os
from lazarus.datasource import TrainingInstance as tri
import numpy as np
from lazarus.utils import feature_extractor as fe
import pickle
import random
from sklearn.model_selection import KFold
#from sklearn.cross_validation import StratifiedKFold

def scaleData(data,scaler):
    '''
    Method to scale the sensor data as a preprocessing step
    :param data: (list) List of all the training instance objects
    :return: data: (list) List of a ll the scaled training data objects
    '''

    #data = np.array([ti.scaleData(scaler) for ti in data])
    d = []
    for i, ti in enumerate(data):
        if i % 50 is 0:
            print(str(i) + ' out of ' + str(len(data)) + 'training instances scaled')
        d.append(ti.scaleData(scaler))
    return np.array(d)

# Serielize objects to disk for future reuse to make things faster
def dumpObject(filePath,object):
    try:
        with open(filePath, 'wb') as f:
            pickle.dump(object, f)
            return True
    except IOError as e:
        return False


def loadObject(filePath):
    if os.path.isfile(filePath):
        with open(filePath, 'rb') as f:
            object = pickle.load(f)
            return object
    else:
        return None

def read_json_file(filepath):
    with open(filepath) as data_file:
        data = json.load(data_file)
        return data

# Get training data from the root directory where the ground truth exists

#using spline interpolation
def createSyntheticTrainingData(rootdir):
    training_class_dirs = os.walk(rootdir)
    labels = []
    labeldirs = []
    target = []
    data = []
    skip = True
    for trclass in training_class_dirs:
        #print(trclass)
        if skip is True:
            labels = trclass[1]
            skip = False
            continue
        labeldirs.append((trclass[0],trclass[2]))

    for i,labeldir in enumerate(labeldirs):
        dirPath = labeldir[0]
        filelist = labeldir[1]
        if not bool(filelist):
            continue
        for file in filelist:
            fileData = read_json_file(dirPath + os.path.sep +file)

            #extract data from the dictionary

            #emg
            emg = fe.max_abs_scaler.fit_transform(np.array(fileData['emg']['data']))
            emgts = np.array(fileData['emg']['timestamps'])

            #accelerometer
            acc = fe.max_abs_scaler.fit_transform(np.array(fileData['acc']['data']))
            accts = np.array(fileData['acc']['timestamps'])

            # gyroscope
            gyr = fe.max_abs_scaler.fit_transform(np.array(fileData['gyr']['data']))
            gyrts = np.array(fileData['gyr']['timestamps'])

            # orientation
            ori = fe.max_abs_scaler.fit_transform(np.array(fileData['ori']['data']))
            orits = np.array(fileData['ori']['timestamps'])

            #create training instance
            #ti = tri.TrainingInstance(labels[i],emg,acc,gyr,ori,emgts,accts,gyrts,orits)

            #append training instance to data list
            #data.append(ti)

            #append class label to target list
            #target.append(labels[i])

    return labels,data,target


def getTrainingData(rootdir):
    '''
    This method gets all the training data from the root directory of the ground truth
    The following is the directory structure for the ground truth
    Root_Dir
        |_Labels
            |_Participants
                |_data_files
    :param rootdir (string): path to the rood directory where the ground truth exists
    :return:    labels      (list),                 A list of class labels
                data        (list),                 A list of training instances
                target      (list),                 A list of class labels corresponding to the training instances in the in 'data'
                labelsdict  (dictionary),           A dictionary for converting class labels from string to integer and vice versa
                avg_len     (float),                The average length of the sensor data (emg, accelerometer, gyroscope and orientation) which would later be used for normalization
                user_map    (dictionary),           A dictionary of all participants and their corresponding file list to be used for leave one out test later
                user_list   (list),                 A list of all participants
                data_dict   (dictionary)            A dictionary containing a mapping of all the class labels, participants, and files of the participants which can be used later for transforming the data for leave one out test
                max_len     (integer)               the maximum length of the sensor data
                data_path   (list)                  A list that will hold the path to every training instance in the 'data list'
    '''

    # List of all training labels
    training_class_dirs = os.walk(rootdir)


    labels = []                         # Empty list to hold all the class labels
    labelsdict = {}                     # Dictionary to store the labels and their correspondig interger values
    labeldirs = []                      # Directory paths of all labels
    target = []                         # List that will hold class labels of the training instances in 'data list'
    data = []                           # List that will hold all the training/validation instances
    sample_len_vec_emg = []             # List that holds that length of all the sensor data. It will be used later for calculating average length
    sample_len_vec_others = []          # List that holds that length of all the sensor data. It will be used later for calculating average length
    data_dict = {}                      # The dictionary that will hold that mappings for all labels, participants of the the label and data files corresponding to all the participants. This will be used later for leave one out test
    user_map = {}                       # A dictionary that will hold the mappings for all participants and their corresponding ids
    user_list = []                      # A list of all participants
    user_ids = np.arange(100).tolist()  # A pre generated list of userids for assigning a unique id to every user
    data_path = []                      # A list that will hold the path to every training instance in the 'data list'

    # Get the list of labels by walking the root directory
    for trclass in training_class_dirs:
        labels = trclass[1]
        break

    # extracting list of participants for each label
    for i,label in enumerate(labels):
        dict = {}                                   # dictionary to store participant information
        lbl_users_lst = []                          # list of participants per label
        labelsdict[label] = i
        labeldir = os.path.join(rootdir,label)

        #list of users for the respective label
        lbl_usrs = os.walk(labeldir)

        #enumerating all the users of the respective label
        for usr in lbl_usrs:
            #print(usr)
            lbl_users_lst = usr[1]

            #assigning unique ids to all the users
            for i,user in enumerate(lbl_users_lst):
                if user not in user_map:
                    id = user_ids.pop()
                    user_map[user] =id
                    user_list.append(id)
            break

        #extracting data file list for every  participant
        for usr in lbl_users_lst:
            usrdir = os.path.join(labeldir,usr)
            filewalk = os.walk(usrdir)
            file_list = []
            for fls in filewalk:
                file_list = fls[2]
                break
            dict[usr] = (usrdir,file_list)

        dict['users'] = lbl_users_lst
        data_dict[label] = dict                 # add all meta information to data_dict

    # Extracting data from the data files from all participants
    for key, value in data_dict.items():
        tar_val = int(key)
        users = value['users']
        for user in users:
            user_dir = value[user]
            dirPath = user_dir[0]
            filelist = user_dir[1]
            for file in filelist:
                fp = os.path.join(dirPath,file)

                data_path.append(fp)

                fileData = read_json_file(fp)
                # extract data from the dictionary
                # emg
                emg = np.array(fileData['emg']['data'])
                emgts = np.array(fileData['emg']['timestamps'])

                # accelerometer
                acc = np.array(fileData['acc']['data'])
                accts = np.array(fileData['acc']['timestamps'])

                # gyroscope
                gyr = np.array(fileData['gyr']['data'])
                gyrts = np.array(fileData['gyr']['timestamps'])

                # orientation
                ori = np.array(fileData['ori']['data'])
                orits = np.array(fileData['ori']['timestamps'])

                # create training instance
                ti = tri.TrainingInstance(key, emg, acc, gyr, ori, emgts, accts, gyrts, orits)

                # add length for resampling later to the sample length vector
                sample_len_vec_emg.append(emg.shape[0])
                sample_len_vec_others.append(acc.shape[0])

                # split raw data
                ti.separateRawData()

                # append training instance to data list
                data.append(ti)

                # append class label to target list
                target.append(tar_val)
    avg_len_emg = int(np.mean(sample_len_vec_emg))
    avg_len_acc = int(np.mean(sample_len_vec_others))
    max_length_emg = np.amax(sample_len_vec_emg)
    max_length_others = np.amax(sample_len_vec_others)
    return labels,data,target,labelsdict,avg_len_emg,avg_len_acc,user_map,user_list,data_dict,max_length_emg,max_length_others,data_path

def normalizeTrainingData(data,max_length_emg,max_len_others):
    '''
    Method to normalize the training data to fixed length
    :param data: (list) List of all the training instance objects
    :param max_length_emg: (int) Normalized length for EMG signals
    :param max_len_others: (int) Normalized length of IMU signals
    :return: data (list) List of all the normalized training instance objects
    '''
    #data = np.array([ti.normalizeData(max_length_emg,max_len_others) for ti in data])
    d = []
    for i, ti in enumerate(data):
        if i % 50 is 0:
            print(str(i) + ' out of ' + str(len(data)) + 'training instances normalized')
        if i is 2 or i is 4:
            print('case')
        d.append(ti.normalizeData(max_length_emg,max_len_others))
    return np.array(d)

def resampleTrainingData(data,sampling_rate,avg_len,emg=True,imu=True):
    '''
    Method to resample the training instances to a given sampling frequency in HZ.
    It calls consolidate data implicitly.
    Can remove the consolidation to a different method.
    :param data: (list) List of all the training instance objects
    :param sampling_rate: (int) The new sampling rate in Hz
    :param avg_len: (int) Average length of vectors in case both EMG and IMU needs to be resampled and consolidated
    :param emg: (boolean) Flag to indicate that we need to consider emg signals for consolidating the data after resampling
    :param imu: (boolean) Flag to indicate that we need to consider IMU signals for consolidating the data after resampling
    :return: data : resampled data
    '''
    #data = np.array([ti.resampleData(sampling_rate,avg_len,emg,imu) for ti in data])
    d = []
    for i, ti in enumerate(data):
        if i % 50 is 0:
            print(str(i) + ' out of ' + str(len(data)) + 'training instances normalized')
        d.append(ti.resampleData(sampling_rate,avg_len,emg,imu))
    return np.array(d)

def extractFeatures(data,scaler = None,window=True,rms=False,f_mfcc=False,emg=True,imu=True):
    '''
    method to loop through all training instances and extract features from the signals
    @params: data (list)                : list of training instances
             scaler (sclaer object)     : scaler object to scale features if necessary
             window (Boolean)           : To get overlapping window features
             rms (Boolean)              : To get features from the rms value of the signals in all directions
             f_mfcc (Boolean)           :  to get the MFCC features as well
    @return: data (list)                : list of training instances with extracted features
    '''
    #data = np.array([ti.extractFeatures(window,scaler,rms,f_mfcc,emg,imu) for ti in data])
    d = []
    for i,ti in enumerate(data):
        if i%20 is 0:
            print('features extracted from '+str(i)+' out of ' + str(len(data)) +'training instances')
        d.append(ti.extractFeatures(window, scaler, rms, f_mfcc, emg, imu))
    return np.array(d)
'''
def prepareDataPC(target, data):
    consolidate = zip(target,data)
    for lbl,d in consolidate:

        con_mat = d.getConsolidatedFeatureMatrix()
        if train_x is None:
            train_x = con_mat
        else:
            train_x = np.append(train_x,con_mat,axis=0)
        train_y.append(int(key))


    return np.array(train_x),np.array(train_y),np.array(test_x),np.array(test_y)
'''
def prepareTrainingDataSvm(trainingIndexes,testingIndexes, target, data):
    train_x = None    #training data
    train_y = []    #training labels

    test_x = None     #testing data
    test_y = []     #testing labels

    for tid in trainingIndexes:
        key = target[tid]
        ti = data[tid]
        con_mat = ti.getConsolidatedFeatureMatrix()
        if train_x is None:
            train_x = con_mat
        else:
            train_x = np.append(train_x,con_mat,axis=0)
        train_y.append(int(key))

    for tid in testingIndexes:
        key = target[tid]
        ti = data[tid]
        con_mat = ti.getConsolidatedFeatureMatrix()
        if test_x is None:
            test_x = con_mat
        else:
            test_x = np.append(test_x,con_mat,axis=0)
        test_y.append(int(key))

    return np.array(train_x),np.array(train_y),np.array(test_x),np.array(test_y)

def prepareTrainingDataHmmFeatures(trainingIndexes, target, data):
    trainingData = {}
    for l,tid in enumerate(trainingIndexes):

        # printing corrent status so that the wait is not too boring :-P
        if l % 50 is 0:
            print(str(l) + ' out of ' + str(len(trainingIndexes)) + 'training instances prepared')

        key = target[tid]
        ti = data[tid]
        #con_data = ti.getConsolidatedDataMatrix()
        if key in trainingData:

            # get data from existing dictionary
            trld = trainingData.get(key)
            lbl_data = trld.get('data')
            n_data = trld.get('datal')
            # extract data from the training instance

            #get consolidated data matrix
            con_mat = ti.getConsolidatedFeatureMatrix()

            # append
            lbl_data = np.append(lbl_data, con_mat, axis=0)
            n_data.append(con_mat.shape[0])

            # replace in the existing dict
            trld['data'] = lbl_data
            trld['datal'] = n_data

            trainingData[key] = trld

        else:
            trld = {}
            # extract others and get features for creating an svm model
            con_mat = ti.getConsolidatedFeatureMatrix()

            trld['data'] = con_mat
            trld['datal'] = [con_mat.shape[0]]

            trainingData[key] = trld

    return trainingData

def discritizeLabels(target):
    n_classes = np.unique(target)
    d_labels = []
    for t in target:
        d_l = np.zeros(n_classes.size,dtype=np.int)
        d_l[t] = 1
        d_labels.append(d_l)
    return np.array(d_labels)

def splitDataset(train,test,target,data):
    train_x = np.take(data,train,axis=0)
    train_y = np.take(target,train,axis=0)

    val_x = np.take(data,test,axis=0)
    val_y = np.take(target,test,axis=0)

    return train_x,train_y,val_x,val_y

def prepareDataset(data):
    d = []
    for i,ti in enumerate(data):
        if i%20 is 0:
            print('prepared '+str(i)+' out of ' + str(len(data)) +'instances')
        d.append(ti.consolidatedDataMatrix)
    return np.array(d)

def prepareTrainingDataHmmRaw(trainingIndexes, target, data):
    trainingData = {}
    for l,tid in enumerate(trainingIndexes):
        #printing corrent status so that the wait is not too boring :-P
        if l % 50 is 0:
            print(str(l) + ' out of ' + str(len(trainingIndexes)) + 'training instances prepared')

        key = target[tid]
        ti = data[tid]
        #con_data = ti.getConsolidatedDataMatrix()
        if key in trainingData:

            # get data from existing dictionary
            trld = trainingData.get(key)
            lbl_data = trld.get('data')
            n_data = trld.get('datal')
            # extract data from the training instance

            #get consolidated data matrix
            con_mat = ti.getConsolidatedDataMatrix()

            # append
            lbl_data = np.append(lbl_data, con_mat, axis=0)
            n_data.append(con_mat.shape[0])

            # replace in the existing dict
            trld['data'] = lbl_data
            trld['datal'] = n_data

            trainingData[key] = trld

        else:
            trld = {}
            # extract others and get features for creating an svm model
            con_mat = ti.getConsolidatedDataMatrix()

            trld['data'] = con_mat
            trld['datal'] = [con_mat.shape[0]]

            trainingData[key] = trld

    return trainingData



def prepareTrainingData(trainingIndexes, target, data):
    #dictionary that holds all the consolidated training data
    trainingDict = {}

    for tid in trainingIndexes:
        key = target[tid]
        ti = data[tid]
        #call separate raw data to create models for the others but for now use raw data
        if key in trainingDict:

            #get data from existing dictionary
            trld = trainingDict.get(key)
            emg = trld.get('emg')
            emgl = trld.get('emgl')

            acc = trld.get('acc')
            accl = trld.get('accl')

            gyr = trld.get('gyr')
            gyrl = trld.get('gyrl')

            ori = trld.get('ori')
            oril = trld.get('oril')

            #extract data from the training instance
            emg_t,acc_t,gyr_t,ori_t = ti.getRawData()

            #append
            emg = np.append(emg,emg_t,axis=0)
            emgl.append(len(emg_t))

            acc = np.append(acc, acc_t, axis=0)
            accl.append(len(acc_t))

            gyr = np.append(gyr, gyr_t, axis=0)
            gyrl.append(len(gyr_t))

            ori = np.append(ori, ori_t, axis=0)
            oril.append(len(ori_t))

            #replace in the existing dict
            trld['emg'] = emg
            trld['emgl'] = emgl

            trld['acc'] = acc
            trld['accl'] = accl

            trld['gyr'] = gyr
            trld['gyrl'] = gyrl

            trld['ori'] = ori
            trld['oril'] = oril

            trainingDict[key] = trld

        else:
            trld = {}
            #extract others and get features for creating an svm model
            emg_t, acc_t, gyr_t, ori_t = ti.getRawData()

            trld['emg'] = emg_t
            trld['emgl'] = [len(emg_t)]

            trld['acc'] = acc_t
            trld['accl'] = [len(acc_t)]

            trld['gyr'] = gyr_t
            trld['gyrl'] = [len(gyr_t)]

            trld['ori'] = ori_t
            trld['oril'] = [len(ori_t)]

            trainingDict[key] = trld

    return trainingDict

#For CTC implementation
def target_list_to_sparse_tensor(targetList):
    '''make tensorflow SparseTensor from list of targets, with each element
       in the list being a list or array with the values of the target sequence
       (e.g., the integer values of a character map for an ASR target string)
       See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
       for example of SparseTensor format'''
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))

#For CTC implementation
def data_lists_to_batches(inputList, targetList, batchSize, trainMaxSteps = 0):
    '''Takes a list of input matrices and a list of target arrays and returns
       a list of batches, with each batch being a 3-element tuple of inputs,
       targets, and sequence lengths.
       inputList: list of 2-d numpy arrays with dimensions nFeatures x timesteps
       targetList: list of 1-d arrays or lists of ints
       batchSize: int indicating number of inputs/targets per batch
       returns: dataBatches: list of batch data tuples, where each batch tuple (inputs, targets, seqLengths) consists of
                    inputs = 3-d array w/ shape nTimeSteps x batchSize x nFeatures
                    targets = tuple required as input for SparseTensor
                    seqLengths = 1-d array with int number of timesteps for each sample in batch
                maxSteps: maximum number of time steps across all samples'''

    assert len(inputList) == len(targetList)
    nFeatures = inputList[0].shape[0]
    maxSteps = 0
    for inp in inputList:
        maxSteps = max(maxSteps, inp.shape[1])

    if(trainMaxSteps):
        maxSteps = trainMaxSteps
    randIxs = np.random.permutation(len(inputList))
    start, end = (0, batchSize)
    dataBatches = []

    while end <= len(inputList):
        batchSeqLengths = np.zeros(batchSize)
        for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]
        batchInputs = np.zeros((maxSteps, batchSize, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(randIxs[start:end]):
            padSecs = maxSteps - inputList[origI].shape[1]
            batchInputs[:, batchI, :] = np.pad(inputList[origI].T, ((0, padSecs), (0, 0)),
                                               'constant', constant_values=0)
            batchTargetList.append(targetList[origI])
        dataBatches.append((batchInputs, target_list_to_sparse_tensor(batchTargetList),
                            batchSeqLengths))
        start += batchSize
        end += batchSize
    return dataBatches, maxSteps


def separateRawData(acc, gyr, ori):
    if acc is not None:
        accList = np.array([np.array(acc[:, 0]), np.array(acc[:, 1]), np.array(acc[:, 2])])

    if gyr is not None:
        gyrList = np.array([np.array(gyr[:, 0]), np.array(gyr[:, 1]), np.array(gyr[:, 2])])

    if ori is not None:
        oriList = np.array(
            [np.array(ori[:, 0]), np.array(ori[:, 1]), np.array(ori[:, 2]), np.array(ori[:, 3])])

    return accList, gyrList, oriList


def getTrainingDataDrnn(rootdir, scaler):
    '''
    This method gets all the training data from the root directory of the ground truth
    The following is the directory structure for the ground truth
    Root_Dir
        |_Labels
            |_Participants
                |_data_files
    :param rootdir (string): path to the rood directory where the ground truth exists
    :return:    labels      (list),                 A list of class labels
                data        (list),                 A list of training instances
                target      (list),                 A list of class labels corresponding to the training instances in the in 'data'
                labelsdict  (dictionary),           A dictionary for converting class labels from string to integer and vice versa
                avg_len     (float),                The average length of the sensor data (emg, accelerometer, gyroscope and orientation) which would later be used for normalization
                user_map    (dictionary),           A dictionary of all participants and their corresponding file list to be used for leave one out test later
                user_list   (list),                 A list of all participants
                data_dict   (dictionary)            A dictionary containing a mapping of all the class labels, participants, and files of the participants which can be used later for transforming the data for leave one out test
                max_len     (integer)               the maximum length of the sensor data
                data_path   (list)                  A list that will hold the path to every training instance in the 'data list'
    '''

    # List of all training labels
    training_class_dirs = os.walk(rootdir)


    labels = []                         # Empty list to hold all the class labels
    labelsdict = {}                     # Dictionary to store the labels and their correspondig interger values
    labeldirs = []                      # Directory paths of all labels
    target = []                         # List that will hold class labels of the training instances in 'data list'
    data = []                           # List that will hold all the training/validation instances
    sample_len_vec_emg = []             # List that holds that length of all the sensor data. It will be used later for calculating average length
    sample_len_vec_others = []          # List that holds that length of all the sensor data. It will be used later for calculating average length
    data_dict = {}                      # The dictionary that will hold that mappings for all labels, participants of the the label and data files corresponding to all the participants. This will be used later for leave one out test
    user_map = {}                       # A dictionary that will hold the mappings for all participants and their corresponding ids
    user_list = []                      # A list of all participants
    user_ids = np.arange(100).tolist()  # A pre generated list of userids for assigning a unique id to every user
    data_path = []                      # A list that will hold the path to every training instance in the 'data list'

    # Get the list of labels by walking the root directory
    for trclass in training_class_dirs:
        labels = trclass[1]
        break

    # extracting list of participants for each label
    for i,label in enumerate(labels):
        dict = {}                                   # dictionary to store participant information
        lbl_users_lst = []                          # list of participants per label
        labelsdict[label] = i
        labeldir = os.path.join(rootdir,label)

        #list of users for the respective label
        lbl_usrs = os.walk(labeldir)

        #enumerating all the users of the respective label
        for usr in lbl_usrs:
            #print(usr)
            lbl_users_lst = usr[1]

            #assigning unique ids to all the users
            for i,user in enumerate(lbl_users_lst):
                if user not in user_map:
                    id = user_ids.pop()
                    user_map[user] =id
                    user_list.append(id)
            break

        #extracting data file list for every  participant
        for usr in lbl_users_lst:
            usrdir = os.path.join(labeldir,usr)
            filewalk = os.walk(usrdir)
            file_list = []
            for fls in filewalk:
                file_list = fls[2]
                break
            dict[usr] = (usrdir,file_list)

        dict['users'] = lbl_users_lst
        data_dict[label] = dict                 # add all meta information to data_dict

    # Extracting data from the data files from all participants
    for key, value in data_dict.items():
        tar_val = int(key)
        users = value['users']
        for user in users:
            user_dir = value[user]
            dirPath = user_dir[0]
            filelist = user_dir[1]
            for file in filelist:
                fp = os.path.join(dirPath,file)

                data_path.append(fp)

                fileData = read_json_file(fp)
                # extract data from the dictionary
                # emg
                emg = np.array(fileData['emg']['data'])
                emgts = np.array(fileData['emg']['timestamps'])

                # accelerometer
                acc = np.array(fileData['acc']['data'])
                accts = np.array(fileData['acc']['timestamps'])

                # gyroscope
                gyr = np.array(fileData['gyr']['data'])
                gyrts = np.array(fileData['gyr']['timestamps'])

                # orientation
                ori = np.array(fileData['ori']['data'])
                orits = np.array(fileData['ori']['timestamps'])

                # create training instance
                accList, gyrList, oriList = separateRawData(acc,gyr,ori)

                #scaling data
                accList, gyrList, oriList = scaledatactc(accList, gyrList, oriList, scaler)

                ti = np.concatenate((accList, gyrList, oriList), axis=0)  # consolidated data
                #ti = tri.TrainingInstance(key, emg, acc, gyr, ori, emgts, accts, gyrts, orits)

                # add length for resampling later to the sample length vector
                sample_len_vec_emg.append(emg.shape[0])
                sample_len_vec_others.append(acc.shape[0])

                # append training instance to data list
                data.append(ti)

                # append class label to target list
                target.append(tar_val)
    avg_len_emg = int(np.mean(sample_len_vec_emg))
    avg_len_acc = int(np.mean(sample_len_vec_others))
    max_length_emg = np.amax(sample_len_vec_emg)
    max_length_others = np.amax(sample_len_vec_others)
    return labels,data,target,labelsdict,avg_len_emg,avg_len_acc,user_map,user_list,data_dict,max_length_emg,max_length_others,data_path


def scaledatactc(accList, gyrList, oriList, scaler):
    # scaling data
    norm_accs = []
    norm_gyrs = []
    norm_oris = []
    for a, b in zip(accList, gyrList):
        a = a.reshape(-1, 1)
        a = scaler.fit_transform(a)
        reshaped_a = a.reshape(a.shape[0])
        norm_accs.append(reshaped_a)
        b = b.reshape(-1, 1)
        b = scaler.fit_transform(b)
        reshaped_b = b.reshape(a.shape[0])
        norm_gyrs.append(reshaped_b)

    for x in oriList:
        x = x.reshape(-1, 1)
        x = scaler.fit_transform(x)
        reshaped = x.reshape(x.shape[0])
        norm_oris.append(reshaped)
    return np.array(norm_accs), np.array(norm_gyrs), np.array(norm_oris)

def consolidateTrainingData(data,emg,imu,avg_len = 0):
    '''
    Method to consolidated the training data
    :param data: (list) List of all the training instance objects
    :param avg_len: (int) length to resample data to, not required when both emg and imu are not required simultaneously (no resampling)
    :param emg: (bollean) choice to include emg values
    :param imu: (bollean) choice to include imu values
    :return: data (list) List of all the consolidated training instance objects
    '''

    d = []
    for ti in data :
        #d.append(tri.TrainingInstance.consolidateData(ti,avg_len,False,True))
        d.append(ti.onlyConsolidateData(avg_len,False,True))
    return np.array(d)

def groupData(inputList,targetList,groupSize):
    '''
    This method groups inputList of single instance data into sequence of fixed groupSize
    :param inputList (List): containing data values of each instance
    :param targetList (List): list of labels corresponding to data
    :param groupSize (int): size of the sequence groups to be formed
    :return:    ginputList      (list), A list of sequences of data of size = groupSize
                data        (list), A list of sequences of labels of size = groupSize
    '''

    randIxs = np.random.permutation(len(inputList))

    gtargetList = []
    ginputList = []
    start, end = (0, groupSize)
    while end <= len(inputList):
        #shape = np.array(inputList[0]).shape
        #ginput = np.array([], dtype=np.float64).reshape(10,423)
        ginput = np.array([])
        gtarget = []
        #ginput = np.concatenate((inputList[orig] for orig in randIxs[start:end]),axis=1)
        for batchI, origI in enumerate(randIxs[start:end]):
            ginput = np.concatenate((ginput, inputList[origI]), axis=1) if ginput.size else inputList[origI]
            gtarget.append(targetList[origI])
        gtargetList.append(gtarget)
        ginputList.append(ginput)

        start += groupSize
        end += groupSize

    return ginputList, gtargetList


# def load_batched_data(rootdir, batchSize, scaler):
#     '''returns 3-element tuple: batched data (list), max # of time steps (int), and
#        total number of samples (int)'''
#     _, inputList, targetList, _, _, _, _, _, _, _, _, _ = getTrainingDataDrnn(rootdir,scaler)
#
#     ginputList, gtargetList = groupData(inputList,targetList,groupSize=1)
#
#     return data_lists_to_batches(ginputList, gtargetList, batchSize) + (len(gtargetList),)

def splitDynSizeDataset(train,test,target,data):
    train_x = []
    train_y = []

    for i in train:
        train_x.append(data[i])
        train_y.append(target[i])

    val_x = []
    val_y = []

    for i in test:
        val_x.append(data[i])
        val_y.append(target[i])

    return train_x, train_y, val_x, val_y

def read_data_sets(rootdir,
                   scaler,
                   n_folds):
    '''returns 3-element tuple: batched data (list), max # of time steps (int), and
           total number of samples (int)'''
    _, inputList, targetList, _, _, _, _, _, _, _, _, _ = getTrainingDataDrnn(rootdir, scaler)

    ginputList, gtargetList = groupData(inputList, targetList, groupSize=5)

    skf = KFold(n_folds)
    if n_folds > 0:
        kFolds = []
        # for train, test in skf:
        #     print('split train and validation data')
        #     train_x,train_y,val_x,val_y = splitDataset(train,test,gtargetList,ginputList)
        #     #train = data_lists_to_batches(train_x, train_y, batchSize) + (len(train_y),)
        #     #validation = data_lists_to_batches(val_x, val_y, batchSize) + (len(val_y),)
        for train_index, test_index in skf.split(ginputList, gtargetList):
            print("TRAIN:", train_index, "TEST:", test_index)
            train_x, train_y, val_x, val_y = splitDynSizeDataset(train_index, test_index, gtargetList, ginputList)
            #train_x, val_x = ginputList[train_index], ginputList[test_index]
            #train_y, val_y = gtargetList[train_index], gtargetList[test_index]
            train = (train_x, train_y)
            validation = (val_x, val_y)
            kFolds.append((train,validation))
        return kFolds
    else:
        return ginputList, gtargetList

#
def next_batch(train, batch_size, step):
    '''
    This method returns data from train dataset in the form of x, y batches, looping over all the batches
    :param train: Dataset object containing instances and labels
    :param batch_size (int): size of a batch
    :param step (int): epoch step in training
    :return:    labels      (list),                 A list of class labels
                data        (list)                  A list of training instances
    '''

    no_batches = int(len(train.labels) / batch_size)
    start = (step % no_batches) * batch_size
    end = start + batch_size

    return train.instances[start:end], train.labels[start:end]
