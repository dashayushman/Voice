
import json
import os
from datasource import TrainingInstance as tri
import numpy as np
from utils import feature_extractor as fe
import pickle
import random

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

def getTrainingData(rootdir):
    training_class_dirs = os.walk(rootdir)
    labels = []
    labelsdict = {}
    labeldirs = []
    target = []
    data = []
    sample_len_vec = []
    data_dict = {}
    user_map = {}
    user_list = []
    user_ids = np.arange(100).tolist()

    for trclass in training_class_dirs:
        labels = trclass[1]
        break

    for i,label in enumerate(labels):
        dict = {}
        lbl_users_lst = []
        labelsdict[label] = i
        labeldir = os.path.join(rootdir,label)
        lbl_usrs = os.walk(labeldir)

        for usr in lbl_usrs:
            #print(usr)
            lbl_users_lst = usr[1]
            for i,user in enumerate(lbl_users_lst):
                if user not in user_map:
                    id = user_ids.pop()
                    user_map[user] =id
                    user_list.append(id)
            break

        for usr in lbl_users_lst:
            usrdir = os.path.join(labeldir,usr)
            filewalk = os.walk(usrdir)
            file_list = []
            for fls in filewalk:
                file_list = fls[2]
                break
            dict[usr] = (usrdir,file_list)

        dict['users'] = lbl_users_lst
        data_dict[label] = dict

    for key, value in data_dict.items():
        tar_val = int(key)
        users = value['users']
        for user in users:
            user_dir = value[user]
            dirPath = user_dir[0]
            filelist = user_dir[1]
            for file in filelist:
                fileData = read_json_file(os.path.join(dirPath,file))

                # extract data from the dictionary

                # emg
                emg = fe.max_abs_scaler.fit_transform(np.array(fileData['emg']['data']))
                emgts = np.array(fileData['emg']['timestamps'])

                # accelerometer
                acc = fe.max_abs_scaler.fit_transform(np.array(fileData['acc']['data']))
                accts = np.array(fileData['acc']['timestamps'])

                # gyroscope
                gyr = fe.max_abs_scaler.fit_transform(np.array(fileData['gyr']['data']))
                gyrts = np.array(fileData['gyr']['timestamps'])

                # orientation
                ori = fe.max_abs_scaler.fit_transform(np.array(fileData['ori']['data']))
                orits = np.array(fileData['ori']['timestamps'])

                # create training instance
                ti = tri.TrainingInstance(key, emg, acc, gyr, ori, emgts, accts, gyrts, orits)

                # add length for resampling later to the sample length vector
                sample_len_vec.append(emg.shape[0])
                sample_len_vec.append(acc.shape[0])

                # split raw data
                ti.separateRawData()

                # append training instance to data list
                data.append(ti)

                # append class label to target list
                target.append(tar_val)
    '''
    for i,labeldir in enumerate(labeldirs):
        dirPath = labeldir[0]
        filelist = labeldir[1]
        for file in filelist:
            fileData = read_json_file(dirPath + os.pathsep +file)

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
            ti = tri.TrainingInstance(labels[i],emg,acc,gyr,ori,emgts,accts,gyrts,orits)

            #add length for resampling later to the sample length vector
            sample_len_vec.append(emg.shape[0])
            sample_len_vec.append(acc.shape[0])

            #split raw data
            ti.separateRawData()

            #append training instance to data list
            data.append(ti)

            #append class label to target list
            target.append(labels[i])
    '''
    avg_len = int(np.mean(sample_len_vec))
    return labels,data,target,labelsdict,avg_len,user_map,user_list,data_dict

def resampleTrainingData(data,sample_length):
    data = np.array([ti.resampleData(sample_length) for ti in data])
    return data

def extractFeatures(data,window=True):
    data = np.array([ti.extractFeatures(window) for ti in data])
    return data

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
    for tid in trainingIndexes:
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


def prepareTrainingDataHmmRaw(trainingIndexes, target, data):
    trainingData = {}
    for tid in trainingIndexes:
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

