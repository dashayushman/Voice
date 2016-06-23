
import json
import os
from datasource import TrainingInstance as tri
import numpy as np
from utils import feature_extractor as fe

def read_json_file(filepath):
    with open(filepath) as data_file:
        data = json.load(data_file)
        return data

def getTrainingData(rootdir):
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
        for file in filelist:
            fileData = read_json_file(dirPath + '\\' +file)

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

            #append training instance to data list
            data.append(ti)

            #append class label to target list
            target.append(labels[i])

    return labels,data,target

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

