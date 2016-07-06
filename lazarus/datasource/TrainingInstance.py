import numpy as np
from scipy import signal
from utils import feature_extractor as fe

class TrainingInstance:

    def __init__(self,label,emg,acc,gyr,ori,emgts=None,accts=None,gyrts=None,orits=None):

        self.m_label = label
        #raw data
        self.emg = emg
        self.acc = acc
        self.gyr = gyr
        self.ori = ori

        #time stamps
        self.emgts = emgts
        self.accts = accts
        self.gyrts = gyrts
        self.orits = orits

        #splitted flag
        self.splitted = False
        self.consolidated = False
        self.consolidatedFeatures = False

    def separateRawData(self):
        if self.emg is not None:
            self.emgList = np.array([np.array(self.emg[:,0]),np.array(self.emg[:,1]),np.array(self.emg[:,2]),np.array(self.emg[:,3]),np.array(self.emg[:,4]),np.array(self.emg[:,5]),np.array(self.emg[:, 6]),np.array(self.emg[:, 7])])

        if self.acc is not None:
            self.accList = np.array([np.array(self.acc[:, 0]),np.array(self.acc[:, 1]),np.array(self.acc[:, 2])])

        if self.gyr is not None:
            self.gyrList = np.array([np.array(self.gyr[:, 0]),np.array(self.gyr[:, 1]),np.array(self.gyr[:, 2])])

        if self.ori is not None:
            self.oriList = np.array([np.array(self.ori[:, 0]),np.array(self.ori[:, 1]),np.array(self.ori[:, 2]),np.array(self.ori[:, 3])])

        self.splitted = True

    def resampleData(self,sample_length):
        if self.splitted == True:
            self.emgList = np.array([signal.resample(x,sample_length) for x in self.emgList])
            self.accList = np.array([signal.resample(x, sample_length) for x in self.accList])
            self.gyrList = np.array([signal.resample(x, sample_length) for x in self.gyrList])
            self.oriList = np.array([signal.resample(x, sample_length) for x in self.oriList])
            self.consolidateData()
        return self

    def extractFeatures(self, window):
        if self.splitted == True:
            self.emgFeatures = np.array([fe.getFeatures(x,200,window) for x in self.emgList])
            self.accFeatures = np.array([fe.getFeatures(x,200,window) for x in self.accList])
            self.gyrFeatures = np.array([fe.getFeatures(x,200,window) for x in self.gyrList])
            self.oriFeatures = np.array([fe.getFeatures(x,200,window) for x in self.oriList])
            self.consolidateFeatures()
        return self

    def consolidateFeatures(self):
        if self.splitted == True:
            n_emg_rows = self.emgFeatures[0].shape[0]
            n_emg_columns = self.emgFeatures[0].shape[1]
            new_n_emg_columns = self.emgFeatures.shape[0] * n_emg_columns

            n_acc_rows = self.accFeatures[0].shape[0]
            n_acc_columns = self.accFeatures[0].shape[1]
            new_n_acc_columns = self.accFeatures.shape[0] * n_acc_columns

            n_gyr_rows = self.gyrFeatures[0].shape[0]
            n_gyr_columns = self.gyrFeatures[0].shape[1]
            new_n_gyr_columns = self.gyrFeatures.shape[0] * n_gyr_columns

            n_ori_rows = self.oriFeatures[0].shape[0]
            n_ori_columns = self.oriFeatures[0].shape[1]
            new_n_ori_columns = self.oriFeatures.shape[0] * n_ori_columns

            con_emg_feat = np.reshape(self.emgFeatures, (n_emg_rows, new_n_emg_columns))
            con_acc_feat = np.reshape(self.accFeatures, (n_acc_rows, new_n_acc_columns))
            con_gyr_feat = np.reshape(self.gyrFeatures, (n_gyr_rows, new_n_gyr_columns))
            con_ori_feat = np.reshape(self.oriFeatures, (n_ori_rows, new_n_ori_columns))

            consolidatedFeatureMatrix = np.concatenate((con_emg_feat, con_acc_feat), axis=1)
            consolidatedFeatureMatrix = np.concatenate((consolidatedFeatureMatrix, con_gyr_feat), axis=1)
            consolidatedFeatureMatrix = np.concatenate((consolidatedFeatureMatrix, con_ori_feat), axis=1)
            self.consolidatedFeatureMatrix = consolidatedFeatureMatrix
            self.consolidatedFeatures = True
            return consolidatedFeatureMatrix
        else:
            return None

    def consolidateData(self):
        if self.splitted == True:
            consolidatedDataMatrix = np.concatenate((self.emgList,self.accList,self.gyrList,self.oriList),axis=0)
            self.consolidatedDataMatrix = consolidatedDataMatrix.transpose()
            self.consolidated = True
            return consolidatedDataMatrix
        else:
            return None

    def getConsolidatedFeatureMatrix(self):
        if self.consolidatedFeatures:
            return self.consolidatedFeatureMatrix

    def getConsolidatedDataMatrix(self):
        if self.consolidated:
            return self.consolidatedDataMatrix

    def getRawData(self):
        return self.emg,self.acc,self.gyr,self.ori

    def getData(self):
        if self.splitted is True:
            return self.emg, self.acc, self.gyr, self.ori,self.emgList,self.accList,self.gyrList,self.oriList
        else:
            return self.emg, self.acc, self.gyr, self.ori