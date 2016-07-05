import numpy as np
from scipy import signal

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

    def consolidateData(self):
        if self.splitted == True:
            consolidatedDataMatrix = np.concatenate((self.emgList,self.accList,self.gyrList,self.oriList),axis=0)
            self.consolidatedDataMatrix = consolidatedDataMatrix.transpose()
            self.consolidated = True
            return consolidatedDataMatrix
        else:
            None

    def getConsolidatedDataMatrix(self):
        if self.consolidated == True:
            return self.consolidatedDataMatrix

    def getRawData(self):
        return self.emg,self.acc,self.gyr,self.ori

    def getData(self):
        if self.splitted is True:
            return self.emg, self.acc, self.gyr, self.ori,self.emgList,self.accList,self.gyrList,self.oriList
        else:
            return self.emg, self.acc, self.gyr, self.ori