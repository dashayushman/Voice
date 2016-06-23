import numpy as np

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

    def separateRawData(self):
        if self.emg is not None:
            self.emg_1 = np.array(self.emg[:,0])
            self.emg_2 = np.array(self.emg[:,1])
            self.emg_3 = np.array(self.emg[:,2])
            self.emg_4 = np.array(self.emg[:,3])
            self.emg_5 = np.array(self.emg[:,4])
            self.emg_6 = np.array(self.emg[:,5])
            self.emg_7 = np.array(self.emg[:, 6])
            self.emg_8 = np.array(self.emg[:, 7])

        if self.acc is not None:
            self.acc_x = np.array(self.acc[:, 0])
            self.acc_y = np.array(self.acc[:, 1])
            self.acc_z = np.array(self.acc[:, 2])

        if self.gyr is not None:
            self.gyr_x = np.array(self.gyr[:, 0])
            self.gyr_y = np.array(self.gyr[:, 1])
            self.gyr_z = np.array(self.gyr[:, 2])

        if self.ori is not None:
            self.ori_x = np.array(self.ori[:, 0])
            self.ori_y = np.array(self.ori[:, 1])
            self.ori_z = np.array(self.ori[:, 2])
            self.ori_w = np.array(self.ori[:, 3])

        self.splitted = True


    def getRawData(self):
        return self.emg,self.acc,self.gyr,self.ori

    def getData(self):
        if self.splitted is True:
            return self.emg, self.acc, self.gyr, self.ori, self.emg_1, self.emg_2, self.emg_3, self.emg_4, self.emg_5, self.emg_6, self.emg_7, self.emg_8, self.acc_x, self.acc_y, self.acc_z, self.gyr_x, self.gyr_y, self.gyr_z, self.oci_x, self.oci_y, self.oci_z, self.oci_w
        else:
            return self.emg, self.acc, self.gyr, self.ori