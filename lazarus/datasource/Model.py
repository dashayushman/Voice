class Model:
    'Common base class for all Models'
    def __init__(self,label, emg, acc, gyr, ori, emg_1=None, emg_2=None, emg_3=None, emg_4=None, emg_5=None, emg_6=None, emg_7=None, emg_8=None ):

        self.m_label = label

        self.m_emg = emg
        self.m_acc = acc
        self.m_gyr = gyr
        self.m_ori = ori

        self.m_emg_1 = emg_1
        self.m_emg_2 = emg_2
        self.m_emg_3 = emg_3
        self.m_emg_4 = emg_4
        self.m_emg_5 = emg_5
        self.m_emg_6 = emg_6
        self.m_emg_7 = emg_7
        self.m_emg_8 = emg_8

        '''
        self.m_emg_mfcc_1 = emg_mfcc_1
        self.m_emg_mfcc_2 = emg_mfcc_2
        self.m_emg_mfcc_3 = emg_mfcc_3
        self.m_emg_mfcc_4 = emg_mfcc_4
        self.m_emg_mfcc_5 = emg_mfcc_5
        self.m_emg_mfcc_6 = emg_mfcc_6
        self.m_emg_mfcc_7 = emg_mfcc_7
        self.m_emg_mfcc_8 = emg_mfcc_8
        '''

        self.m_acc_x = emg_8
        self.m_acc_y = emg_8
        self.m_acc_z = emg_8

        self.m_gyr_x = emg_8
        self.m_gyr_y = emg_8
        self.m_gyr_z = emg_8

        self.m_ori_x = emg_8
        self.m_ori_y = emg_8
        self.m_ori_z = emg_8
        self.m_ori_w = emg_8


    def getModels(self):
        return self.m_emg,self.m_acc,self.m_gyr,self.m_ori
