from utils import utility as util
from utils import dataprep as dp
import os
import numpy as np
import pandas as pd
from pandas.tools.plotting import parallel_coordinates,andrews_curves
import matplotlib.pyplot as plt


dataRep = r'C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Features\GlobalFeatures'
rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Training Data\individual"
cacheDir = r'G:\PROJECTS\Voice\lazarus\resources\pcord'
plotDir = r'C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Features\GlobalFeaturesPlot'
cacheRefresh = False

def saveParallelCoord(fnale):
    data = pd.read_csv(os.path.join(plotDir, fnale))
    plt.figure()
    #parallel_coordinates(data, 'label')
    andrews_curves(data, 'label')
    plt.show()
    flname = fnale+'.png'
    #plt.savefig(os.path.join(plotDir, flname))


if __name__ == "__main__":
    util.createDir(dataRep)
    labels, data, target, labelsdict, avg_len, user_map, user_list, data_dict = dp.getTrainingData(rootDir)

    # resample also calls consolidate data so there is no need to call consolidate raw data again
    data = dp.resampleTrainingData(data, avg_len)

    # extract features and consolidate features into one single matrix
    if cacheRefresh:
        os.remove(os.path.join(cacheDir, 'featdata.pkl'))
    # extract features and consolidate features into one single matrix
    featData = dp.loadObject(os.path.join(cacheDir, 'featdata.pkl'))
    if featData is None:
        data = dp.extractFeatures(data, None, window=False, rms=False, f_mfcc=False)
        dp.dumpObject(os.path.join(cacheDir, 'featdata.pkl'), data)
    else:
        data = featData
    con = zip(target,data)

    header = ['t_gradient change', 't_zero crossing', 't_window length', 't_rms', 't_mean', 't_variance', 't_ssi',
              't_iemg', 't_peaks', 't_minima', 't_maxima', 'f_mean', 'f_peaks', 'f_total power', 'f_power variance',
              'f_minima', 'f_maxima','label']
    con_emg_0 = []
    con_emg_1 = []
    con_emg_2 = []
    con_emg_3 = []
    con_emg_4 = []
    con_emg_5 = []
    con_emg_6 = []
    con_emg_7 = []

    con_acc_x = []
    con_acc_y = []
    con_acc_z = []

    con_gyr_x = []
    con_gyr_y = []
    con_gyr_z = []

    con_ori_x = []
    con_ori_y = []
    con_ori_z = []
    con_ori_w = []

    for lbl,d in con:
        emg_0_feat, emg_1_feat, emg_2_feat, emg_3_feat, emg_4_feat, emg_5_feat, emg_6_feat, emg_7_feat, acc_x_feat, acc_y_feat, acc_z_feat, gyr_x_feat, gyr_y_feat, gyr_z_feat, ori_x_feat, ori_y_feat, ori_z_feat, ori_w_feat = d.getIndevidualFeatures()

        con_emg_0.append(emg_0_feat)
        con_emg_1.append(emg_1_feat)
        con_emg_2.append(emg_2_feat)
        con_emg_3.append(emg_3_feat)
        con_emg_4.append(emg_4_feat)
        con_emg_5.append(emg_5_feat)
        con_emg_6.append(emg_6_feat)
        con_emg_7.append(emg_7_feat)

        con_acc_x.append(acc_x_feat)
        con_acc_y.append(acc_y_feat)
        con_acc_z.append(acc_z_feat)

        con_gyr_x.append(gyr_x_feat)
        con_gyr_y.append(gyr_y_feat)
        con_gyr_z.append(gyr_z_feat)

        con_ori_x.append(ori_x_feat)
        con_ori_y.append(ori_y_feat)
        con_ori_z.append(ori_z_feat)
        con_ori_w.append(ori_w_feat)

    np.savetxt(os.path.join(dataRep, 'emg_0.csv'), con_emg_0, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'emg_1.csv'), con_emg_1, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'emg_2.csv'), con_emg_2, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'emg_3.csv'), con_emg_3, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'emg_4.csv'), con_emg_4, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'emg_5.csv'), con_emg_5, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'emg_6.csv'), con_emg_6, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'emg_7.csv'), con_emg_7, delimiter=",")

    np.savetxt(os.path.join(dataRep, 'acc_0.csv'), con_acc_x, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'acc_1.csv'), con_acc_y, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'acc_2.csv'), con_acc_z, delimiter=",")

    np.savetxt(os.path.join(dataRep, 'gyr_0.csv'), con_gyr_x, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'gyr_1.csv'), con_gyr_y, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'gyr_2.csv'), con_gyr_z, delimiter=",")

    np.savetxt(os.path.join(dataRep, 'ori_0.csv'), con_ori_x, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'ori_1.csv'), con_ori_y, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'ori_2.csv'), con_ori_z, delimiter=",")
    np.savetxt(os.path.join(dataRep, 'ori_3.csv'), con_ori_w, delimiter=",")

    saveParallelCoord('acc_0.csv')