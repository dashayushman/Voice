import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import parallel_coordinates,andrews_curves
import os

rootDir = r'C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Features\GlobalFeatures\data'
plotDir = r'C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Features\GlobalFeatures\plots'
def plot_confusion_matrix(cm, labels,title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def visualizeFeatures(rootdir,plotdir,fname,save=True,plot=True,andrews=True,pc=True):
    data = pd.read_csv(os.path.join(rootdir, fname))

    '''
    del data['t_zero_crossing']
    del data['t_window_length']
    del data['t_rms']
    del data['t_mean']
    del data['t_variance']
    del data['t_peaks']

    del data['t_minima']
    del data['t_maxima']
    del data['f_mean']
    del data['f_peaks']
    del data['f_minima']
    '''
    plt.figure()
    plt.title(fname)
    if pc:
        andrews_curves(data, 'label')
        #parallel_coordinates(data, 'label')
    #if andrews:
    #    andrews_curves(data, 'label')
    if save:
        plt.savefig(os.path.join(plotdir, fname + '_ac_.png'))
    if plot:
        plt.show()

    #plt.savefig(os.path.join(plotDir, flname))

def getSensorFeatureFileList(rootDir):
    filewalk = os.walk(rootDir)
    file_list = []
    for fls in filewalk:
        file_list = fls[2]
        break
    return file_list


if __name__ == "__main__":
    file_list = getSensorFeatureFileList(rootDir)
    for f in file_list:
        visualizeFeatures(rootDir,plotDir,f,plot=False)
