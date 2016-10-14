from scipy import signal
import numpy as np
from utils import dataprep
import matplotlib.pyplot as plt

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Algos\HMM\Training Data\New"

labels, data, target, labelsdict = dataprep.getTrainingData(rootDir)

emg_t, acc_t, gyr_t, ori_t = data[0].getRawData()

sig = emg_t[:,0]
x = np.arange(sig.shape[0])
resig = signal.resample(sig, 10000)
x2=np.arange(resig.shape[0])

plt.figure(1)

# linear
plt.subplot(221)
plt.plot(x, sig)

# log
plt.subplot(222)
plt.plot(x2, resig)

plt.show()