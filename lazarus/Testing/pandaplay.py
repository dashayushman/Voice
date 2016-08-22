import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import parallel_coordinates,andrews_curves
import os

data = pd.read_csv(r'C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Features\GlobalFeatures\data\acc_0.csv')
col_list = list(data.columns.values)
print(data)
print(list(data.columns.values))
del data['t_zero_crossing']
del data['t_window_length']
del data['t_rms']
del data['t_mean']
del data['t_variance']

del data['t_minima']
del data['t_maxima']
del data['f_mean']
del data['f_peaks']
del data['f_minima']

#data = data.div(data[['t_zero_crossing']])
print(data)