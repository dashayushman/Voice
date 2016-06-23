import json
import numpy as np
import datasource
from utils import feature_extractor
import os

def read_json_file(filepath):
    with open(filepath) as data_file:
        data = json.load(data_file)
        return data

if __name__ == '__main__':
    data = read_json_file('A_air_Amit.json')
    emg = data['emg']['data']
    emg = np.array(emg)
    mfcc_feat, feature_matrix, cwtmatr, d_wavelet_features = feature_extractor.get_features(emg[:, 0])
    print(feature_matrix)

