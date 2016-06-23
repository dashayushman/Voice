import json
import datasource
from utils import feature_extractor,dataprep

root_dir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Algos\HMM\Training Data\New"

def read_json_file(filepath):
    with open(filepath) as data_file:
        data = json.load(data_file)
        return data

if __name__ == '__main__':
    labels, data, target = dataprep.getTrainingData(root_dir)
    #mfcc_feat, feature_matrix, cwtmatr, d_wavelet_features = feature_extractor.get_features(emg[:, 0])
    #print(feature_matrix)

