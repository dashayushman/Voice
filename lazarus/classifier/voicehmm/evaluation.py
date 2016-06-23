from model_generator import generateModels
from utils import dataprep
from sklearn.cross_validation import StratifiedKFold
from utils import dataprep as dp

rootDir = r"C:\Users\Ayushman\Google Drive\TU KAISERSLAUTERN\INFORMARTIK\PROJECT\SigVoice\Work\Algos\HMM\Training Data\New"

if __name__ == "__main__":
    labels, data, target = dataprep.getTrainingData(rootDir)
    skf = StratifiedKFold(target, 5)
    accuracy = []
    for train, test in skf:
        trainingData = dp.prepareTrainingData(train,target,data)
        models,modelLabels = generateModels(trainingData,labels)

        print("%s %s" % (train, test))
        #modelsDict = generateModels()

def vealuateAccuracy(test,data,target,models):
    