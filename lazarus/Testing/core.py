#import sys
#import csv
#import os
import os.path
import numpy
from sklearn import hmm
from collections import defaultdict
#import utils
import math
import random
#import datetime
import matplotlib.pyplot as plt
#import time
import warnings
from features import mfcc
#from features import logfbank
#from features import fbank
import mlpy
from collections import Counter
import copy
#import statistics

#from sklearn.decomposition import PCA
from sklearn.lda import LDA

def show_data(data):
    fig = plt.figure()
    for key in data.keys():
        for seq in data[key]:
            plt.clf()
            ax = fig.add_subplot(111)
            #ax.set_aspect(1)
            for sig_index in range(len(seq[0])):
                sig = [x[sig_index] for x in seq]
                #sig2= [x[1] for x in seq]
                plt.plot(range(len(seq)),sig)
            #plt.plot(range(len(seq)),sig2)
            plt.title(key)
            plt.show()

def normalize_mean_std_training(dataset):
    transformed_dataset = defaultdict(list)
    means = []
    stds = []
    for k in range(len(dataset.keys())):
        key = dataset.keys()[k]
        sequences = dataset[key]
        for sequence in sequences:
            #print "length of sequence before: ", len(sequence)
            s1 = numpy.array([x[0] for x in sequence])
            s2 = numpy.array([x[1] for x in sequence])
            s3 = numpy.array([x[2] for x in sequence])
            s4 = numpy.array([x[3] for x in sequence])
                            
            m1 = numpy.mean(s1)
            m2 = numpy.mean(s2)
            m3 = numpy.mean(s3)
            m4 = numpy.mean(s4)
            #print m1
            means.append([1,m1])
            means.append([2,m2])
            means.append([3,m3])
            means.append([4,m4])
            
            std1 = numpy.std(s1)
            std2 = numpy.std(s2)
            std3 = numpy.std(s3)
            std4 = numpy.std(s4)
            
            stds.append([1,std1])
            stds.append([2,std2])
            stds.append([3,std3])
            stds.append([4,std4])
            
            s1_norm = numpy.array([float(x-m1)/std1 for x in s1])
            s2_norm = numpy.array([float(x-m2)/std2 for x in s2])
            s3_norm = numpy.array([float(x-m3)/std3 for x in s3])
            s4_norm = numpy.array([float(x-m4)/std4 for x in s4])
            
            fod1 = derivative(s1)
            fod2 = derivative(s2)
            fod3 = derivative(s3)
            fod4 = derivative(s4)
            """
            """
            fod1 = derivative(s1_norm)
            fod2 = derivative(s2_norm)
            fod3 = derivative(s3_norm)
            fod4 = derivative(s4_norm)
            
            sod1 = derivative(fod1)
            sod2 = derivative(fod2)
            sod3 = derivative(fod3)
            sod4 = derivative(fod4)
            
            seq_norm = []
            for i in range(len(sequence)-2):
                temp = numpy.array([s1_norm[i],s2_norm[i],s3_norm[i],s4_norm[i]  ,
                                    fod1[i] , fod2[i] , fod3[i] , fod4[i]        ,
                                    sod1[i] , sod2[i] , sod3[i] , sod4[i]])
                #temp = numpy.array([s1_norm[i],s2_norm[i],s3_norm[i],s4_norm[i]])
                seq_norm.append(temp)
            transformed_dataset[key].append(numpy.array(seq_norm))
            #print "length of sequence after: ", len(seq_norm)    
    
    
    means_arr = []
    stds_arr = []
    for i in range(4):
        mean = numpy.average([x[1] for x in means if x[0]==i+1])
        #print [x for x in means if x[0]==i+1]
        std = numpy.average([x[1] for x in stds if x[0]==i+1])
        means_arr.append(mean)
        stds_arr.append(std)
        #print "average mean: ", mean
        #print "average std: ", std
    
    return transformed_dataset, means_arr, stds_arr

def normalize_mean_std_validation(dataset, means, stds):
    
    transformed_dataset = defaultdict(list)
    for k in range(len(dataset.keys())):
        key = dataset.keys()[k]
        sequences = dataset[key]
        for sequence in sequences:
            #print "length of sequence before: ", len(sequence)
            s1 = numpy.array([x[0] for x in sequence])
            s2 = numpy.array([x[1] for x in sequence])
            s3 = numpy.array([x[2] for x in sequence])
            s4 = numpy.array([x[3] for x in sequence])
            
            s1_norm = numpy.array([float(x-means[0])/stds[0] for x in s1])
            s2_norm = numpy.array([float(x-means[1])/stds[1] for x in s2])
            s3_norm = numpy.array([float(x-means[2])/stds[2] for x in s3])
            s4_norm = numpy.array([float(x-means[3])/stds[3] for x in s4])
            """
            fod1 = derivative(s1)
            fod2 = derivative(s2)
            fod3 = derivative(s3)
            fod4 = derivative(s4)
            """
            
            fod1 = derivative(s1_norm)
            fod2 = derivative(s2_norm)
            fod3 = derivative(s3_norm)
            fod4 = derivative(s4_norm)
            
            sod1 = derivative(fod1)
            sod2 = derivative(fod2)
            sod3 = derivative(fod3)
            sod4 = derivative(fod4)
            
            seq_norm = []
            for i in range(len(sequence)-2):
                #temp = numpy.array([s1_norm[i],s2_norm[i],s3_norm[i],s4_norm[i]])
                temp = numpy.array([s1_norm[i],s2_norm[i],s3_norm[i],s4_norm[i] ,
                                    fod1[i] , fod2[i] , fod3[i] , fod4[i]       ,
                                    sod1[i] , sod2[i] , sod3[i] , sod4[i]])
                seq_norm.append(temp)
            transformed_dataset[key].append(numpy.array(seq_norm))
            #print "length of sequence after: ", len(seq_norm)    
    
    return transformed_dataset

def derivative(signal):
    temp = []
    for i in range(len(signal)-1):
        temp.append(signal[i+1]-signal[i])
    return temp

def read_result_file(filename, tries = 0):
    #print "updating current file"

    """
    try: 
        f = open(filename,'r')
    except:
        #print filename
        filepatharray = filename.split('/')[:-1]
        filepath = '/'
        for el in filepatharray:
            filepath = filepath+el+'/'
        #print filepath
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        f = open(filename,'w')
        f.write('0 0 0\n') #create a sample file with the first line that can be applied to sum
        f.close
        f = open(filename,'r')
    """
    create_file(filename)
    index = []
    data = []
    with open(filename,'r') as f:
      
        #f = open(filename,'r')
        
        for line in f:
            #print line
            if line.startswith('#labels'):
                index = line.split(' ')[1:-1]
                #print "index: ", index
            if not(line.startswith('#')):
                #print [x for x in line.split(' ')]
                line_int = [int(x) for x in line.split(' ')[:-1]]
                data.append(line_int)
                #print line_int
        f.close
    #print data
    checksum = [sum(i) for i in data]
    check = True
    #tries = 0
    """
    for i in range(len(checksum)):
        if checksum[0] != checksum[i]:
            #print "try: ",tries
            #some process is currently writing inside the file -> try again
            tries += 1  #the columns dont match, try another time
            time.sleep(1)   #wait so that the other processes can finish writing
            if tries<5:
                res = read_result_file(filename, tries = tries)
            else:
                os.remove(filename)
                create_file(filename)
                res = read_result_file(filename, tries = 0)
                return res
    """            
    return [index, data]

def create_file(filename):
    #print filename
    filepatharray = filename.split('/')[:-1]
    #print filepatharray
    #filepath = '/'
    filepath = ''
    for el in filepatharray:
        filepath = filepath+el+'/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if not os.path.isfile(filename):
        #print "inside"
        f = open(filename,'w')
        f.write('0 0 0\n')
        #f.flush()
        f.close   
        
def evaluate_dwt(train, test):
    results = []
    for i in range(len(test)):
        #dist = [(-1,999999999),(-1,999999999),(-1,999999999),(-1,999999999)]
        
        seq_val = test[i][1]
        min_dist = []
        min_class = []
        n_sensors = len(seq_val[0])
        for k in range(n_sensors):
            min_dist.append(99999999999999)
            min_class.append(-1)
        #min_dist = [999999999,999999999,999999999,999999999]
        #min_class = [-1,-1,-1,-1]
        
        s1 = []
        for index1 in range(len(seq_val[0])):
            s1.append([x[index1] for x in seq_val])
        #s1 = [x[0] for x in seq_val]
        #s2 = [x[1] for x in seq_val]
        #s3 = [x[2] for x in seq_val]
        #s4 = [x[3] for x in seq_val]
        #print s1
        #min_dist=9999999999999999
        #min_class = -1
        for j in range(len(train)):
        #for j in range(len(fulldata)):
            key = train[j][0]
            seq_gt = train[j][1]
            #key = fulldata[j][0]
            #seq_gt = fulldata[j][1]
            s2 = []
            for index2 in range(len(seq_gt[0])):
                s2.append([x[index2] for x in seq_gt])
            #gts1 = [x[0] for x in seq_gt]
            #gts2 = [x[1] for x in seq_gt]
            #gts3 = [x[2] for x in seq_gt]
            #gts4 = [x[3] for x in seq_gt]
            #print s2
            tdist = []
            for sensor in range(len(s1)):
                temp_dist, cost, path = mlpy.dtw_std(s1[sensor], s2[sensor], dist_only=False, squared=False)
                tdist.append(temp_dist)
            
            #temp_dist1, cost, path = mlpy.dtw_std(s1, gts1, dist_only=False, squared=False)
            #temp_dist2, cost, path = mlpy.dtw_std(s2, gts2, dist_only=False, squared=False)
            #temp_dist3, cost, path = mlpy.dtw_std(s3, gts3, dist_only=False, squared=False)
            #temp_dist4, cost, path = mlpy.dtw_std(s4, gts4, dist_only=False, squared=False)
            #tdist = [temp_dist1,temp_dist2,temp_dist3,temp_dist4]
            for sensor in range(len(s1)):
                if tdist[sensor]<min_dist[sensor]:
                    min_dist[sensor] = tdist[sensor]
                    min_class[sensor] = key
        
        #if validation[i][0]==min_class:
        #    correct+=1
        #total+=1
        #print "gt: ",validation[i][0], "    pred: ",min_class ,"      distance: ", min_dist
        for sensor in range(len(s1)):
            #print "sensor : ",sensor, "   gt: ",validation[i][0],"        \t pred: ",min_class[sensor],"      \t dist: ", '%.6f' % min_dist[sensor] 
            temp = Counter(min_class)
        #print " "
        most = temp.most_common(1)
        #print most[0][0]
        results.append((test[i][0],most[0][0]))
        #print most
        """
        if most[0][0]==test[i][0]:
            correct += 1
            correct_cycle += 1
        total+=1
        total_cycle += 1
        """
        
            #print ""
    return results

def split_dataset_leave_p_out_to_arrays(dataset, p=1):
    training = []
    validation = []
    for key in dataset.keys():
        training_temp = []
        sequences = dataset[key][:]
        for v in range(p):
            val_index = random.randint(0, len(sequences)-1)
            val_seq = sequences[val_index]
            validation.append((key,val_seq))
            training_temp = sequences.pop(val_index)
            #print training_temp
        for i in range(len(sequences)):
            #if i != val_index:
            training.append((key,sequences[i]))
    
    return training, validation

def calculate_lda(dataset, n_lda_features=3):
    warnings.filterwarnings("ignore")
    lda = LDA(n_components = n_lda_features)
    
    samplerate = 20
    #winlen = 0.5 #in seconds
    #winstep=0.25 #in seconds
        
    temp_data_data = []
    temp_data_class = []
    for i in range(len(dataset.keys())):
        key = dataset.keys()[i]
        #print key
        sequences = dataset[key]
        for sequence in sequences:

            #print mfcc_feat1
            mfcc_feat = []
            for k in range(len(sequence)):
                #print "temp class name: ",i," , key: ", key
                features = sequence[k]
                temp_data_data.append(features)
                temp_data_class.append(i)
                #print features
                #print "using ",len(features)," features per timestep"
                #mfcc_feat.append(numpy.array(features))
            #print sequence
    transformer = lda.fit(temp_data_data , temp_data_class)
    
    return transformer

def apply_cepstrum(dataset, samplerate=20, cepstrum_ws=0.5, cepstrum_wo=0.05):
    warnings.filterwarnings("ignore")
    transformed_dataset = defaultdict(list)
    winlen = cepstrum_ws
    winstep = cepstrum_wo
    #samplerate = 20
    #winlen = 0.5 #in seconds
    #winstep=0.25 #in seconds
    #winstep=0.25 #in seconds
    temp_data_data = []
    temp_data_class = []
    for k in range(len(dataset.keys())):
        key = dataset.keys()[k]
        #print key
        sequences = dataset[key]
        for sequence in sequences:

            s1 = numpy.array([x[0] for x in sequence])
            s2 = numpy.array([x[1] for x in sequence])
            s3 = numpy.array([x[2] for x in sequence])
            s4 = numpy.array([x[3] for x in sequence])

            mfcc_feat1 = mfcc(s1,samplerate=samplerate,winlen=winlen,winstep=winstep)[1:]
            mfcc_feat2 = mfcc(s2,samplerate=samplerate,winlen=winlen,winstep=winstep)[1:]
            mfcc_feat3 = mfcc(s3,samplerate=samplerate,winlen=winlen,winstep=winstep)[1:]
            mfcc_feat4 = mfcc(s4,samplerate=samplerate,winlen=winlen,winstep=winstep)[1:]
            
            mfcc_feat = []
            for i in range(len(mfcc_feat1)):
                #print "mfcc_feat1 - ",i," : ",mfcc_feat1[i][0]
                features_arr = [mfcc_feat1[i],mfcc_feat2[i],mfcc_feat3[i],mfcc_feat4[i]]
                features = [item for sublist in features_arr for item in sublist]
                
                for temp in range(len(features)):
                    if math.isnan(features[temp]):
                        features[temp] = 0
                    if math.isinf(features[temp]):
                        features[temp] = 0

                mfcc_feat.append(numpy.array(features))

   
    return transformed_dataset
        
def apply_lda(dataset, transformer):
    transformed_dataset = defaultdict(list)
    warnings.filterwarnings("ignore")
    samplerate = 20
    winlen = 0.5 #in seconds
    winstep=0.25 #in seconds
        
    temp_data_data = []
    temp_data_class = []
    for i in range(len(dataset.keys())):
        key = dataset.keys()[i]
        #print key
        sequences = dataset[key]
        for sequence in sequences:
            """
            s1 = numpy.array([x[0] for x in sequence])
            #print s1
            s2 = numpy.array([x[1] for x in sequence])
            s3 = numpy.array([x[2] for x in sequence])
            s4 = numpy.array([x[3] for x in sequence])
            mfcc_feat1 = mfcc(s1,samplerate=samplerate,winlen=winlen,winstep=winstep)
            mfcc_feat2 = mfcc(s2,samplerate=samplerate,winlen=winlen,winstep=winstep)
            mfcc_feat3 = mfcc(s3,samplerate=samplerate,winlen=winlen,winstep=winstep)
            mfcc_feat4 = mfcc(s4,samplerate=samplerate,winlen=winlen,winstep=winstep)
            #print mfcc_feat1
            """
            mfcc_feat = []
            for k in range(len(sequence)):
                
                features = sequence[k]
                
                #print features
                #print "using ",len(features)," features per timestep"
                #mfcc_feat.append(numpy.array(features))
                mfcc_feat.append(numpy.array(transformer.transform(features)[0]))
            #print sequence
            transformed_dataset[key].append(numpy.array(mfcc_feat))
    
    """
    for i in range(len(transformed_dataset.keys())):
        key = transformed_dataset.keys()[i]
        print key
        sequences = transformed_dataset[key]
        for sequence in sequences:
            print sequence
    """       
    return transformed_dataset



def read_data_dict(csv, sensors='all'):
    #print csv
    file = open(csv,'r')
    arrays = []
    inputsfinal = numpy.array([])
    outputsfinal = numpy.array([]).reshape(0,10)
    #dataset = numpy.array([]).reshape(0,2)
    dataset = defaultdict(list)
    #dataset = defaultdict(lambda:defaultdict(list))
    
    test = numpy.array([]).reshape(0,2)

    #print no_output
    
    #exp_type = int(csv.split('\\')[-1][1])
    exp_type = int(csv.split('\\')[-1].split('_')[2][1])
    #print "experiment type:", exp_type
    
    for el in file:
        
        temp = el.split(' ')
        #output = one_to_many(temp[0])
        input = []
        gesture_name = str(temp[0])
        if gesture_name.startswith('gesture_13') or gesture_name.startswith('gesture_14') or gesture_name.startswith('gesture_15'):
             if exp_type==1 or exp_type==3:
                 gesture_name = gesture_name+"_air"
             if exp_type==2 or exp_type==4 or exp_type==5:
                 #print csv
                 gesture_name = gesture_name+"_surface"    
        for element in temp[1:]:
            elements = element.split(',') #has length 4 (4 sensors)
            if elements[-1][-1] == '\n':
                #print elements
                #print "found end"
                elements[-1] = elements[-1][:-1]
                #print elements
            
            a = [float(x) for x in elements]
            if sensors=='1':
                a = [float(elements[0])]
            #a = numpy.array(a)
            #print a.shape
            #print input.shape
            """
            if input.shape == (0,):
                input = a 
            else:
                #print input
                #print a
                input = numpy.concatenate((input,a),axis=0)
            print input.shape
            """
            input.append(a)
        #print gesture_name,": ",input
        input = numpy.array(input)
        #print temp[0]
        #dataset[str(temp[0])].append(input)
        dataset[gesture_name].append(input)
        #dataset[str(temp[0])] += input
    #print dataset[dataset.keys()[0]]
    return dataset

def read_new_dataset(path = '', data = 'numbers', leave_out = 'ch',interaction='finger', type='raw'):
    if type == 'raw':
        path = 'C:\\Google Drive\\Master Thesis\\Data\\new_data_converted\\'
    if type == 'mod':
        path = 'C:\\Google Drive\\Master Thesis\\Data\\new_data_converted_mod\\'
    files_train = []
    files_validate = []
    for filename in os.listdir(path+data+'\\') :
        #print filename
        if filename.endswith(data+'_hmm.csv'):
            
            subject = filename.split('_')[3]
            interact = filename.split('_')[0]
            if subject[:2] != leave_out[:2] and interaction==interact:
            #print "Reading file: ", filename
                print "Training on: ", filename
                
                files_train.append(read_data_dict(path+'\\'+data+'\\'+filename))
            if subject[:2] == leave_out[:2] and interaction==interact:
                print "Validating on: ", filename
                
                files_validate.append(read_data_dict(path+'\\'+data+'\\'+filename))
    training = combine_datasets(files_train)  
    validating = combine_datasets(files_validate)
    #print training
    #print validating
    return training,validating

def results_to_matrix(results):
    index_names = [x[0] for x in results]
    index_names = sorted(set(index_names))
    matrix = numpy.zeros((len(index_names),len(index_names)))
    for i in range(len(results)):
        gt = results[i][0]
        pred = results[i][1]
        gti = index_names.index(gt)
        predi = index_names.index(pred)
        matrix[gti][predi] += 1
    return [index_names, matrix]
    
def write_result_file(name, matrix, old, config = ""):
    #print name
    index = matrix[0]
    mat_new = matrix[1]
    #print "index: ",index
    index2 = old[0]
    mat_old = old[1]
    if not(len(mat_old)==1):    #this is not the first iteration
        for i in range(len(mat_old)):   #number of elements
            #indx = index2[i] #element name
            for j in range(len(mat_old[0])):    # for every element in row
                mat_new[i][j] += mat_old[i][j]
        #everything is combined and can be saved now
    iteration = sum(mat_new[0])
    #write the data to file
    filename = name
    f = open(filename,"w")
    
    #first write the config
    for el in config:
        if el.startswith("#number of iterations"):
            f.write("#number of iterations:\t" +str(int(iteration))+"\n")
        else:
            f.write(el + "\n")
    f.write("#labels: ")
    for i in range(len(index)):
        f.write(index[i]+" ")
    f.write("\n")
    for i in range(len(mat_new)):
        for j in range(len(mat_new[i])):
            f.write(str(int(mat_new[i][j]))+" ")
        f.write("\n")
    f.close()
    
def dict_to_arr(dict):
    matrix = dict[1]
    names_dict = dict[0]
    names = [0]*len(names_dict.keys())
    for index in range(len(names)):
        key = names_dict.keys()[index]
        pos = names_dict[key]
        names[pos] = key
    if type=="words" or type=="all":
        names[-3]="sentence_1"
        names[-2]="sentence_2"
        names[-1]="sentence_3"
    conf_arr = dict[1]
    return [names, conf_arr]


def build_hmms(n_states, data):

    hmms = {}
    
    for key in data.keys():
        #print "training on class: ", key ,"with number of sequences: ", len(data[key])

        try:
            #try if training works (sometimes some sequences result in errors)
            temp_data = numpy.copy(data[key])
            temp_hmm = hmm.GaussianHMM(n_states, "full")
            temp_hmm.fit(temp_data)
            hmms[key]=temp_hmm
            #print "training class ", key, " on ", len(temp_data), " of ", len(temp_data), " sequences"
        except:
            #some sequences made an error during training
            sequences = data[key]
            successful = []
            for i in range(len(sequences)):
                try:
                    print "training sequence of ", key," of length : ", len(sequences[i])
                    temp_data = copy.copy(successful)
                    temp_data.append(sequences[i])
                    
                    temp_hmm = hmm.GaussianHMM(n_states, "full")
                    temp_hmm.fit(temp_data)

                    successful.append(sequences[i])
                except:
                    #do not use this sequence as it produced an error
                    #print "not using sequence ", i
                    pass
                
            temp_hmm = hmm.GaussianHMM(n_states, "full")
            temp_hmm.fit(successful)
            hmms[key] = temp_hmm
            #print "training class ", key, " on ", len(successful), " of ", len(sequences), " sequences"

    return hmms
"""
def build_hmms_variable(n_states, data):
    #n_classes = len(data.keys()) 
    hmms = {}
    for key in data.keys():
        if key == '0' or key == '1' or key == '2':
            hmms[key] = hmm.GaussianHMM(3, "full")
            hmms[key].fit(data[key])
        if key == '3' or key == '4'  or key == '7' or key == '6'  or key == '9' or key == '8' :
            hmms[key] = hmm.GaussianHMM(4, "full")
            hmms[key].fit(data[key])
        if key == '5' :
            hmms[key] = hmm.GaussianHMM(5, "full")
            hmms[key].fit(data[key])
    return hmms
"""

    
def evaluate_random_hmm(n_states, train, val):
    
    n_classes = len(train.keys()) 
    validation = []

    hmms = build_hmms(n_states, train)
    
    correct = 0
    total   = 0
    
    dict_names_index = defaultdict(list)
    confusion_matrix = numpy.zeros((n_classes, n_classes))
    
    for i in range(len(train.keys())):
        key = train.keys()[i]
        dict_names_index[key] = i
        
    for key in val.keys():
        sequences = val[key][:]
        for i in range(len(sequences)):
            
            val_seq = sequences[i]
            validation.append([key,val_seq])
    
    for i in range(len(validation)):    #for every validation element
        try:
            key = validation[i][0]
            eval_ob = validation[i]
            eval_seq = eval_ob[1]
            gt_class = eval_ob[0]
            #print "class: ", gt_class
            best = -1

            bestscore = -100000000000
            
            for j in range(len(hmms.keys())):   #iterate over all the hmms

                hmm_name = hmms.keys()[j]
                hmm_temp = hmms[hmm_name]   # the hmm to be evaluated
                score = hmm_temp.score(eval_seq)

                #print "gt: ", gt_class,"\t class: ", hmms.keys()[j], "\t score: ", score

                if score > bestscore:

                    best = hmm_name
                    bestscore = score
            if str(best) == gt_class:
                correct += 1

            ind_x = dict_names_index[gt_class]
            ind_y = dict_names_index[str(best)]
            confusion_matrix[ind_x,ind_y] += 1
            
            total += 1
        except:
            pass
        #print "best: ",best, "gt: ", gt_class

    return [float(correct)/total , [dict_names_index,confusion_matrix]]

def combine_datasets(data_dict_list):
    
    final = defaultdict(list)
    for dict in data_dict_list:
        for key in dict:
            for elem in dict[key]:
                #print key," ",len(elem)
                final[key].append(elem)
    return final

def k_fold_cross_validation_hmm(n_states, train, validate,type=type):
    temp = []
    conf = []

    worked = False
    while not(worked):
        try:

            result = (evaluate_random_hmm(n_states, train, validate))
            worked = True

        except:
            #this should not happen anymore
            pass
            #print "Cholesky error, restarting current singular k-fold iteration"
    #print "precision:",result[0]
    confusion = result[1]
    conf_names = confusion[0]
    conf_matrix = confusion[1]

    temp.append(result[0])
    conf.append(result[1])
    
    combined = combine_confusion_matrices(conf,type=type)
    return combined
        
    return [conf_names, conf_matrix]

def combine_confusion_matrices(matrix_array, type="numbers"):
    
    #this is where the names have to be ordered before visualization
    
    dict_names_index = {}
    index_names = []
    for key in matrix_array[0][0].keys():
        #print key
        index_names.append(key)

    #print index_names
    if type=="numbers" or type=="gestures" or type=='gestures+norepeat':
        index_names.sort()
        
    if type=="words":
        index_names.sort()
        #print index_names
        index_names.remove('pack_my_box_with_five_dozen_liquor_jugs')
        index_names.remove('a_quick_brown_fox_jumps_over_the_lazy_dog')
        index_names.remove('we_quickly_explained_that_many_big_jobs_involve_few_hazards')
        index_names.append('pack_my_box_with_five_dozen_liquor_jugs')
        index_names.append('a_quick_brown_fox_jumps_over_the_lazy_dog')
        index_names.append('we_quickly_explained_that_many_big_jobs_involve_few_hazards')
    
    if type=="numbers+words":
        index_names_temp = []
        #append numbers
        for i in range(10):
            index_names_temp.append(str(i))

        words=[]
        for key in matrix_array[0][0].keys():
            if len(key)>1 and key[:7]!="gesture" and len(key)<20 and key!="signature": #words
                #print key
                words.append(key)    
                #words.remove('pack_my_box_with_five_dozen_liquor_jugs')
                #words.remove('a_quick_brown_fox_jumps_over_the_lazy_dog')
                #words.remove('we_quickly_explained_that_many_big_jobs_involve_few_hazards')
        words.sort()
        words.append('pack_my_box_with_five_dozen_liquor_jugs')
        words.append('a_quick_brown_fox_jumps_over_the_lazy_dog')
        words.append('we_quickly_explained_that_many_big_jobs_involve_few_hazards')
            
        index_names = index_names_temp+words


    if type=="all":
        index_names_temp = []
        #append numbers
        for i in range(10):
            index_names_temp.append(str(i))
        index_names_temp.append("signature")
        #append_gestures
        for i in range(1,10):
            index_names_temp.append("gesture_0"+str(i))
        for i in range(10,16):
            index_names_temp.append("gesture_"+str(i))            
        #append words+sentences
        words=[]
        for key in matrix_array[0][0].keys():
            if len(key)>1 and key[:7]!="gesture" and len(key)<20 and key!="signature": #words
                #print key
                words.append(key)    
                #words.remove('pack_my_box_with_five_dozen_liquor_jugs')
                #words.remove('a_quick_brown_fox_jumps_over_the_lazy_dog')
                #words.remove('we_quickly_explained_that_many_big_jobs_involve_few_hazards')
        words.sort()
        words.append('pack_my_box_with_five_dozen_liquor_jugs')
        words.append('a_quick_brown_fox_jumps_over_the_lazy_dog')
        words.append('we_quickly_explained_that_many_big_jobs_involve_few_hazards')
            
        index_names = index_names_temp+words
        
    #print index_names

    #print index_names
    for i in range(len(index_names)):
        dict_names_index[index_names[i]] = i
    #just use the sorting of the first confusion matrix.
    #alternatively, this could also be sorted differently
    #dict_names_index = matrix_array[0][0]
    confusion_matrix = numpy.zeros(matrix_array[0][1].shape)
    
    #we need to iterate over this as it is not clear that the labels 
    #are sorted the same way for all the confusion matrices
    for i in range(len(matrix_array)):
        index = matrix_array[i][0]
        matrix = matrix_array[i][1]
        for x in index.keys():
            ind_x = dict_names_index[x]
            #print "x: ",x ," ", index[x],"  ind_x: ", ind_x
            for y in index.keys():
                ind_y = dict_names_index[y]

                confusion_matrix[ind_x,ind_y] += matrix[index[x]][index[y]]
    
    return [dict_names_index,confusion_matrix]