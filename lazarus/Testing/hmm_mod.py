import sys
import csv
import os
import numpy
from sklearn import hmm
from collections import defaultdict
import math
import random
import datetime
import matplotlib.pyplot as plt
import time
import core as core
import warnings


def evaluate(n_states = 5, experiment='all',data_type='raw', data='numbers', subjects='both', iterations=100, val_number=1, interaction = 'finger'):
    

    #printing config in console
    print '######################################################'
    
    print "#number of states: \t ", n_states
    print "#number of iterations: \t ", iterations
    
    print "#Data type: \t \t ", data_type
    print "#Dataset: \t \t ", data
    print "#using experiment(s): \t ", interaction
    print "#excluding subject(s): \t ", subjects
    print "#eval number: \t \t ", val_number
    
    print '######################################################'
    print ""
    config = []
    config.append("#number of states: \t"+ str(n_states))
    config.append("#number of iterations: \t"+ str(iterations))
    config.append("#Data type: \t \t"+str(data_type))
    config.append("#Dataset: \t \t"+str(data))
    config.append("#using experiment(s): \t"+ str(interaction))
    config.append("#excluding subject(s): \t"+ str(subjects))
    
    out = 'results/hmm/mod/'+data+'/'+str(interaction)+'/'+data_type+"_"+data+"_experiments="+str(experiment)+"_subjects="+str(subjects)+"_nstates="+str(n_states)
    
    path = 'C:\\Google Drive\\Master Thesis\\Data\\ground_truth\\for_hmm\\'+data_type+'\\new_data\\'+data+'\\'
    training_mod, validating_mod = core.read_new_dataset(path = path, data=data, leave_out = subjects, type='mod')
    #core.show_data(training_mod)
    
    training_norm, means, stds = core.normalize_mean_std_training(training_mod)
    validating_norm = core.normalize_mean_std_validation(validating_mod, means, stds)
    
    training = training_norm
    validating = validating_norm
    
    #core.show_data(validating)
    
    #core.show_data(validating_norm)
    #print len(validating[validating.keys()[0]])
    #print training
    #data_dict = combine_datasets(files)  
    #data_cepstrum = core.apply_cepstrum(data_dict)     
    #transformer = core.calculate_lda(data_cepstrum, n_lda_features=3)     
    #data_dict = core.apply_lda(data_cepstrum, transformer, n_lda_features=3)
    #core.show_data(data_dict)
    #print "after"
    print ""
    #print image_name.split('.')[0]
    filename = out+'.txt'

    old = core.read_result_file(filename)  # get the results from older iterations saved in a text file
    iterations_done = sum(old[1][0])   #already did this many iterations
    res = None

    while iterations_done+1 <= iterations:

        print "starting iteration: ", iterations_done+1
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print "starting iteration ", iterations_done+1, " at ",st
        res = core.k_fold_cross_validation_hmm(n_states, training, validating, type=data)    #res[0] = index names, res[1] = matrix
        res = core.dict_to_arr(res)

        old = core.read_result_file(filename)
        iterations_done = sum(old[1][0])

        if iterations_done<iterations:
            
            core.write_result_file(filename, res, old, config = config)
        iterations_done = sum(old[1][0])
    
   
def main(argv):
    warnings.filterwarnings("ignore")

    n_states = 2
    states = n_states
    iterations = 100 #minimum of evaluated examples per class, n.b. the validation set has already multiple sequences per class

    data_type = 'raw'
    
    data = 'numbers'
    #data = 'gestures'

    interaction= 'finger'
    #interaction= 'pen'
    print "Starting evaluation"

    for lo in ['pa','jac','ian','ch']:  #timl is excluded as he is left-handed
                                        #lo indicates which person is used for validation
        for states in range(2,8):
            #print "Evaluating with ", states, " hidden states"
            evaluate(n_states = states, iterations=iterations, data_type=data_type, data=data,subjects=lo, interaction=interaction)
    
    print "finished evaluating everything"
    
    return

if __name__ == "__main__":
    main(sys.argv[1:])