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
import core

def read_data_dict(csv):
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
    
    exp_type = int(csv.split('\\')[-1][1])
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
                 gesture_name = gesture_name+"_surface"    
        for element in temp[1:]:
            elements = element.split(',')
            if elements[-1][-1] == '\n':
                #print elements
                #print "found end"
                elements[-1] = elements[-1][:-1]
                #print elements
            a = [float(x) for x in elements]
            a = numpy.array(a)
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
        input = numpy.array(input)
        #print temp[0]
        #dataset[str(temp[0])].append(input)
        dataset[gesture_name].append(input)
        #dataset[str(temp[0])] += input
    return dataset

def read_data(csv):
    print(csv)
    file = open(csv,'r')
    arrays = []
    inputsfinal = numpy.array([])
    outputsfinal = numpy.array([]).reshape(0,10)
    #dataset = numpy.array([]).reshape(0,2)
    dataset = []
    test = numpy.array([]).reshape(0,2)
    no_output = numpy.zeros(11) #10 number + 1 class no_outpout
    no_output[10] = 1
    #print no_output
    
    for el in file:
        
        temp = el.split(' ')
        output = one_to_many(temp[0])
        input = []
        for element in temp[1:]:
            elements = element.split(',')
            if elements[-1][-1] == '\n':
                #print elements
                #print "found end"
                elements[-1] = elements[-1][:-1]
                #print elements
            a = [float(x) for x in elements]
            a = numpy.array(a)
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
        input = numpy.array(input)
        dataset.append(input)
        #print "input.shape", input.shape
        #print input
        final = numpy.array([]).reshape(2,0)
        #print len(input)
        """
        for i in range(len(input)-1):
            #print input[i],no_output
            if final.shape == (2,0):
                final = numpy.array((input[i],no_output))
            else:
                #print final.shape
                temp = numpy.array((input[i],no_output))
                #print temp
                #print numpy.array([input[i],no_output]).shape
                final = numpy.column_stack((final,temp))
        final = numpy.column_stack(([final,(input[-1],output)]))
        print final.shape
        #print final[0]
        """
        outputs = []
        for i in range(len(input)-1):
            #print input[i],no_output
            #print final.shape
            #temp = numpy.array((input[i],no_output))
            #print temp
            #print numpy.array([input[i],no_output]).shape
            outputs.append([no_output])
        outputs.append(output)
        outputs = numpy.array(outputs)
        """
        print len(outputs)
        for i in range(len(input)):
            if final.shape == (2,0):
                final = numpy.array((input[i],outputs[i]))
            else:
                final = numpy.column_stack([input,outputs])

        print outputs
        """
        #print "outputs.shape", outputs.shape
        final = []
        #print len(input)
        #print len(outputs)
        for i in range(len(input)):
            #print i
            final.append([input[i],outputs[i]])
        #final = numpy.column_stack([input,outputs])
        final = numpy.array(final)
        #print final.shape
        #print final[0]
        #print final[0]
        """
        input=numpy.array(input)
        output = )
        #print input
        #print output
        
        #input.shape = (4,input.shape[0]/4)
        #print input.shape
        #print input
        obs = numpy.array([input, output])
        #print obs
        
        #print numpy.array([obs]).shape
        
        #print test.shape
        test = numpy.concatenate((test,[obs]))
        #print test.shape
        #print test[0]
        #dataset = numpy.append(dataset , numpy.array([input, output]))
        dataset = numpy.concatenate((dataset , test))
    #print dataset.shape
    #print test[0]
    #dataset.reshape(40,2)
        
        #inputsfinal = numpy.append(inputsfinal,numpy.array([input]))
        #outputsfinal = numpy.append(outputsfinal,numpy.array([output]))
    #print dataset[0]
        
        
    #return numpy.array([outputsfinal,inputsfinal])
    return test
    #return dataset
    """
    return dataset
    #return input
    #return final

def read_result_file(filename):
    #print "updating current file"
    try: 
        f = open(filename,'r')
    except:
        f = open(filename,'w')
        f.write('0 0 0\n') #create a sample file with the first line that can be applied to sum
        f.close
        f = open(filename,'r')
    
    index = []
    data = []
    for line in f:
        if line.startswith('#labels'):
            index = line.split(' ')[1:-1]
            #print "index: ", index
        if not(line.startswith('#')):
            #print [x for x in line.split(' ')]
            line_int = [int(x) for x in line.split(' ')[:-1]]
            data.append(line_int)
            #print line_int
    #print data
    checksum = [sum(i) for i in data]
    check = True
    for i in range(len(checksum)):
        if checksum[0] != checksum[i]:
            #some process is currently writing inside the file -> try again
            data = read_result_file(filename)
    return [index, data]
            
def write_result_file(name, matrix, old, config = ""):
    
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

def evaluate(n_states = 5, experiment='all',data_type='raw', data='numbers', subjects='both', iterations=100, val_number=1, binary=True):
    

    path = 'C:\\Google Drive\\Master Thesis\\Data\\new_data_converted_mod\\'
    for filename in os.listdir(path+data+'\\') :
        #print filename
        if filename.endswith(data+'_hmm.csv'):
            
            subject = filename.split('_')[3]
            interact = filename.split('_')[0]
            
            #print "Reading file: ", filename
            print "Visualizing: ", filename
            
            data_dict = core.read_data_dict(path+'\\'+data+'\\'+filename)
            training_norm, means, stds = core.normalize_mean_std_training(data_dict)
    #data_dict = combine_datasets(files) 
    
            visualize(training_norm,filename,subject)
    


def visualize(data_dict,filename_out,subjects):
    path = 'C:\\Google Drive\\Master Thesis\\Python\\workspace\\clean_code\\visualization\\'+filename_out+'/'
    final = defaultdict(list)
    fig = plt.figure()
    for key in data_dict:
        
        i=0
        
        #if len(key)==1:    #for the numbers dataset to exclude the signature
        if True:
            print key
            #print data_dict[key]
            for elem in data_dict[key]:
                #print elem
                #print key," ",len(elem)
                #name="filename="+filename_out+"_gesture="+key+'_'+"_subject="+str(subjects)+"_example="+str(i)+'.png'
                name="gesture="+key+"_example="+str(i)+'.png'
                
                filename = path+name
                
                plt.clf()
                ax = fig.add_subplot(111)
                #ax.set_aspect(1)
                plt.plot(range(len(elem)),elem)
                core.create_file(filename)
                plt.savefig(filename,format='png',dpi=300)
                i+=1
    return final
    return

def visualize_confusion_matrix(matrix, rotated = True, name = 'confusion_matrix.png', type = "numbers", config = []):
    if rotated:
        rotation = 90
    else:
        rotation = 0
    
    #create the array with the labels for the axis
    names_dict = matrix[0]
    names = [0]*len(names_dict.keys())
    for index in range(len(names)):
        key = names_dict.keys()[index]
        pos = names_dict[key]
        names[pos] = key
    if type=="words" or type=="all":
        names[-3]="sentence_1"
        names[-2]="sentence_2"
        names[-1]="sentence_3"
    conf_arr = matrix[1]
    
    #write the data to file
    filename = name.split('.')[0]+'.txt'
    f = open(filename,"w")
    
    #first write the config
    for el in config:
        f.write(el + "\n")
    f.write("#labels: ")
    for i in range(len(names)):
        f.write(names[i]+" ")
    f.write("\n")
    for i in range(len(conf_arr)):
        for j in range(len(conf_arr[i])):
            f.write(str(int(conf_arr[i][j]))+" ")
        f.write("\n")
    f.close()
    
    #normalize the confusion matrix for the color coding
    norm_conf = []
    for i in conf_arr:
        a = 0

        tmp_arr = []
        a = sum(i, 0)
        #print "a = ",a
        
        for j in i:
            tmp_arr.append(float(j)/float(sum(i)))
            #print float(j)/float(sum(i))
        #print tmp_arr
        norm_conf.append(tmp_arr)
    
    
    #create the visualization image
    
    #fig = plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    #vmin and vmax are the range of the colorbar
    res = ax.imshow(numpy.array(norm_conf), interpolation='nearest',  vmin=0, vmax=1)
    
    width = len(conf_arr)
    height = len(conf_arr[0])
    
    
    #plt.gcf().subplots_adjust(bottom=0.1)
    
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(int(conf_arr[x][y])), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    
    cb = fig.colorbar(res)
    plt.xticks(range(width), names[:width],rotation = rotation)
    plt.yticks(range(height), names[:height])
    
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground Truth')
    
    if type=="numbers" or type=="words":
        print "type = ",type
        plt.tight_layout()
    if type=="all" or type=="gestures":
        plt.gcf().subplots_adjust(bottom=0.1)
    if type=="all":
        fig.set_size_inches(20,15)
    """
    if type=="words":
        print "inside words"
        plt.gcf().subplots_adjust(bottom=0.22)
    """
    #fig = plt.gcf()
    
    #fig.set_size_inches(18.5,10.5)

    plt.savefig(name, dpi=300,format='png',bbox_inches='tight')

def combine_confusion_matrices(matrix_array, type="numbers", binary=True):

    dict_names_index = {}
    index_names = []
 
    #this is where the names have to be ordered before visualization
    if binary==True:
        if type=="numbers+words":
            index_names.append('number')
            index_names.append('word')
        if type=="numbers+words+sentences":
            index_names.append('number')
            index_names.append('word')
            index_names.append('sentence')
    else:

        for key in matrix_array[0][0].keys():
            #print key
            index_names.append(key)
    
        #print index_names
        if type=="numbers" or type=="gestures":
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
                #print "y: ",y," ", index[y],"  ind_y: ", ind_y
                #print confusion_matrix[ind_x,ind_y]
                #print matrix[int(x)][int(y)]
                confusion_matrix[ind_x,ind_y] += matrix[index[x]][index[y]]
      
    #print confusion_matrix
    #visualize_confusion_matrix([dict_names_index,confusion_matrix])    
    
    return [dict_names_index,confusion_matrix]
    
def build_hmms(n_states, data):
    #n_classes = len(data.keys()) 
    hmms = {}
    for key in data.keys():
        hmms[key] = hmm.GaussianHMM(n_states, "full")
        hmms[key].fit(data[key])
    return hmms

def k_fold_cross_validation(n_states, data, iterations=10,type=type, val_number=1):
    temp = []
    conf = []
    for i in range(iterations):
        #print "k_fold_cross_validation - starting iteration ",i+1," of ",iterations
        worked = False
        while not(worked):
            try:
                #print filename
                #iterations_done = sum(read_result_file(filename)[0])
                #print "iterations done"
                result = (evaluate_random(n_states, data, val_number=val_number))
                #result_converted = 
                #return result
                worked = True
                #print "this iteration worked"
            except:
                pass
                #True
                #print "Cholesky error, restarting current singular k-fold iteration"
        #print "precision:",result[0]
        confusion = result[1]
        conf_names = confusion[0]
        conf_matrix = confusion[1]
        #print conf_matrix
        temp.append(result[0])
        conf.append(result[1])
        
        combined = combine_confusion_matrices(conf,type=type)
        return combined
        
    #print "combined confusion matrix:"
    #combine_confusion_matrices(conf)
    
    #return [combined,sum(temp)/len(temp)]
    return [conf_names, conf_matrix]
    
def evaluate_random(n_states, data, val_number=1):
    #print data
    #train_data = defaultdict(list)

    n_classes = len(data.keys()) 
    train_data = {}
    #train_data = []
    validation = []
    #val_number=val_ratio*
    train_dict = defaultdict(list)
    
    for key in data.keys():
        training_temp = []
        sequences = data[key][:]
        #print "training on ",len(sequences)-val_number," sequences for class ", key, ", validating on ", val_number
        val_index = random.randint(0, len(sequences)-1) #select the index of the validation sequence
        #print "val_index:", val_index
        #print "before: ",len(sequences)
        for v in range(val_number):
            
            val_seq = sequences[val_index]
            validation.append([key,val_seq])
            training_temp = sequences.pop(val_index)
        for i in range(len(sequences)):
            #if i != val_index:
            train_dict[key].append(sequences[i])
            #print "i: ",i
            #print "sequence length: ",len(training_temp[i])
            #print training_temp[i]
            #train_dict[key].append(training_temp[i])
        #train_dict[key] = sequences.pop(val_index)
        #print "after: ",len(sequences)
        #print len(train_dict[key])
        #print "number of samples: ",len(train_dict[key])
        #sequences.pop(val_index)
        #training = sequences.pop(val_index)
        #for i in range(len(training)):
        #    training_temp.append(training[i])

    #print train_dict
    hmms = build_hmms(n_states, train_dict)
    #print hmms
    #print "after"
    correct = 0
    total   = 0
    
    dict_names_index = defaultdict(list)
    confusion_matrix = numpy.zeros((n_classes, n_classes))
    
    for i in range(len(data.keys())):
        key = data.keys()[i]
        dict_names_index[key] = i
    
    
    #for i in range(len(validation)):    #for every validation element
    for i in range(len(validation)):    #for every validation element
    
        key = validation[i][0]
        eval_ob = validation[i]
        eval_seq = eval_ob[1]
        gt_class = eval_ob[0]
        best = -1
        bestscore = -100000000000
        for j in range(len(hmms.keys())):   #iterate over all the hmms
            #print "j: ",j
            hmm_name = hmms.keys()[j]
            hmm_temp = hmms[hmm_name]   # the hmm to be evaluated
            score = hmm_temp.score(eval_seq)
            #print score
            if score > bestscore:
                best = hmm_name
                bestscore = score
        if str(best) == gt_class:
            correct += 1
        
        ind_x = dict_names_index[gt_class]
        ind_y = dict_names_index[str(best)]
        confusion_matrix[ind_x,ind_y] += 1
        
        total += 1
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

def combine_to_binary_dataset(data_dict_list):
    final = defaultdict(list)
    for dict in data_dict_list:
        for key in dict:
            
            for elem in dict[key]:
                #if ord(key)>=48 and ord(key)<=57:
                if len(key)==1:
                    c = 'number'
                else: 
                    if len(key)<=20:
                        c = 'word'
                    else:
                        c = 'sentence'
                #print key," ",len(elem)
                final[c].append(elem)
    return final

def main(argv):
    
    n_states = 2
    iterations = 100
    val_number = 40 #the number of sequences that are excluded from training to evaluate on (also decreases the training iterations needed
    #experiment = "all"
    #experiment = 4
    experiment = 5
    binary=True
    data_type = 'raw'
    #data_type ='filtered'
    
    #data = 'numbers'
    #data = 'numbers+words'
    #data = 'numbers+words+sentences'
    data = 'gestures'
    #data = 'words'

    #data = 'all'

    subjects = 1
    #subjects = 2
    #subjects = "both"
    print "Starting evaluation"
    for states in range(1,2):
        #print "Evaluating with ", states, " hidden states"
        evaluate(n_states = states, iterations=iterations, experiment=experiment, data_type=data_type, data=data,subjects=subjects, val_number=val_number, binary=binary)
    print "finished evaluating everything"
    #evaluate(n_states = n_states, iterations=iterations, experiment=experiment, data_type=data_type, data=data,subjects=subjects)
    
    return

if __name__ == "__main__":
    main(sys.argv[1:])