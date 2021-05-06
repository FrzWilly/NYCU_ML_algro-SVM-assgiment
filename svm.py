from libsvm.svmutil import *
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import random
import math

def flatten(lst):
    
    newlist = [item for sublist in lst for item in sublist]
    return newlist

def get_trn_vld_data(lst, fold):
    trn = lst[:fold] + lst[fold+1:]
    trn = flatten(trn)
    vld = lst[fold]

    return trn, vld

def main():

    y, x = svm_read_problem('satimage.scale.merged')

    ################ preprocess y #############################
    for i in range(len(y)):
        if y[i] == 6:
            y[i] = 1
        else:
            y[i] = -1

    ################ row-wise data normalization #######################

    for data in x:
        a = []

        for i in range(1,36):
            try:
                a.append(data[i])
            except:
                continue
        
        arr = np.asarray(a)
        mean = np.mean(arr)
        std = np.std(arr)

        for key in range(1,36):
            try:
                data[key] = (arr[key-1] - mean)/std
            except:
                continue

    ################# ten-fold cross validation ###########################

    index = [i for i in range(len(x))]
    random.shuffle(index)

    cv_tenfold_x = []
    cv_tenfold_y = []
    for i in range(10):
        emptylistx = []
        emptylisty = []
        cv_tenfold_x.append(emptylistx)
        cv_tenfold_y.append(emptylisty)

    count = 0
    for i in index:
        idx = math.floor(count/(len(x)/10))
        # print(idx, count, i)
        cv_tenfold_x[idx].append(x[i])
        cv_tenfold_y[idx].append(y[i])

        count += 1

    ################### training for each degree and C ##################

    for degree in range(4):
        mean_vec = []
        std_vec = []
        for k in range(-10, 10):
            C = 2**k
            option = f'-t 0 -c {C} -d {degree}'
            acc = []
            for fold in range(10):

                print(degree, k, fold)
                trn_x, vld_x = get_trn_vld_data(cv_tenfold_x, fold)
                trn_y, vld_y = get_trn_vld_data(cv_tenfold_y, fold)

                print(len(trn_x), len(vld_x))
                m = svm_train(trn_y, trn_x, option)
                p_label, p_acc, p_val = svm_predict(vld_y, vld_x, m)
                
                acc.append(p_acc)
                
            acc_arr = np.array(acc)
            mean_vec.append(np.mean(acc_arr))
            std_vec.append(np.std(acc_arr))

        print(mean_vec)

        break

    # data = np.array(lines)
    # test = lines[0].split(" |:")
    #print(test)

    # normed_data = preprocessing.normalize(data)

    # print(normed_data)

main()