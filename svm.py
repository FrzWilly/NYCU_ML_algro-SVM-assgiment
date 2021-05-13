from libsvm.svmutil import *
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
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

def preprocess_data(x, y, method = "A"):

    ################ preprocess y #############################
    for i in range(len(y)):
        if y[i] == 6:
            y[i] = 1
        else:
            y[i] = -1

    ################ row-wise data normalization #######################

    # normalize method A
    if method == "A":
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

    # normalize method B
    else:
        for data in x:
            a = 0

            for i in range(1,36):
                try:
                    a += (data[i])**2
                except:
                    continue
            
            a = a**0.5

            for key in range(1,36):
                try:
                    data[key] /= a
                except:
                    continue

def train_for_each_dC_pair(cv_tenfold_x, cv_tenfold_y):
    xcoordinate = [k for k in range(-10, 11)]
    xsimp = [k for k in range(-10, -5)]

    best_C = []
    best_C_acc = []

    for degree in range(1,5):
        mean_vec = []
        std_vec = []
        for k in range(-10, 11):
            C = 2**k
            option = f'-t 1 -c {C} -d {degree} -h 0'
            acc = []
            for fold in range(10):

                print(degree, k, fold)
                trn_x, vld_x = get_trn_vld_data(cv_tenfold_x, fold)
                trn_y, vld_y = get_trn_vld_data(cv_tenfold_y, fold)

                m = svm_train(trn_y, trn_x, option)
                p_label, p_acc, p_val = svm_predict(vld_y, vld_x, m)
                
                acc.append(p_acc[0])

            acc_arr = np.array(acc)
            mean_vec.append(np.mean(acc_arr))
            std_vec.append(np.std(acc_arr))

        title = f'degree: {degree}'
        plt.title(title)
        plt.xlabel("k", fontsize=20)
        plt.ylabel("acc", fontsize=20)
        plt.errorbar(xcoordinate, mean_vec, yerr=std_vec,fmt='o',capthick=2)
        plt.plot(xcoordinate, mean_vec)
        plt.savefig(title, bbox_inches='tight')
        
        #plt.clf()

        mean_arr = np.array(mean_vec)
        best_C.append(np.argmax(mean_arr))
        best_C_acc.append(np.max(mean_arr))

    for i in range(4):
        print(best_C[i], best_C_acc[i])

    return best_C[np.argmax(np.array(best_C_acc))]

def fix_C_plot(cv_tenfold_x, cv_tenfold_y, xt, yt, ks):
    xcoordinate = [d for d in range(1, 5)]
    tr_mean_vec = []
    tr_std_vec = []
    ts_mean_vec = []
    ts_std_vec = []
    nr_sv_mean = []
    nr_msv_mean = []
    print('C* = ', 2**ks)
    for degree in range(1,5):


        C = 2 ** ks
        option = f'-t 1 -c {C} -d {degree} -h 0'
        tr_acc = []
        ts_acc = []
        nr_sv = []
        nr_msv = []
        for fold in range(10):

            print(degree, fold)
            trn_x, vld_x = get_trn_vld_data(cv_tenfold_x, fold)
            trn_y, vld_y = get_trn_vld_data(cv_tenfold_y, fold)

            m = svm_train(trn_y, trn_x, option)
            p_label, p_acc, p_val = svm_predict(vld_y, vld_x, m)
            t_label, t_acc, t_val = svm_predict(yt, xt, m)
                
            tr_acc.append(p_acc[0])
            ts_acc.append(t_acc[0])
            nr_sv.append(m.get_nr_sv())
            SV = m.get_sv_coef()
            nBV = 0
            for sv in SV:
                if abs(sv[0]) == 1024:
                    nBV += 1
            nr_msv.append(m.get_nr_sv() - nBV)


        tracc_arr = np.array(tr_acc)
        tsacc_arr = np.array(ts_acc)
        nrsv_arr = np.array(nr_sv)
        nrmsv_arr = np.array(nr_msv)
        tr_mean_vec.append(np.mean(tracc_arr))
        tr_std_vec.append(np.std(tracc_arr))
        ts_mean_vec.append(np.mean(tsacc_arr))
        ts_std_vec.append(np.std(tsacc_arr))
        nr_sv_mean.append(np.mean(nrsv_arr))
        nr_msv_mean.append(np.mean(nrmsv_arr))

    title = 'C* to degree acc on training & testing set'
    plt.title(title)
    plt.xlabel("degree", fontsize=20)
    plt.ylabel("acc", fontsize=20)
    plt.errorbar(xcoordinate, tr_mean_vec, yerr=tr_std_vec,fmt='o',capthick=2)
    plt.plot(xcoordinate, tr_mean_vec)
    plt.errorbar(xcoordinate, ts_mean_vec, yerr=ts_std_vec,fmt='o',capthick=2)
    plt.plot(xcoordinate, ts_mean_vec)
    plt.savefig(title, bbox_inches='tight')

    plt.clf()

    title = 'C* to degree # of SVs'
    plt.title(title)
    plt.xlabel("# of SVs", fontsize=20)
    plt.ylabel("acc", fontsize=20)
    plt.plot(xcoordinate, nr_sv_mean, '-o')
    plt.plot(xcoordinate, nr_msv_mean, '-o')
    plt.savefig(title, bbox_inches='tight')

def false_positive_error_plot(cv_tenfold_x, cv_tenfold_y):

    xcoordinate = [k for k in range(-10, 11)]
    xsimp = [k for k in range(-10, -5)]

    for degree in range(1, 5):

        mean_vecs = []
        std_vecs = []
        for ck in [0, 2, 4, 8, 16]:
            w = ck
            mean_vec = []
            std_vec = []
            for k in range(-10, 11):
                C = 2**k
                option = f'-t 1 -c {C} -d {degree} -h 0 -w1 {w}'
                acc = []
                for fold in range(10):

                    print(degree, ck, k, fold)
                    trn_x, vld_x = get_trn_vld_data(cv_tenfold_x, fold)
                    trn_y, vld_y = get_trn_vld_data(cv_tenfold_y, fold)

                    m = svm_train(trn_y, trn_x, option)
                    p_label, p_acc, p_val = svm_predict(vld_y, vld_x, m)
                    
                    acc.append(p_acc[0])

                acc_arr = np.array(acc)
                mean_vec.append(np.mean(acc_arr))
                std_vec.append(np.std(acc_arr))

            mean_vecs.append(mean_vec)
            std_vecs.append(std_vec)

        title = f'degree: {degree} -wi ver.'
        plt.title(title)
        plt.xlabel("log(C)", fontsize=20)
        plt.ylabel("acc", fontsize=20)
        lg = []
        for i in range(len(mean_vecs)):
            plt.errorbar(xcoordinate, mean_vecs[i], yerr=std_vecs[i],fmt='o',capthick=2)
            ll, = plt.plot(xcoordinate, mean_vecs[i], label=f'line {i}')
            lg.append(ll)
        plt.legend(handles=lg, labels = ['k=0', 'k=2', 'k=4', 'k=8', 'k=16'])
        plt.savefig(title, bbox_inches='tight')

        plt.clf()

def main():

    y, x = svm_read_problem('satimage.scale.merged')
    yt, xt = svm_read_problem('satimage.scale.testing')

    # preprocess training set & testing set
    preprocess_data(x, y)
    preprocess_data(xt, yt)

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

    # training for each degree and C
    #ks = train_for_each_dC_pair(cv_tenfold_x, cv_tenfold_y)-10

    # fix C* and plot for every degree
    #fix_C_plot(cv_tenfold_x, cv_tenfold_y, xt, yt, ks)

    # penalize false positive error more
    false_positive_error_plot(cv_tenfold_x, cv_tenfold_y)

    


#   mean_arr = np.array(mean_vec)
#   best_C.append(np.argmax(mean_arr))
#   best_C_acc.append(np.max(mean_arr))



if __name__ == '__main__':
    main()
