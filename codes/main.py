import pandas as pd
import numpy as np
import config as cf

import sys
import numpy
numpy.set_printoptions(threshold = sys.maxsize)

from indx_features import indx_ID
from sp_quicksort import quickSort
from Diagnosis_class import Diagnosis_class
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from math import pow

from sklearn import svm #main
from sklearn.feature_selection import RFE #main
from sklearn.svm import SVC #main
from sklearn.utils.validation import column_or_1d #mian
from sklearn.model_selection import train_test_split #main
from sklearn.metrics import recall_score #main
from sklearn.model_selection import GridSearchCV

from sklearn import datasets #test
from sklearn.decomposition import PCA #test
from sklearn.datasets import make_classification #test
from sklearn.feature_selection import RFECV #test
from sklearn.model_selection import StratifiedKFold #test
iris = datasets.load_iris()

from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
#from sklearn.pipeline import Pipeline
my_list = []
speaker_list = []

def prediction_svm(X_train,y_train,X_predict,y_actual,ker,ker_coee,C):
        #ker = 'linear'
        #clf = svm.SVC(gamma='auto',kernel = ker)
        clf = svm.SVC(gamma = ker_coee, kernel = ker, C = C,verbose = True)
        #sklearn.model_selection.GridSearchCV
        clf.fit(X_train, np.ravel(y_train,order='C'))
        pred = clf.predict(X_predict)
        #N = np.shape(pred)[0]
        '''
        TP = 0 
        FP = 0 
        FN = 0 
        TN = 0
        '''
        #UAR evaluation
        # print(pred)
        #UAR metrics.balanced_accuracy_score(y_true, y_pred)
        '''
        for i in range(N):
                if pred[i] == 1 and y_actual[i] == 1:
                        TP = TP + 1
                if pred[i] == 1 and y_actual[i] == 2:
                        FP = FP + 1
                if pred[i] == 2 and y_actual[i] == 1:
                        FN = FN + 1
                if pred[i] == 2 and y_actual[i] == 2:
                        TN = TN + 1
        print(TP,FP,TN,FN)
        if TP + FP == 0:
                R1 = -1
                return TN/(TN+FN)
        else:
                R1 = TP/(TP+FP)
                
        if TN + FN == 0:
                R2 = -1
                return R1
        else:
                R2 = TN/(TN+FN)
                
        
        UAR = (R1+R2)/2
        '''
        UAR = recall_score(y_actual,pred,average = 'macro')
        f = open("result.txt","a")
        f.write("\n" + ker + " " + ker_coee + " " + str(C) + "\n" + np.array2string(y_actual.T) + "\n\n" + np.array2string(pred.T))
        f.close()
        return UAR
        
def dim_elimination(Features, labels, task):
        #RFE = Ranking recursive feature elimination
        ker = "linear"
        C_value = 1
        N_features = 5
        N_steps = 1
        svc = SVC(kernel = ker, C = C_value)
        rfe = RFE(estimator = svc, n_features_to_select = N_features, step = N_steps)
        N_labels = np.shape(task)[1]
        #rfe = RFECV(estimator=svc, step=N_steps, cv=StratifiedKFold(2),scoring='accuracy')
        if task == "binary":
                for i in range(N_labels):
                        if labels[i] > 1:
                                labels[i] = 2
                        
        rfe.fit(Features, labels)
        ranking = rfe.ranking_
        return ranking

def ranking(path1,path2,task):
        label = read(path1)
        inp = read(path2)
        y = np.array(label)
        X_tmp = np.array(inp)
        size = np.shape(X_tmp)[1]
        X = X_tmp[:,2:size]
        if task == "binary":
                rank = dim_elimination(X,np.ravel(y,order='C'),"binary")
        if task == "multiple":
                rank = dim_elimination(X,np.ravel(y,order='C'),"multiple")
        return rank

def getFeatures(X):
        size = np.shape(X)[1]
        return X[:,2:size]

def binary_array(array):
        N = np.shape(array)[0]
        var = np.zeros(N)
        for i in range(N):
                if array[i] > 1:
                        var[i] = 2
                else:
                        var[i] = 1
        return var

def group_features(train_class, test_class):
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []
        train_size = np.shape(train_class)[0]
        test_size = np.shape(test_class)[0]

        for i in range(train_size):
                feature_size = np.shape(train_class[i].features)[0]
                for j in range(feature_size):
                        #train_features = np.vstack((train_class[i].features,train_features))
                        train_features.append(train_class[i].features[j])
                        train_labels.append(train_class[i].label[j])
        
        for i in range(test_size):
                test_feature_size = np.shape(test_class[i].features)[0]
                for j in range(test_feature_size):
                        test_features.append(test_class[i].features[j])
                        test_labels.append(test_class[i].label[j])
        #print(train_features, np.shape(train_features))
        '''
        print(np.shape(train_features),np.shape(train_class))
        print(np.shape(train_labels))
        print(np.shape(test_features),np.shape(test_class))
        print(np.shape(test_labels))
        '''
        return np.array(train_features), np.array(train_labels), np.array(test_features), np.array(test_labels)

def exhaustive_search(my_list):
        kernel_coee = ['scale','auto','auto_deprecated']
        kernel = ["linear","rbf","sigmoid"]
        uars = []

        N_coee = 3
        N_kernel = 3
        N_C = 17
        #C = [1,2,3,4,5] #how much you want to avoid misclassifying each training example
        C = 1e-10
        times = 6
        
        for i in range(N_coee):
                for j in range(N_kernel):
                        for k in range(N_C):
                                if C < 1:
                                        C = C *10
                                else:
                                        C = C + 1
                                for n in range(times):
                                        #refold my_list
                                        #train class, test class
                                        #print("1")
                                        train_class,test_class = train_test_split(my_list, test_size = 0.15)
                                        #print("2")
                                        train_features, train_labels, test_features, test_labels = group_features(train_class,test_class)
                                        train_labels = binary_array(train_labels)
                                        test_labels = binary_array(test_labels)
                                        #print("3")
                                        tmp_uar = prediction_svm(train_features,train_labels,test_features,test_labels,kernel[j],kernel_coee[i],C)
                                        #print("4")
                                        uars.append(tmp_uar)
                                        #split class->training by prediction svm->find appropriate coefficient exhaustively->dimension elimination
        
        return uars
        

#13/06 grouping features and labels to an array
def class_init():
        tmp_list = []
        path_label = "../result_csv/all_label.csv"
        path_result = "../result_csv/all_result.csv"
        path_speakerID = "../ComParE2013_Autism.csv"
        tmp_label = read(path_label)
        tmp_result = read(path_result)
        tmp_speaker = read(path_speakerID)
        label = np.array(tmp_label)
        features = np.array(tmp_result)
        speakerID = np.array(tmp_speaker)[:,1]
        N1 = np.shape(label)[0]
        N2 = cf.NUM_SPEAKERS
        #print(N2)
        for i in range(N1):
                tmp_list.append(Diagnosis_class(label[i],speakerID[i],features[i]))
        quickSort(tmp_list,0,N1-1)
        my_list.append(tmp_list[0])
        j = 0
        for i in range(1,N1):
                if(tmp_list[i-1].speaker_ID) == (tmp_list[i].speaker_ID):
                        my_list[j].label = np.vstack((tmp_list[i].label,my_list[j].label))
                        my_list[j].features = np.vstack((tmp_list[i].features,my_list[j].features))
                else:
                        my_list.append(tmp_list[i])
                        j = j + 1

        #x,y = train_test_split(my_list,test_size = 0.15, random_state = 10)
        #x,y = train_test_split(my_list,test_size = 0.15)
        #train_features, train_labels, test_features, test_labels = group_features(x,y)
        return my_list
        #return train_features, train_labels, test_features, test_labels
        #print(np.shape(y))

        #random state = initialization of random seed
        #dim x[t].features is (a,88), which a indicates the number of speakers and 88 is the num of features
        
        #print(my_list[N2-1].features,np.shape(my_list[N2-1].features))
        #print(my_list[1].speaker_ID,my_list[1].features,my_list[2].speaker_ID,my_list[2].features)
        #may not need group label together


def cross_validation(train_features, train_labels, test_features, test_labels):
        parameters = {'kernel':('linear','rbf','sigmoid'), 'C' : [1,10]}
        svc = svm.SVC(gamma = "scale")
        out = GridSearchCV(svc,parameters, cv = 5)
        out.fit(train_features,train_labels)
        print(sorted(out.cv_results_.keys()))


def main():
        path1 = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/devel_lable.csv"
        path2 = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/result_devel.csv"
        path3 = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/result_train.csv"
        path4 = "C:/Users/Yitian Chen.DESKTOP-TIIP443/Desktop/tmp1/training/result_csv/train_lable.csv"
        label_devel = read(path1)
        inp_devel = read(path2)
        y_train = np.array(label_devel)
        X_train = np.array(inp_devel)
        X_train = getFeatures(X_train)
        
        label_train = read(path4)
        inp_train = read(path3)
        y_actual = np.array(label_train)
        X_predict = np.array(inp_train)
        X_predict = getFeatures(X_predict)
        UAR = prediction_svm(X_train,binary_array(y_train),X_predict,binary_array(y_actual))
        print(UAR)

        #rank = ranking(path1,path2,"binary")
        #print(rank)
        
def main2():
        #train_features, train_labels, test_features, test_labels = class_init()
        #cross_validation(train_features, train_labels, test_features, test_labels)
        my_list = class_init()
        uar = exhaustive_search(my_list)
        print(uar)


def my_split(my_list,chosen):
        N = np.shape(my_list)[0]
        start = 0
        end = N
        if start == chosen:
                start = 2
        if end == chosen:
                end = N-1
        y1 = my_list[start:chosen]
        y2 = my_list[chosen+1:end]
        x = my_list[chosen]
        return y1,y2,x

def test_split():
        X = [1,2,3,4,5,6]
        for i in range(6):
                y1,y2,x = my_split(X,i)
                print(y1,y2,x)

def test_array():
        x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,]

        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1,
        2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1,
        2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2,
        1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2,]
        uar = recall_score(np.array(x),np.array(y),average = 'macro')
        print(uar)

def read(csvPath):
    fileCVS = pd.read_csv(csvPath)
    return fileCVS
    
#initialize()
#test()
#main()
#test1()
#class_init()
main2()
#test_split()
#test_array()