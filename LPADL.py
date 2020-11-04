import os
os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility, needs to be the first 2 lines in a script

import logging
import json

from theano import ifelse
import theano.tensor as T
from keras.models import Model#, load_model
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn import metrics
from sklearn.metrics import accuracy_score
import collections
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from keras import optimizers
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import catboost
from keras.utils import np_utils, generic_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE, SelectKBest, f_classif, chi2, RFECV
from sklearn.feature_selection import SelectFromModel
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, ZeroPadding1D
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint,TensorBoard, EarlyStopping
from keras.layers.core import Flatten
from keras.layers.recurrent import LSTM
from keras.layers import merge, Input
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import metrics, regularizers
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.optimizers import RMSprop
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
import scipy.io as sio
import numpy as np
from numpy import linalg as la
import argparse
from keras.utils import plot_model
from ddrop.layers import DropConnectDense, DropConnect

def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**4
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        n=n/base
        ch3=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    return  nucle_com   

def get_tris():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com   

def get_di():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base=len(chars)
    end=len(chars)**2
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        #n=n/base
        #ch2=chars[n%base]
        nucle_com.append(ch0 + ch1)
    return  nucle_com   
           
def get_tri_nucleotide_composition(tris, seq):
    seq_len = len(seq)
    tri_feature = []
    for val in tris:
        num = seq.count(val)
        tri_feature.append(float(num)/seq_len)
    return tri_feature

def get_3_protein_trids():
    nucle_com = []
    chars = ['0', '1', '2', '3', '4', '5', '6']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com

def get_3_protein_struct_trids():
    nucle_com = []
    chars = ['H', 'E', 'C']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com    

def get_4_nucleotide_composition(tris, seq, pythoncount = True):
    seq_len = len(seq)
    tri_feature = []
    
    if pythoncount:
        for val in tris:
            num = seq.count(val)
            tri_feature.append(float(num)/seq_len)
    else:
        k = len(tris[0])
        tmp_fea = [0] * len(tris)
        for x in range(len(seq) + 1- k):
            kmer = seq[x:x+k]
            if kmer in tris:
                ind = tris.index(kmer)
                tmp_fea[ind] = tmp_fea[ind] + 1
        tri_feature = [float(val)/seq_len for val in tmp_fea]        
    return tri_feature

def translate_sequence (seq, TranslationDict):
    import string
    from_list = []
    to_list = []
    for k,v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)    
    TRANS_seq = seq.translate(string.maketrans(str(from_list), str(to_list)))    
    return TRANS_seq

def TransDict_from_list(groups):
    transDict = dict()
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group) 
        for c in g_members:            
            result[c] = str(tar_list[index]) 
        index = index + 1
    return result

coden_dict = {'GCU': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,                             # alanine<A>
              'UGU': 1, 'UGC': 1,                                                 # systeine<C>
              'GAU': 2, 'GAC': 2,                                                 # aspartic acid<D>
              'GAA': 3, 'GAG': 3,                                                 # glutamic acid<E>
              'UUU': 4, 'UUC': 4,                                                 # phenylanaline<F>
              'GGU': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,                             # glycine<G>
              'CAU': 6, 'CAC': 6,                                                 # histidine<H>
              'AUU': 7, 'AUC': 7, 'AUA': 7,                                       # isoleucine<I>
              'AAA': 8, 'AAG': 8,                                                 # lycine<K>
              'UUA': 9, 'UUG': 9, 'CUU': 9, 'CUC': 9, 'CUA': 9, 'CUG': 9,         # leucine<L>
              'AUG': 10,                                                          # methionine<M>
              'AAU': 11, 'AAC': 11,                                               # asparagine<N>
              'CCU': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,                         # proline<P>
              'CAA': 13, 'CAG': 13,                                               # glutamine<Q>
              'CGU': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,   # arginine<R>
              'UCU': 15, 'UCC': 15, 'UCA': 15, 'UCG': 15, 'AGU': 15, 'AGC': 15,   # serine<S>
              'ACU': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,                         # threonine<T>
              'GUU': 17, 'GUC': 17, 'GUA': 17, 'GUG': 17,                         # valine<V>
              'UGG': 18,                                                          # tryptophan<W>
              'UAU': 19, 'UAC': 19,                                               # tyrosine(Y)
              'UAA': 20, 'UAG': 20, 'UGA': 20,                                    # STOP code
              }

# the amino acid code adapting 21-dimensional vector (20 amino acid and 1 STOP code)


def coden(seq):
    vectors = np.zeros((len(seq) - 2, 21))
    for i in range(len(seq) - 2):
        vectors[i][coden_dict[seq[i:i+3].replace('T', 'U')]] = 1
    return vectors.tolist()

def prepare_RPI_feature(deepmind = False, seperate=False):
    print 'RPI dataset'
    lncRNA = pd.read_csv("zma_lncRNA.csv")
    protein = pd.read_csv("zma_rbp.csv")
    interaction = pd.read_fwf("athinteraction.txt") #fwf stands for fixed width formatted lines
    
    interaction_pair = {}
    RNA_seq_dict = {}
    protein_seq_dict = {}
    with open('athinteraction.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label = values[1]
                name = values[0].split('_')
                protein = name[0] + '-' + name[1]
                RNA = name[0] + '-' + name[1]
                if label == 'interactive':
                    interaction_pair[(protein, RNA)] = 1
                else:
                    interaction_pair[(protein, RNA)] = 0
                index  = 0
            else:
                seq = line[:-1]
                if index == 0:
                    protein_seq_dict[protein] = seq
                else:
                    RNA_seq_dict[RNA] = seq
                index = index + 1             
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()    
    tridi = get_di()
    train = []
    label = []
    RNA_fea = coden(seq)[::-1]
    for key, val in interaction_pair.iteritems():
        protein, RNA = key[1], key[1]
        if RNA_seq_dict.has_key(RNA) and protein_seq_dict.has_key(protein): 
            label.append(val)
            RNA_seq = RNA_seq_dict[RNA]
            protein_seq = translate_sequence (protein_seq_dict[protein], group_dict)
	    RNA_fea1 = get_4_nucleotide_composition(tris, RNA_seq, pythoncount =False)
            if deepmind:
                RNA_tri_fea = RNA_fea1+list(RNA_fea[1]) 	
                protein_tri_fea = get_RNA_seq_concolutional_array(protein_seq) 
                train.append((RNA_tri_fea, protein_tri_fea))
            else:               
                RNA_tri_fea = RNA_fea1+list(RNA_fea[1]) 
		protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)
                if seperate:
                    tmp_fea = (protein_tri_fea, RNA_tri_fea)                    
                else:
                    tmp_fea = protein_tri_fea + RNA_tri_fea                    
                train.append(tmp_fea)                
        else:
            print RNA, protein   
    
    return np.array(train), label


def get_data(dataset):
    if dataset == 'RPI450':
        X, labels = prepare_RPI450_feature(deepmind = False, seperate=False)
        
    print X.shape   
    X, scaler = preprocess_data(X)
    
    
    dims = X.shape[1]
    print(dims, 'dims')
    
    return X, labels

def preprocess_data(X, scaler=None, stand = True):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

delta = 0.1
def huber(target, output):
    d = target - output
    a = .5 * d**2
    b = delta * (abs(d) - delta / 2.)
    l = T.switch(abs(d) <= delta, a, b)
    return l.sum()

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
    else:
        precision = float(tp)/(tp+ fp)
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
        MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, precision, sensitivity, specificity, MCC 

def plot_roc_curve(labels, probality, legend_text, auc_tag = True):
    fpr, tpr, thresholds = roc_curve(labels, probality) #probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text +' (AUC=%6.3f) ' %roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text )

def plot_auprc_curve(labels, probality, legend_text, auprc_tag = True):
    precision1, recall, threshods = precision_recall_curve(labels, probality)
    aupr_score = auc(recall, precision1)
    if auprc_tag:
        rects1 = plt.plot(recall, precision1, label=legend_text +' (AUPRC=%6.3f) ' %aupr_score)
    else:
        rects1 = plt.plot(recall, precision1, label=legend_text )

def LSTM_model():
    #timesteps = 1
    data_dim = 200
    timesteps = 1
    batch_size = 64        
    print 'LPA_DL'	
    model = Sequential()        
    model.add(LSTM(64, return_sequences=False,input_shape=(timesteps, data_dim), name='lstm1'))  
    model.add(DropConnect(Dense(2, activation='relu'), prob=0.25, name='full_connect'))
    model.add(Activation('sigmoid'))
    model.summary()
    print('Compiling the Model...')
    model.compile(loss='mean_squared_error', #'mean_squared_error',  #huber
              optimizer='adam',
              metrics=['accuracy'])
    return model

def LPI(dataset = 'RPI'):
    data_dim = 620
    timesteps = 1
    batch_size = 64  
    epochs = 10
    X, labels = get_data(dataset)    
    y, encoder = preprocess_labels(labels)
    
    num_cross_val = 5
    all_performance_lpa = []
    all_performance_rf = []
    all_performance_xgb = []
    all_performance_rse = []
    all_performance_blend1 = []
    all_performance = []

    all_labels = []
    all_prob = {}
    num_classifier = 3
    all_prob[0] = []
    all_prob[1] = []
    for fold in range(num_cross_val):
        train = []
        test = []
        train = np.array([x for i, x in enumerate(X) if i % num_cross_val != fold])
        test = np.array([x for i, x in enumerate(X) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
	train1 = np.reshape(train, (train.shape[0], 1, train.shape[1]))
	test1 = np.reshape(test, (test.shape[0], 1, test.shape[1]))
          
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)

        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)
        
        blend_train = np.zeros((train1.shape[0], num_classifier)) # Number of training data x Number of classifiers
        blend_test = np.zeros((test1.shape[0], num_classifier)) # Number of testing data x Number of classifiers         
        
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)
                
        all_labels = all_labels + real_labels
        '''
        prefilter_train1 = xgb.DMatrix( prefilter_train, label=train_label_new)
        evallist  = [(prefilter_train1, 'train')]
        num_round = 10
        clf = xgb.train( plst, prefilter_train1, num_round, evallist )
        prefilter_test1 = xgb.DMatrix( prefilter_test)
        ae_y_pred_prob = clf.predict(prefilter_test1)
        '''
        tmp_aver = [0] * len(real_labels)

	
        print("Train...")

	svc = OneVsRestClassifier(SVC(kernel="linear", random_state=123, probability=True), n_jobs=-1) #, C=1
	#svc=SVC(kernel='poly',degree=2,gamma=1,coef0=0)
    	rfe = RFE(estimator=svc, n_features_to_select=200, step=1)
	rfe.fit(train, train_label_new)
	train2 = rfe.transform(train)
	test2 = rfe.transform(test)
	train11 = np.reshape(train2, (train2.shape[0], 1, train2.shape[1]))
	test11 = np.reshape(test2, (test2.shape[0], 1, test2.shape[1]))

	class_index = 0
	model = KerasRegressor(build_fn=LSTM_model, epochs=15, verbose=0)
	model.fit(train11, train_label, epochs=15, verbose=2)
	pred_prob = model.predict(test11)[:,1]
	all_prob[class_index] = all_prob[class_index] + [val for val in pred_prob]
        proba = transfer_label_from_prob(pred_prob)        
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr_1, tpr_1, auc_thresholds = roc_curve(real_labels, pred_prob)
        auc_score_1 = auc(fpr_1, tpr_1)
	precision1, recall, threshods = precision_recall_curve(real_labels, pred_prob)
        aupr_score = auc(recall, precision1)
        print "LPA_DL :", acc, precision, sensitivity, specificity, MCC, auc_score_1, aupr_score
        all_performance_lpa.append([acc, precision, sensitivity, specificity, MCC, auc_score_1, aupr_score])
	print '---' * 50 

	model = Sequential()
        model.add(LSTM(64, return_sequences=False,input_shape=(timesteps, data_dim), name='lstm1'))  #kernel_regularizer=regularizers.l2(0.0001),# returns a sequence of vectors of dimension 32
        model.add(Dropout(0.25, name='dropout'))
        #model.add(Dense(2, name='full_connect'))
	model.add(DropConnect(Dense(2, activation='relu', kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001)), prob=0.25, name='full_connect'))
        model.add(Activation('sigmoid'))
        model.summary()

        print('Compiling the Model...')
        model.compile(loss=huber,
              optimizer='adam',
              metrics=['accuracy']) 

	es = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=5)
        model.fit(train1, train_label, batch_size=batch_size,epochs=epochs, callbacks=[es], shuffle=True, verbose=2) #validation_split=0.1,
        class_index = class_index + 1
        proba = model.predict_proba(test1)[:,1]
	tmp_aver = [val1 + val2/3 for val1, val2 in zip(proba, tmp_aver)]
	all_prob[class_index] = all_prob[class_index] + [val for val in proba]
        y_pred_xgb = transfer_label_from_prob(proba)
        
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)
        
        #pdb.set_trace()            
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_xgb,  real_labels)
        fpr_1, tpr_1, auc_thresholds = roc_curve(real_labels, proba)
        auc_score_1 = auc(fpr_1, tpr_1)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, proba)
        aupr_score = auc(recall, precision1)
        print acc, precision, sensitivity, specificity, MCC, auc_score_1, aupr_score
        all_performance.append([acc, precision, sensitivity, specificity, MCC, auc_score_1, aupr_score])
        print '---' * 50

    print 'mean performance of LPA_DL-FS'
    print np.mean(np.array(all_performance_lpa), axis=0)
    print '---' * 50 
    print 'mean performance of LPA_DL'
    print np.mean(np.array(all_performance), axis=0)
    print '---' * 50 
    
    Figure = plt.figure()
    plot_roc_curve(all_labels, all_prob[0], 'LPI_WFS')
    plot_roc_curve(all_labels, all_prob[1], 'LPI_NFS')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show() 
  
def transfer_label_from_prob(proba):
    label = [1 if val>=0.5 else 0 for val in proba]
    return label

if __name__=="__main__":
    LPI()
