#coding=utf-8
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier

#target is [0 1] for helix and [1 0] for non-helix

#parameters
size_1h = 20  #size of one-hot amino acid encoding
size_w2v = 5  #size of word2vec amino acid encoding 
classes = 2 #number of classes (helix / non-helix)
sequence_length = 14 #length of the sequences in the dataset
splitting_seed = 76347917
#hyperparameters
encoding = '1h' #can be '1h' (one hot) or 'w2v' (word2vec)

#load data
print("Loading input data")
X_helix = None #helix input tensor
X_nonhelix = None #non-helix input tensor
size_encoding = None #size of amino acid encoding 


#load data according to specified encoding
if encoding == '1h':
    X_helix = pd.read_csv('Data/HelicesOneHot.csv')
    X_helix["target"]=1
    X_nonhelix = pd.read_csv("Data/Not_HelicesOneHot.csv")
    X_nonhelix["target"]=0
    size_encoding = size_1h
elif encoding == 'w2v':
    X_helix = pd.read_csv('Data/Helicesw2v.csv')
    X_helix["target"]=1
    X_nonhelix = pd.read_csv("Data/Not_Helicesw2v.csv")
    X_nonhelix["target"]=0
    size_encoding = size_w2v
#detect errors in data loading (expectedly caused by wrong encoding specification)
if X_helix is None or X_nonhelix is None:
    sys.exit("ERROR: could not load input data, check correctness of encoding specification")
    
Train=X_helix.append(X_nonhelix,ignore_index=True)

#build the model
print("Building RF model")
for i in range(0,20):
    
    Train=Train.sample(random_state=int(splitting_seed), frac=1)

    X_train=Train.iloc[:-2400, :-1].values
    y_train=Train.iloc[:-2400, -1:].values
    X_test=Train.iloc[-2399:, :-1].values
    y_test=Train.iloc[-2399:, -1:].values

    clf = RandomForestClassifier(n_estimators=1000, max_depth=21, n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)
    #print(clf.get_params())

    attention_final=[]
    att_vec=clf.feature_importances_.reshape(14,5)
    for vec in att_vec:
        attention_final.append((sum(vec)/5))

    predictions=clf.predict(X_test)

    #evaluation of results
    print("Evaluating model performances")
    TP, TN, FP, FN = 0, 0, 0, 0
    #compare predicted target tensor to real target tensor, counting true/false positive/negative examples
    for i in range(y_test.shape[0]):
        if y_test[i] >= 0.5:
            if predictions[i] >= 0.5:
                TP += 1
            else:
                FN += 1
        else:
            if predictions[i] >= 0.5:
                FP += 1
            else:
                TN += 1
    #calculate evaluation metrics
    precision = None
    recall = None
    f1score = None
    if TP+FP != 0:
        precision = float(TP)/float(TP+FP)
    if TP+FN != 0:
        recall = float(TP)/float(TP+FN)
    if precision is not None and recall is not None:
        f1score = 2*precision*recall / (precision + recall)
    accuracy = float(TP + TN) / float(TP + TN + FP + FN)

    #terminate execution
    print("Precision : "+str(precision))
    print("Recall : "+str(recall))
    print("F1-score : "+str(f1score))
    print("Accuracy : "+str(accuracy))

    attent_norm=[round(x/sum(attention_final),3) for x in attention_final]
    #plot average attention bar-chart
    print("Plotting average attention histogram")
    figure, axs = plt.subplots(1, 1)
    axs.set_xlabel("Sequence Spot")
    axs.set_ylabel("Attention")
    normalizer = colors.Normalize() 
    att_norm = normalizer(attent_norm)
    colours = plt.cm.inferno(att_norm)
    axs.bar(range(1, sequence_length+1), attent_norm, color=colours)
    figure.savefig('RFC_plot_w2v.pdf')
    plt.close(figure)