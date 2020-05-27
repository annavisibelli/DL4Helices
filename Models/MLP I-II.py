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

#target is [0 1] for helix and [1 0] for non-helix
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

for run in range(20): #multiple run
    size_1h = 20  #size of one-hot amino acid encoding
    size_w2v = 5  #size of word2vec amino acid encoding 
    classes = 2 #number of classes (helix / non-helix)
    test_split = 0.1 #fraction of examples for the test set
    splitting_seed = seeds[run]
    sequence_length = 14 #length of the sequences in the dataset
    run_string =  "run_"+str(run) #string to discriminate results of different runs

    #hyperparameters
    encoding = '1h' #can be '1h' (one hot) or 'w2v' (word2vec)
    epochs = 1000  #number of training epochs
    slr = 0.001 #starting learning rate
    hu_dense_w2v = 135 #hidden units in the dense layer 
    hu_dense_1h = 135 #hidden units in the dense layer 
    batch_size = 5000   #size of training batch
    class_weights = [1, 1]

    #set visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    #build paths for results
    fold_results_dir = "Results/post_"+encoding+"/fold_by_fold/"+run_string+"/"
    average_attention_path = "Results/post_"+encoding+"/average_attention/AVG_"+run_string+".png"
    heatmap_path = "Results/post_"+encoding+"/heatmaps/heatmap_"+run_string+".png"
    summary_path = "Results/post_"+encoding+"/summaries/"+run_string+".txt"

    #build directory system for run results
    if not os.path.exists("Results/post_"+encoding):
        os.makedirs("Results/post_"+encoding)
    if not os.path.exists("Results/post_"+encoding+"/average_attention"):
        os.makedirs("Results/post_"+encoding+"/average_attention")
    if not os.path.exists("Results/post_"+encoding+"/heatmaps"):
        os.makedirs("Results/post_"+encoding+"/heatmaps")
    if not os.path.exists("Results/post_"+encoding+"/summaries"):
        os.makedirs("Results/post_"+encoding+"/summaries")
    if not os.path.exists("Results/post_"+encoding+"/fold_by_fold"):
        os.makedirs("Results/post_"+encoding+"/fold_by_fold")
    if not os.path.exists("Results/post_"+encoding+"/fold_by_fold/"+run_string):
        os.makedirs("Results/post_"+encoding+"/fold_by_fold/"+run_string)

    #load data
    print("Loading input data")
    X_helix = None #helix input tensor
    X_nonhelix = None #non-helix input tensor
    size_encoding = None #size of amino acid encoding 

    #load data according to specified encoding
    if encoding == '1h':
        X_helix = pd.read_csv('../Data/HelicesOneHot.csv')
        X_helix["target"]=1
        X_nonhelix = pd.read_csv("../Data/Non_HelicesOneHot.csv")
        X_nonhelix["target"]=0
        size_encoding = size_1h
        hu_dense = hu_dense_1h
        input_dims=280

    elif encoding == 'w2v':
        X_helix = pd.read_csv('../Data/Helicesw2v.csv')
        X_helix["target"]=1
        X_nonhelix = pd.read_csv("../Data/Non_Helicesw2v.csv")
        X_nonhelix["target"]=0
        size_encoding = size_w2v
        hu_dense = hu_dense_w2v
        input_dims=70
    #detect errors in data loading (expectedly caused by wrong encoding specification)
    if X_helix is None or X_nonhelix is None:
        sys.exit("ERROR: could not load input data, check correctness of encoding specification")

    Train=X_helix.append(X_nonhelix,ignore_index=True)
    #definisco un vettore 'a' di indici, lo mischio e lo aggiungo al dataframe 'de'

    #shuffle only along first axis (standard shuffle)
    Train=Train.sample(random_state=int(splitting_seed), frac=1)

    Xy_train=Train.iloc[:-2400, :]
    X_test=Train.iloc[-2399:, :-1]
    y_test=Train.iloc[-2399:, -1:]

    y_test=to_categorical(y_test)

    #build the model
    print("Building MLP model")
    #build optimizer instance
    optimizer = optimizers.Adam(lr = slr)
    #build early stopper
    stopper = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto')

    #determine fold size for ten-fold crossvalidation
    fold_size = Xy_train.shape[0]/10
    fold_sizes = list()
    for i in range(10):
        fold_sizes.append(int(fold_size))
    #check that the fold sizes do not exceed total training set size and adjust them accordingly
    i=0
    while(sum(fold_sizes)>Xy_train.shape[0]):
        fold_sizes[i] = fold_sizes[i] - 1
        i = i +1
        if i>=10:
            i = 0 
    #check that the fold sizes sum up to the total training set size and adjust them accordingly
    i=0
    while(sum(fold_sizes)<Xy_train.shape[0]):
        fold_sizes[i] = fold_sizes[i] + 1
        i = i +1
        if i>=10:
            i = 0 

    #split training set into ten folds
    Xy_fold = list()
    start = 0
    for i in range(10):
        stop = start+fold_sizes[i]
        Xy_fold.append(Xy_train[start:stop])
        start = stop

    #declare global statistics
    GLOBAL_attention_vector_list = list()
    GLOBAL_TP = 0
    GLOBAL_TN = 0
    GLOBAL_FP = 0
    GLOBAL_FN = 0

    #start crossvalidation procedure
    print("Starting cross-validation procedure")
    for k in range(10):
    #build training and validation sets
        Xy_val = Xy_fold[k]
        Xy_tr = (pd.concat([Xy_fold[i]for i in range(10) if i != k ]))

        #split target and data tensors
        Xa = Xy_tr.iloc[:-2400, :-1].values
        ya = Xy_tr.iloc[:-2400, -1:].values
        Xb = Xy_val.iloc[-2399:, :-1].values
        yb = Xy_val.iloc[-2399:, -1:].values

        ya= to_categorical(ya)
        yb= to_categorical(yb)
        #delete extra tensors
        del Xy_val, Xy_tr

        #build the model

        layer_input  = layers.Input(shape=(input_dims,))

        layer_attention_scores = layers.Dense(input_dims, activation='softmax', name='attention_softmax')(layer_input)
        layer_multiplication = layers.multiply([layer_input, layer_attention_scores])
        layer_hidden = layers.Dense(units=hu_dense, activation='selu')(layer_multiplication)
        layer_output = layers.Dense(units=classes, activation='softmax')(layer_hidden)

        model = models.Model(inputs = layer_input, outputs = layer_output)
        model.compile(loss=losses.categorical_crossentropy ,optimizer = optimizer, metrics=['accuracy'])
        print(model.summary())
        #train the model
        print("Training the model")
        model.fit(Xa, ya, epochs=epochs, batch_size=batch_size, validation_data = ( Xb, yb ) , class_weight = class_weights, callbacks = [stopper], verbose=1)

        #test the model
        print("Test the model")
        predictions = model.predict(X_test, batch_size=batch_size, verbose = 1)

        #evaluation of results
        print("Evaluating model performances")
        TP, TN, FP, FN = 0, 0, 0, 0
        #compare predicted target tensor to real target tensor, counting true/false positive/negative examples
        for i in range(y_test.shape[0]):
            if y_test[i][1] >= 0.5:
                if predictions[i][1] >= 0.5:
                    TP += 1
                else:
                    FN += 1
            else:
                if predictions[i][1] >= 0.5:
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

        #build model to retrieve attention vectors
        print("Retrieving attention vectors")
        attention_monitor = models.Model(inputs = model.inputs, outputs = model.get_layer('attention_softmax').output)
        #run the monitor over the helix set
        attention_vectors = attention_monitor.predict(X_test, batch_size=batch_size, verbose = 1)
        #visualize attention
        attention_vector = np.mean(np.array(attention_vectors), axis=0)
        average_attention = np.mean(attention_vector, axis=0)

        attention_final=[]
        att_vec=attention_vector.reshape(14,5)
        for vec in att_vec:
            attention_final.append((sum(vec)/5))
            
        #add attention vector to global list of attention vectors
        GLOBAL_attention_vector_list.append(attention_final)
        #add local statistics to global statistics
        GLOBAL_TP = GLOBAL_TP + TP
        GLOBAL_TN = GLOBAL_TN + TN
        GLOBAL_FP = GLOBAL_FP + FP
        GLOBAL_FN = GLOBAL_FN + FN

        #print fold-by-fold results
        out_path = fold_results_dir+"f_"+str(k+1)+".txt"
        print("Printing results to "+out_path)
        out_file = open(out_path, 'w')
        out_file.write("HYPERPARAMETERS\n")
        out_file.write("Encoding : "+encoding+"\n")
        out_file.write("Dense hidden units : "+str(hu_dense)+"\n")
        out_file.write("Epochs : "+str(epochs)+"\n")
        out_file.write("Starting learning rate : "+str(slr)+"\n")
        out_file.write("Batch size : "+str(batch_size)+"\n")
        out_file.write("Class weights : "+str(class_weights)+"\n")
        out_file.write("\n")
        out_file.write("RESULTS\n")
        out_file.write("TP : "+str(TP)+"\n")
        out_file.write("TN : "+str(TN)+"\n")
        out_file.write("FP : "+str(FP)+"\n")
        out_file.write("FN : "+str(FN)+"\n")
        out_file.write("Precision : "+str(precision)+"\n")
        out_file.write("Recall : "+str(recall)+"\n")
        out_file.write("F1-score : "+str(f1score)+"\n")
        out_file.write("Accuracy : "+str(accuracy)+"\n")
        out_file.write("Average attention : "+str(average_attention))
        out_file.close()

        #plot average attention bar-chart
        print("Plotting average attention histogram")
        figure, axs = plt.subplots(1, 1)
        axs.set_xlabel("Sequence Spot")
        axs.set_ylabel("Attention")
        normalizer = colors.Normalize() 
        att_norm = normalizer(average_attention)
        colours = plt.cm.inferno(att_norm)
        bars = axs.bar(range(1, sequence_length+1), attention_final, color=colours)
        figure.savefig(fold_results_dir+"AVG_"+str(k+1)+".png")
        plt.close(figure)

    #compute average values of the attention vectors
    GLOBAL_attention_vector = np.mean(np.array(GLOBAL_attention_vector_list), axis=0)
    GLOBAL_average_attention = np.mean(GLOBAL_attention_vector, axis=0)
    #compute average values of the statistics over the ten folds
    GLOBAL_precision = None
    GLOBAL_recall = None
    GLOBAL_f1score = None
    if GLOBAL_TP+GLOBAL_FP != 0:
        GLOBAL_precision = float(GLOBAL_TP)/float(GLOBAL_TP+GLOBAL_FP)
    if GLOBAL_TP+GLOBAL_FN != 0:
        GLOBAL_recall = float(GLOBAL_TP)/float(GLOBAL_TP+GLOBAL_FN)
    if GLOBAL_precision is not None and GLOBAL_recall is not None:
        GLOBAL_f1score = 2*GLOBAL_precision*GLOBAL_recall / (GLOBAL_precision + GLOBAL_recall)
    GLOBAL_accuracy = float(GLOBAL_TP + GLOBAL_TN) / float(GLOBAL_TP + GLOBAL_TN + GLOBAL_FP + GLOBAL_FN)

    #print aggregated results
    print("Printing results to "+summary_path)
    out_file = open(summary_path, 'w')
    out_file.write("HYPERPARAMETERS\n")
    out_file.write("Encoding : "+encoding+"\n")
    out_file.write("Dense hidden units : "+str(hu_dense)+"\n")
    out_file.write("Epochs : "+str(epochs)+"\n")
    out_file.write("Starting learning rate : "+str(slr)+"\n")
    out_file.write("Batch size : "+str(batch_size)+"\n")
    out_file.write("Class weights : "+str(class_weights)+"\n")
    out_file.write("\n")
    out_file.write("RESULTS\n")
    out_file.write("TP : "+str(GLOBAL_TP)+"\n")
    out_file.write("TN : "+str(GLOBAL_TN)+"\n")
    out_file.write("FP : "+str(GLOBAL_FP)+"\n")
    out_file.write("FN : "+str(GLOBAL_FN)+"\n")
    out_file.write("Precision : "+str(GLOBAL_precision)+"\n")
    out_file.write("Recall : "+str(GLOBAL_recall)+"\n")
    out_file.write("F1-score : "+str(GLOBAL_f1score)+"\n")
    out_file.write("Accuracy : "+str(GLOBAL_accuracy)+"\n")
    out_file.write("Global average attention : "+str(GLOBAL_average_attention))
    out_file.close()

    #plot average attention bar-chart
    print("Plotting average attention histogram")
    figure, axs = plt.subplots(1, 1)
    axs.set_xlabel("Sequence Spot")
    axs.set_ylabel("Attention")
    normalizer = colors.Normalize() 
    att_norm = normalizer(GLOBAL_attention_vector)
    colours = plt.cm.inferno(att_norm)
    bars = axs.bar(range(1, sequence_length+1),  GLOBAL_attention_vector, color = colours)
    figure.savefig(average_attention_path)
    plt.close(figure)

    #terminate execution
    print("Average Precision : "+str(GLOBAL_precision))
    print("Average Recall : "+str(GLOBAL_recall))
    print("Average F1-score : "+str(GLOBAL_f1score))
    print("Average Accuracy : "+str(GLOBAL_accuracy))