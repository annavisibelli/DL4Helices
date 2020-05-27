# W2V encoding https://radimrehurek.com/gensim/models/word2vec.html

import gensim
import numpy as np
import pandas as pd

de=pd.read_csv("Data/3H.csv",sep=";",header=None).T
dn=pd.read_csv("Data/Not_Helices.csv",sep=";",header=None).T
de=de.append(dn,ignore_index=True)

sentences=de.values.tolist()
model = gensim.models.Word2Vec(min_count=1,size=5, window=10,workers=1) 

#Vocabulary creation
model.build_vocab(sentences)

#Model training
model.train(sentences,total_examples=model.corpus_count,epochs=5000)

#Save the model
model.save('./model5.txt')

#Reload the helices dataset
de=pd.read_csv("Data/3H.csv",sep=";",header=None).T.values
dataset=[]
for i in range(0,de.shape[0]):
    string=np.zeros(de.shape[1]*5)
    for j in range(0,de.shape[1]):
        string[j*5:(j+1)*5]=model[de[i][j]]
    dataset.append(string)
de=pd.DataFrame(dataset)
prova=(de.loc[0])
de.to_csv("Data/Helicesw2v.csv",index=False)

#Reload the Not-helices dataset
de=pd.read_csv("Data/Not_Helices.csv",sep=";",header=None).T.values
dataset=[]
for i in range(0,de.shape[0]):
    string=np.zeros(de.shape[1]*5)
    for j in range(0,de.shape[1]):
        string[j*5:(j+1)*5]=model[de[i][j]]
    dataset.append(string)
de=pd.DataFrame(dataset)
de.to_csv("Data/Not_Helicesw2v.csv",index=False)

#Reload the model
w2vec = gensim.models.Word2Vec.load("./model5.txt")
