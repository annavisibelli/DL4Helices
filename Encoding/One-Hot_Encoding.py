from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import numpy as np
import pandas as pd

de=pd.read_csv("../Data/3H.csv",sep=";",header=None).T
dn=pd.read_csv("../Data/Non_Helices.csv",sep=";",header=None).T

#Helices One-Hot-Encoding
des=[]
one_hot=[]
voc=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
for i in range (0,len(de)):
    des.append(de.iloc[i].astype("category", categories=voc).cat.codes)
des=pd.DataFrame(des)    
des= array(des)
encoded = to_categorical(des)

enc=encoded.tolist()
for i in range(0,len(de)):
    j=0
    x=[]
    while j < 14:
        x=x+(enc[i])[j]
        j=j+1       
    z=np.array(x)
    one_hot.append(z)
    
one_hot=pd.DataFrame(one_hot)
one_hot.to_csv("../Data/HelicesOneHot.csv",index=False)

#Not_Helices One-Hot-Encoding
dns=[]
one_hot=[]
for i in range (0,len(dn)):
    dns.append(dn.iloc[i].astype("category", categories=voc).cat.codes)
dns=pd.DataFrame(dns)    
dns= array(dns)
encoded = to_categorical(dns)

enc=encoded.tolist()
for i in range(0,len(dn)):
    j=0
    x=[]
    while j < 14:
        x=x+(enc[i])[j]
        j=j+1       
    z=np.array(x)
    one_hot.append(z)
    
one_hot=pd.DataFrame(one_hot)
one_hot.to_csv("../Data/Non_HelicesOneHot.csv",index=False)