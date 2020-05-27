import pandas as pd

#Residue propensity values in 3H helices

aminoacids=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

#percentage concentration of each amino acid in the entire dataset
Perc_Conc_dataset=[8.7,1.2,5.8, 7.1, 4.1, 7, 2.4, 6, 5.8, 10.1, 1.7, 4, 4.4, 3.8, 5.3, 5.8, 5.1, 7, 1.3, 3.4]

df3H=pd.read_csv("./Data/3H.csv",sep=";",header=None).T #Open the file 3H
length3H=4126 #Number of 3H helices
positions3H=['-3B','-2B','-1B','+1B','+2B','+3B','+4B','-4E','-3E','-2E','-1E','+1E','+2E','+3E']

freq_positions=[]
for i in range(len(df3H.columns)):
    freq_position=[]
    freq=df3H[i].value_counts().sort_index()
    for j in (aminoacids):
        if j in freq.index:
            freq_position.append(round((freq.loc[j]*100/length3H),1)) #Propensity values calculation
        else:
            freq_position.append(0)
    freq_positions.append(freq_position)
    
freq3H=pd.DataFrame(freq_positions).T
freq3H.index=aminoacids
freq3H.columns=positions3H
freq3H.insert(0, "%", Perc_Conc_dataset, True)
freq3H.to_csv("./Data/3H_freq.csv", sep=';') #Table creation


#Residue propensity values in 2H helices

df2H=pd.read_csv("./Data/2H.csv",sep=";",header=None).T #Open the file 2H
length2H=12763 #Number of 2H helices
positions2H=['-2B','-1B','+1B','+2B','+3B','+4B','-4E','-3E','-2E','-1E','+1E','+2E']

freq_positions=[]
for i in range(len(df2H.columns)):
    freq_position=[]
    freq=df2H[i].value_counts().sort_index()
    for j in (aminoacids):
        if j in freq.index:
            freq_position.append(round((freq.loc[j]*100/length2H),1)) #Propensity values calculation
        else:
            freq_position.append(0)
    freq_positions.append(freq_position)
    
freq2H=pd.DataFrame(freq_positions).T
freq2H.index=aminoacids
freq2H.columns=positions2H
freq2H.insert(0, "%", Perc_Conc_dataset, True)
freq2H.to_csv("./Data/2H_freq.csv", sep=';') #Table creation