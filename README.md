# A deep attention network for predicting amino acid signals in the formation of α-helices

This rep contains the code to reproduce the experiments described in the paper "A deep attention network for predicting amino acid signals in the formation of α-helices".

# Requirements

This project is entirely build with Python (version>=3.3). The required Python modules are:

* Pandas
* Numpy
* Scikit-learn
* Keras
* Tensorflow
* Gensim
* Matplotlib
* Os
* Sys

# Project Structure

The three main protein classes have been downloaded from the CATH Database (http://www.cathdb.info/browse/tree). The extraction of sequences and secondary structure information from every PDB entry was generated by the Kabsch and Sander DSSP algorithm (https://swift.cmbi.ru.nl/gv/dssp/DSSP_3.html), from which we were able to extract all the helices present in proteins.

Since signals that trigger the helix formation can also be located outside the helix sequence itself, we analyzed the first and the last four aminoacids inside the helix, taking into account also two or three amino acids before and after each helix. We labeled the sequences with two or three external residues with the suffix 2H or 3H, respectively.
**Data** folder contains **3H.csv** and **2H.csv** files.

**Residue_propensity.py** evaluate the residue propensity value for each amino acid in the selected positions.

**3H.csv** and **Non-Helices.csv** contains data necessary to run the ML experiments.

**Encoding** folder contains the implementation of the Word2Vec (https://radimrehurek.com/gensim/models/word2vec.html) and One-Hot-encoding algorithms.

**Models** folder host all the model we have tested. Each model was setted to take as input the One-Hot encoded dataset. The Word2Vec encoded dataset can be used chancing " encoding='1h' " in " encoding='w2v' ".



