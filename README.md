# A deep attention network for predicting amino acid signals in the formation of alpha-helices

This rep contains the code to reproduce the experiments described in the paper "A deep attention network for predicting amino acid signals in the formation of alpha-helices".

The pssp_lstm module contains an implementation of the LSTM RNN specified in Sonderby & Winther, 2015. See the README in that folder for more details and a user guide.

The lm_pretrain module allows users to train bidirectional language models that can be combined with bidirectional RNNs for protein secondary structure prediction. See the README in that folder for more details and a user guide.


# Requirements

This project is entirely build with Python (version>=3.3). The required Python modules are:

Pandas
Numpy
Scikit-learn
Keras
Tensorflow
Gensim

