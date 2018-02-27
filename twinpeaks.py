from lstm_functions import train_model_on_file, use_model

txt_file = "twinpeaks.txt"
epochs = 250
batch_size = 64
name = "twinpeaks"

train_model_on_file(txt_file, name, epochs, batch_size, 'char', None)

"""
/twinpeaks-weights-05-1.7959-bigger.hdf5
/twinpeaks-weights-11-1.5275-bigger.hdf5
/twinpeaks-weights-19-1.3402-bigger.hdf5
/twinpeaks-weights-20-1.3289-bigger.hdf5
"""

#model_file = "twinpeaks-weights-19-1.3402-bigger.hdf5"
#use_model(txt_file, model_file)