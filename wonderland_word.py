from lstm_functions import train_model_on_file, use_model

txt_file = "wonderland.txt"
epochs = 250
batch_size = 64
name = "wonderland"

train_model_on_file(txt_file, name, epochs, batch_size, 'word', None)


"""
/wonderland-weights-18-1.4552-bigger.hdf5
/wonderland-weights-41-1.2619-bigger.hdf5
/wonderland-weights-43-1.2550-bigger.hdf5
/wonderland-weights-45-1.2402-bigger.hdf5
"""

#model_file = "wonderland-weights-41-1.2619-bigger.hdf5"
#use_model(txt_file, 'char', model_file)
#model_file = "wonderland-weights-00-6.8889-biggerword.hdf5"
#use_model(txt_file, 'word', model_file)