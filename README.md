# ArtML_text_generation
Simple implementation of the charbased LSTM used to generate text. Part of ML4Art / Art4ML class

# Installation:

- Set up the prerequisites: Python 3, Keras, Tensoflow, Numpy

  * install Python 3
  * `sudo apt-get install build-essential python-dev`
  * `sudo pip install tensorflow keras numpy`
 
# Use model:

  * train the model with `train_model_on_file(txt_file, name, epochs, batch_size)` (produces ".hdf5" files)
  * evaluate model with `use_model(txt_file, model_file)`
  * see codes in `wonderland.py` and `twinpeaks.py` for example usage
  

# Links:
Tutorial https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
