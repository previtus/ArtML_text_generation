# Larger LSTM Network to Generate Text for Alice in Wonderland
import numpy, sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def load_data(txt_file):
    raw_text = open(txt_file).read()
    raw_text = raw_text.lower()
    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)
    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)
    return X, y, dataX, n_vocab, int_to_char

def build_model(X, y):
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def train_model_on_file(txt_file, name, epochs, batch_size, continue_from_checkpoint=None):

    # load ascii text and covert to lowercase
    X, y, _, _, _ = load_data(txt_file)

    model = build_model(X, y)

    extra = ""
    if continue_from_checkpoint is not None:
        print("Continuing training model from a checkpoint", continue_from_checkpoint)
        print("(Notice the loss (error) staring already around similar value.)")
        model.load_weights(continue_from_checkpoint)
        extra = "fromchkp"

    ###### TRAIN MODEL ###############################

    # define the checkpoint
    filepath=name+"-weights-{epoch:02d}-{loss:.4f}-bigger"+extra+".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model

    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

def use_model(txt_file, model_file = "weights-improvement-47-1.2219-bigger.hdf5"):
    X, y, dataX, n_vocab, int_to_char = load_data(txt_file)
    model = build_model(X, y)

    model.load_weights(model_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    ###### USE MODEL ###############################

    # pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    # generate characters
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")
