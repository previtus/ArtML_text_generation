# Larger LSTM Network to Generate Text for Alice in Wonderland
import numpy, sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import string

def load_data_charbased(txt_file):
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

def load_data_wordbased(txt_file):
    rawtext = open(txt_file,'r').read().split('\n')
    rawtext = ' '.join(rawtext)
    rawtext = [word.strip(string.punctuation) for word in rawtext.split()]
    rawtext = ' '.join(rawtext)
    rawtext = rawtext.replace('-', ' ')
    rawtext = ' '.join(rawtext.split())
    raw_text = rawtext.split()
    unique_words = sorted(list(set(raw_text)))
    n_vocab = len(unique_words)
    print("Total Vocab:", n_vocab)
    word_to_int = dict((w, i) for i, w in enumerate(unique_words))
    int_to_word = dict((i, w) for i, w in enumerate(unique_words))
    n_words = len(raw_text)
    print (n_words)

    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_words - seq_length):
        seq_in = raw_text[i: i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([word_to_int[word] for word in seq_in])
        dataY.append(word_to_int[seq_out])
    n_patterns = len(dataX)
    print('Total patterns:', n_patterns)

    X = numpy.reshape(dataX, (n_patterns, seq_length, 1)) / float(n_vocab)
    y = np_utils.to_categorical(dataY)

    return X, y, dataX, n_vocab, int_to_word

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

def train_model_on_file(txt_file, name, epochs, batch_size, type='char', continue_from_checkpoint=None):

    # load ascii text and covert to lowercase
    if type is 'char':
        X, y, _, _, _ = load_data_charbased(txt_file)
    elif type is 'word':
        X, y, _, _, _ = load_data_wordbased(txt_file)

    model = build_model(X, y)

    extra = type
    if continue_from_checkpoint is not None:
        print("Continuing training model from a checkpoint", continue_from_checkpoint)
        print("(Notice the loss (error) staring already around similar value.)")
        model.load_weights(continue_from_checkpoint)
        extra += "fromchkp"

    ###### TRAIN MODEL ###############################

    # define the checkpoint
    filepath=name+"-weights-{epoch:02d}-{loss:.4f}-bigger"+extra+".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model

    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

def use_model(txt_file, type='char', model_file = "weights-improvement-47-1.2219-bigger.hdf5"):
    if type is 'char':
        X, y, dataX, n_vocab, int_to_txt = load_data_charbased(txt_file)
    elif type is 'word':
        X, y, dataX, n_vocab, int_to_txt = load_data_wordbased(txt_file)

    model = build_model(X, y)

    model.load_weights(model_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    ###### USE MODEL ###############################

    # pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed:")

    space = ''
    if type is 'word':
        space = ' '

    print("\"", space.join([int_to_txt[value] for value in pattern]), "\"")

    # generate characters
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_txt[index]+space
        seq_in = [int_to_txt[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")
