import numpy as np
from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer
from keras.layers import Dense, Input, Embedding, LSTM, GRU, Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from app.load_data import parse_reviews
from app.word_embedding import GloveWordEmbedding


class AttentionLayer(Layer):
    """Attention GRU network"""

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """

        self.init = initializers.get('normal')
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        assert len(input_shape) == 3

        self.Weights = self.init((input_shape[-1],))
        self.trainable_weights = [self.Weights]
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        """

        :param x:
        :param mask:
        :return:
        """
        eij = K.tanh(K.dot(x, self.Weights))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    @staticmethod
    def get_output_shape_for(input_shape):
        """

        :param input_shape:
        :return:
        """
        return input_shape[0], input_shape[-1]


def prepare_data_for_rnn_networks(max_sequence_length=1000,
                                  max_nb_words=20000,
                                  embedding_dim=100,
                                  validation_split=0.2):
    """

    :param max_sequence_length:
    :param max_nb_words:
    :param embedding_dim:
    :param validation_split:
    :return:
    """
    np.random.seed(200)

    training_csv_df = parse_reviews(file_type='train', load_data=False, save_data=False)
    test_csv_df = parse_reviews(file_type='test', load_data=False, save_data=False)

    mapper = {'positive': 1, 'negative': 0}

    train_texts = list(training_csv_df['text'])
    train_labels = list(training_csv_df['polarity'].map(mapper))

    test_texts = list(test_csv_df['text'])
    test_labels = list(test_csv_df['polarity'].map(mapper))

    tokenizer = Tokenizer(num_words=max_nb_words)
    tokenizer.fit_on_texts(train_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index)))

    train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
    test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)

    train_labels = to_categorical(np.asarray(train_labels))
    test_labels = to_categorical(np.asarray(test_labels))

    print('Shape of data tensor:', train_data.shape)
    print('Shape of label tensor:', train_labels.shape)

    # shuffling the training instances
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    # calculating number of validation examples
    nb_validation_samples = int(validation_split * train_data.shape[0])

    # splitting in training and validation data
    x_train = train_data[:-nb_validation_samples]
    y_train = train_labels[:-nb_validation_samples]
    x_val = train_data[-nb_validation_samples:]
    y_val = train_labels[-nb_validation_samples:]

    print('Number of positive and negative reviews in training and validation set ')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    # Instantiating the Glove embeddings
    gwe_obj = GloveWordEmbedding()
    embeddings_index = gwe_obj.get_word_embeddings(dimension=embedding_dim)
    print('Total {} word vectors in Glove 6B {}d.'.format(len(embeddings_index), embedding_dim))

    # constructing an embedding matrix
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return {
        'X_train': x_train,
        'X_val': x_val,
        'X_test': test_data,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': test_labels,
        'embedding_matrix': embedding_matrix,
        'word_index': word_index
    }


def bidirectional_lstm(max_sequence_length=1000,
                       max_nb_words=20000,
                       embedding_dim=100,
                       validation_split=0.2,
                       loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       batch_size=32,
                       nb_epoch=20,
                       activation='softmax'):
    """

    :param max_sequence_length:
    :param max_nb_words:
    :param embedding_dim:
    :param validation_split:
    :param loss:
    :param optimizer:
    :param batch_size:
    :param nb_epoch:
    :param activation:
    :return:
    """
    data_sets = prepare_data_for_rnn_networks(max_sequence_length=max_sequence_length,
                                              max_nb_words=max_nb_words,
                                              embedding_dim=embedding_dim,
                                              validation_split=validation_split)
    x_train = data_sets['X_train']
    x_val = data_sets['X_val']
    y_train = data_sets['y_train']
    y_val = data_sets['y_val']
    embedding_matrix = data_sets['embedding_matrix']
    word_index = data_sets['word_index']

    y_test = data_sets['y_test']
    x_test = data_sets['X_test']

    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=True)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
    preds = Dense(2, activation=activation)(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['acc'])

    print("Model Fitting - Bidirectional LSTM")
    print(model.summary())

    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_val, y_val),
                        nb_epoch=nb_epoch,
                        batch_size=batch_size)

    return history


def bidirectional_gru_with_attention_layer(max_sequence_length=1000,
                                           max_nb_words=20000,
                                           embedding_dim=100,
                                           validation_split=0.2,
                                           loss='categorical_crossentropy',
                                           optimizer='rmsprop',
                                           batch_size=50,
                                           nb_epoch=10,
                                           activation='softmax'):
    """

    :param max_sequence_length:
    :param max_nb_words:
    :param embedding_dim:
    :param validation_split:
    :param loss:
    :param optimizer:
    :param batch_size:
    :param nb_epoch:
    :param activation:
    :return:
    """

    data_sets = prepare_data_for_rnn_networks(max_sequence_length=max_sequence_length,
                                              max_nb_words=max_nb_words,
                                              embedding_dim=embedding_dim,
                                              validation_split=validation_split)
    x_train = data_sets['X_train']
    x_val = data_sets['X_val']
    y_train = data_sets['y_train']
    y_val = data_sets['y_val']
    embedding_matrix = data_sets['embedding_matrix']
    word_index = data_sets['word_index']

    y_test = data_sets['y_test']
    x_test = data_sets['X_test']

    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=True)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    # adding bidirectional GRU
    l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    # Adding attention layer
    l_att = AttentionLayer()(l_gru)
    # Adding output layer
    preds = Dense(2, activation=activation)(l_att)

    model = Model(sequence_input, preds)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['acc'])

    print("model fitting - Attention GRU network")
    print(model.summary())

    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_val, y_val),
                        nb_epoch=nb_epoch,
                        batch_size=batch_size)

    return history


if __name__ == "__main__":
    bidirectional_lstm()
