import numpy as np
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from app.load_data import parse_reviews
from app.word_embedding import GloveWordEmbedding


def simpleCNN(max_sequence_length=100,
              max_nb_words=20000,
              embedding_dim=100,
              validation_split=0.2,
              loss='categorical_crossentropy',
              optimizer='rmsprop',
              batch_size=64,
              nb_epoch=10):
    """

    :param max_sequence_length:
    :param max_nb_words:
    :param embedding_dim:
    :param validation_split:
    :param loss:
    :param optimizer:
    :return:
    """

    training_csv_df = parse_reviews(file_type='train', load_data=False, save_data=False)
    test_csv_df = parse_reviews(file_type='test', load_data=False, save_data=False)

    mapper = {'positive': 1, 'negative': 0}

    train_texts = list(training_csv_df['text'])
    train_labels = list(training_csv_df['polarity'].map(mapper))

    test_texts = list(test_csv_df['text'])
    test_labels = list(test_csv_df['polarity'].map(mapper))

    tokenizer = Tokenizer(nb_words=max_nb_words)
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

    # Creating the Embedding layer using the predefined embedding matrix
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=True)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)
    l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(5)(l_cov2)
    l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
    l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
    l_flat = Flatten()(l_pool3)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(2, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    print("Model fitting - Simplified Convolutional Neural Network")
    print(model.summary())
    print()

    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_val, y_val),
                        nb_epoch=nb_epoch,
                        batch_size=batch_size)

    test_score = model.evaluate(x=test_data,
                                y=test_labels,
                                batch_size=batch_size,
                                verbose=2)

    return test_score


if __name__ == "__main__":
    test_accuracy = simpleCNN()

    print(test_accuracy)

#
#
# MAX_SEQUENCE_LENGTH = 1000
# MAX_NB_WORDS = 20000
# EMBEDDING_DIM = 100
# VALIDATION_SPLIT = 0.2
#
# data_train = parse_reviews(file_type='train', load_data=False, save_data=False)
#
# mapper = {'positive': 1, 'negative': 0}
# texts = list(data_train['text'])
# labels = list(data_train['polarity'].map(mapper))
#
# tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
#
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
#
# data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#
# labels = to_categorical(np.asarray(labels))
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)
#
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
# nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
#
# x_train = data[:-nb_validation_samples]
# y_train = labels[:-nb_validation_samples]
# x_val = data[-nb_validation_samples:]
# y_val = labels[-nb_validation_samples:]
#
# print('Number of positive and negative reviews in training and validation set ')
# print(y_train.sum(axis=0))
# print(y_val.sum(axis=0))
#
# gwe_obj = GloveWordEmbedding()
# embeddings_index = gwe_obj.get_word_embeddings(dimension=EMBEDDING_DIM)
# print('Total {} word vectors in Glove 6B {}d.'.format(len(embeddings_index), EMBEDDING_DIM))
#
# embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#
#
# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)
#
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
# l_pool1 = MaxPooling1D(5)(l_cov1)
# l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
# l_pool2 = MaxPooling1D(5)(l_cov2)
# l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
# l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
# l_flat = Flatten()(l_pool3)
# l_dense = Dense(128, activation='relu')(l_flat)
# preds = Dense(2, activation='softmax')(l_dense)
#
# model = Model(sequence_input, preds)
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['acc'])
#
# print("model fitting - simplified convolutional neural network")
# print(model.summary())
# print()
#
# model.fit(x_train,
#           y_train,
#           validation_data=(x_val, y_val),
#           nb_epoch=10,
#           batch_size=32)
#
#
# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)
#
# # applying a more complex convolutional approach
# convs = []
# filter_sizes = [3, 4, 5]
#
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
#
# for fsz in filter_sizes:
#     l_conv = Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)
#     l_pool = MaxPooling1D(5)(l_conv)
#     convs.append(l_pool)
#
# l_merge = Merge(mode='concat', concat_axis=1)(convs)
# l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
# l_pool1 = MaxPooling1D(5)(l_cov1)
# l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
# l_pool2 = MaxPooling1D(30)(l_cov2)
# l_flat = Flatten()(l_pool2)
# l_dense = Dense(128, activation='relu')(l_flat)
# preds = Dense(2, activation='softmax')(l_dense)
#
# model = Model(sequence_input, preds)
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['acc'])
#
# print("model fitting - more complex convolutional neural network")
# print(model.summary())
# model.fit(x_train,
#           y_train,
#           validation_data=(x_val, y_val),
#           nb_epoch=20, batch_size=50)
