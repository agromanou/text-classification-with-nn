import numpy as np
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from app.load_data import parse_reviews
from app.word_embedding import GloveWordEmbedding

import matplotlib.pyplot as plt
import itertools as it
from pprint import pprint


def prepare_data_for_conv_networks(max_sequence_length=1000,
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


def simpleCNN(max_sequence_length=1000,
              max_nb_words=20000,
              embedding_dim=100,
              validation_split=0.2,
              loss='categorical_crossentropy',
              optimizer='rmsprop',
              batch_size=64,
              nb_epoch=10,
              deep_activation='relu',
              activation='sigmoid'):
    """

    :param max_sequence_length:
    :param max_nb_words:
    :param embedding_dim:
    :param validation_split:
    :param loss:
    :param optimizer:
    :param batch_size:
    :param nb_epoch:
    :return:
    """

    data_sets = prepare_data_for_conv_networks(max_sequence_length=max_sequence_length,
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

    # Creating the Embedding layer using the predefined embedding matrix
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=True)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)
    l_cov1 = Conv1D(128, 5, activation=deep_activation)(embedded_sequences)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation=deep_activation)(l_pool1)
    l_pool2 = MaxPooling1D(5)(l_cov2)
    # l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
    # l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(128, activation=deep_activation)(l_flat)
    preds = Dense(2, activation=activation)(l_dense)

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

    # test_score = model.evaluate(x=test_data,
    #                             y=test_labels,
    #                             batch_size=batch_size,
    #                             verbose=2)

    # #  "Accuracy"
    # # plt.subplot(1, 2, 1)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model Accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.ylim(ymax=1)
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    #
    # # "Loss"
    # # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')

    return history


def betterCNN(max_sequence_length=1000,
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
    :param batch_size:
    :param nb_epoch:
    :return:
    """

    data_sets = prepare_data_for_conv_networks(max_sequence_length=max_sequence_length,
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

    # Creating the Embedding layer using the predefined embedding matrix
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=True)

    # applying a more complex convolutional approach
    convs = []
    filter_sizes = [3, 4, 5]

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    for fsz in filter_sizes:
        l_conv = Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)

    l_merge = Merge(mode='concat', concat_axis=1)(convs)
    l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(2, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    print("Model fitting - A more complex Convolutional Neural Network")
    print(model.summary())
    print()

    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_val, y_val),
                        nb_epoch=nb_epoch,
                        batch_size=batch_size)

    # test_score = model.evaluate(x=test_data,
    #                             y=test_labels,
    #                             batch_size=batch_size,
    #                             verbose=2)

    #  "Accuracy"
    # plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim(ymax=1)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # "Loss"
    # plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    # plt.tight_layout()

    plt.show()

    return history


if __name__ == "__main__":
    params = {'max_sequence_length': [1000, 500],
              'embedding_dim': [50, 100, 200],
              'optimizer': ['rmsprop', 'adam', 'sgd'],
              'loss': ['categorical_crossentropy'],
              'deep_activation': ['relu', 'tanh'],
              'activation': ['softmax']}

    comb = it.product(params['max_sequence_length'],
                      params['embedding_dim'],
                      params['optimizer'],
                      params['loss'],
                      params['deep_activation'],
                      params['activation'])

    results = dict()
    with open('results_cnn.txt', 'a') as f:
        for i in comb:
            history = simpleCNN(max_sequence_length=i[0],
                                embedding_dim=i[1],
                                loss=i[3],
                                optimizer=i[2],
                                deep_activation=i[4],
                                activation=i[5])

            results[str(i)] = (history.history['acc'], history.history['val_acc'])
            print('-' * 30, 'END OF RUN', '-' * 30)
            f.write('{}, {}, {}\n'.format(str(i),
                                          'Accuracy: ' + str(history.history['acc']),
                                          'Accuracy Val : ' + str(history.history['val_acc'])))

    pprint(results)


