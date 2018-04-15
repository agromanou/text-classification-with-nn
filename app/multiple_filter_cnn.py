from app.model import ModelNN
from app.load_data import parse_reviews
from app.word_embedding import GloveWordEmbedding
from app.evaluation import create_clf_report
from app.evaluation import plot_roc_curve, plot_precision_recall_curve

import itertools as it
import operator
import numpy as np

from keras import layers
from keras import models
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical


class MultipleFilterCNN(ModelNN):
    def __init__(self,
                 loss,
                 epochs=10,
                 batch_size=32,
                 activation='softmax',
                 deep_activation='relu',
                 learning_rate=0.001,
                 decay=1e-6,
                 momentum=0.9,
                 optimizer='adam',
                 kernel_regularization_params=('l2', 0.01),
                 dropout=None,
                 max_sequence_length=100,
                 max_nb_words=20000,
                 embedding_dim=100,
                 validation_split=0.2,
                 outfile=None,
                 plot_model=False,
                 load_model=False
                 ):
        """
        :param loss: str, the name of the loss function
        :param epochs: int, the number of epochs
        :param batch_size: int, the size of the batch
        :param activation: str, the name of the activation function
        :param deep_activation: str, the name of the activation function in hidden layers
        :param learning_rate: float, the learning rate
        :param decay: float, the decay
        :param momentum: float, momentum
        :param optimizer: str, the name of the optimizer
        :param kernel_regularization_params: tuple, the regularization params
        :param dropout: float, dropout probability
        :param max_sequence_length:
        :param max_nb_words:
        :param embedding_dim:
        :param validation_split: float, the percentage of the validation split
        :param outfile:
        :param plot_model: boolean, if charts should plotted
        :param load_model: boolean, if existing trained model is available
        """

        self.batch_size = batch_size
        self.deep_activation = deep_activation
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.kernel_regularization_params = kernel_regularization_params
        self.dropout = dropout

        ModelNN.__init__(self,
                         loss=loss,
                         optimizer=optimizer,
                         learning_rate=learning_rate,
                         decay=decay,
                         momentum=momentum,
                         kernel_regularization_params=kernel_regularization_params,
                         epochs=epochs,
                         batch_size=batch_size,
                         validation_size=validation_split,
                         outfile=outfile,
                         plot_model=plot_model,
                         load_model=load_model)

        # cnn related
        self.max_sequence_length = max_sequence_length
        self.max_nb_words = max_nb_words
        self.embedding_dim = embedding_dim

    def build_model(self, word_index, embedding_matrix):
        """
        Creates and compiles a cnn model with Keras deep learning library
        :param word_index:
        :param embedding_matrix:
        """
        # Creating the Embedding layer using the predefined embedding matrix
        embedding_layer = layers.Embedding(len(word_index) + 1,
                                           self.embedding_dim,
                                           weights=[embedding_matrix],
                                           input_length=self.max_sequence_length,
                                           trainable=True)

        # applying a more complex convolutional approach
        convs = []
        filter_sizes = [3, 4, 5]

        sequence_input = layers.Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        for fsz in filter_sizes:
            l_conv = layers.Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)
            l_pool = layers.MaxPooling1D(5)(l_conv)
            convs.append(l_pool)

        l_merge = layers.Merge(mode='concat', concat_axis=1)(convs)
        l_cov1 = layers.Conv1D(128, 5, activation=self.deep_activation)(l_merge)
        l_pool1 = layers.MaxPooling1D(5)(l_cov1)
        l_cov2 = layers.Conv1D(128, 5, activation=self.deep_activation)(l_pool1)
        l_pool2 = layers.MaxPooling1D(30)(l_cov2)
        l_flat = layers.Flatten()(l_pool2)
        l_dense = layers.Dense(128, activation=self.deep_activation)(l_flat)
        preds = layers.Dense(2, activation=self.activation)(l_dense)

        model = models.Model(sequence_input, preds)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['acc'])

        self.model = model


def prepare_data_for_conv_networks(max_sequence_length=1000,
                                   max_nb_words=20000,
                                   embedding_dim=200,
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


def run_parameter_tuning(max_sequence_length,
                         max_nb_words,
                         embedding_dim):
    """
    This function runs the hyper-parameter tuning and plots the learning curves for the best model found.
    :param max_sequence_length:
    :param max_nb_words:
    :param embedding_dim:
    """

    data_sets = prepare_data_for_conv_networks(max_sequence_length=max_sequence_length,
                                               max_nb_words=max_nb_words,
                                               embedding_dim=embedding_dim)
    x_train = data_sets['X_train']
    x_val = data_sets['X_val']
    y_train = data_sets['y_train']
    y_val = data_sets['y_val']
    embedding_matrix = data_sets['embedding_matrix']
    word_index = data_sets['word_index']

    y_test = data_sets['y_test']
    x_test = data_sets['X_test']

    params = {'max_sequence_length': [1000],
              'embedding_dim': [embedding_dim],
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

    average = dict()
    histories = dict()
    with open('results_multiple_cnn.txt', 'a') as f:
        for i in comb:
            mlp = MultipleFilterCNN(max_sequence_length=i[0],
                                    embedding_dim=i[1],
                                    loss=i[3],
                                    optimizer=i[2],
                                    deep_activation=i[4],
                                    activation=i[5])

            mlp.build_model(embedding_matrix=embedding_matrix,
                            word_index=word_index)

            history = mlp.fit(x_train=x_train,
                              y_train=y_train)

            average[tuple(i)] = np.mean(history.history['val_acc'][5:])
            histories[tuple(i)] = {'acc': history.history['acc'],
                                   'acc_val': history.history['val_acc']}

            print('-' * 30, 'END OF RUN', '-' * 30)

            for n in range(0, len(history.history['val_acc']), 10):
                f.write(str(n) + ', ' + str(history.history['val_acc'][n]) + '\n')

            break

    print(average)

    best_settings = max(average.items(), key=operator.itemgetter(1))[0]

    mlp_best = MultipleFilterCNN(max_sequence_length=best_settings[0],
                                 embedding_dim=best_settings[1],
                                 loss=best_settings[3],
                                 optimizer=best_settings[2],
                                 deep_activation=best_settings[4],
                                 activation=best_settings[5])

    mlp_best.build_model(embedding_matrix=embedding_matrix,
                         word_index=word_index)

    history = mlp_best.fit(x_train=x_train,
                           y_train=y_train)

    mlp_best.plot_model_metadata(history)


def best_mpl_model(max_sequence_length,
                   max_nb_words,
                   embedding_dim,
                   load_pre_trained=True):
    """
    Test the best model on test data and plot results
    :param max_sequence_length:
    :param max_nb_words:
    :param embedding_dim:
    :param load_pre_trained: boolean, check if a model is already pre_trained
    in order to skip training and perform evaluation
    """

    data_sets = prepare_data_for_conv_networks(max_sequence_length=max_sequence_length,
                                               max_nb_words=max_nb_words,
                                               embedding_dim=embedding_dim)
    x_train = data_sets['X_train']
    x_val = data_sets['X_val']
    y_train = data_sets['y_train']
    y_val = data_sets['y_val']
    embedding_matrix = data_sets['embedding_matrix']
    word_index = data_sets['word_index']

    y_test = data_sets['y_test']
    x_test = data_sets['X_test']

    cnn_best = MultipleFilterCNN(max_sequence_length=1000,
                                 embedding_dim=100,
                                 loss='categorical_crossentropy',
                                 optimizer='sgd',
                                 deep_activation='tanh',
                                 activation='softmax',
                                 plot_model=False,
                                 load_model=load_pre_trained)

    cnn_best.build_model(embedding_matrix=embedding_matrix,
                         word_index=word_index)

    if not load_pre_trained:
        cnn_best.fit(x_train=x_train, y_train=y_train)

    meta = cnn_best.predict(X=x_test, y=y_test)

    print("Loss: {}, Acc: {}".format(meta['scores'][0], meta['scores'][1]))

    y_true = meta['y_true']
    y_pred = meta['y_pred']
    y_pred_scores = meta['y_pred_scores']

    create_clf_report(y_true, y_pred, ['negative', 'positive'])
    plot_roc_curve(y_true, y_pred_scores, pos_label=1)
    plot_precision_recall_curve(y_true, y_pred_scores, pos_label=1)


if __name__ == '__main__':
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100

    run_parameter_tuning(MAX_SEQUENCE_LENGTH,
                         MAX_NB_WORDS,
                         EMBEDDING_DIM)

    # best_mpl_model(MAX_SEQUENCE_LENGTH,
    #                MAX_NB_WORDS,
    #                EMBEDDING_DIM)
