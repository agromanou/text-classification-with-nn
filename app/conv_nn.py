import os
import pickle

import numpy as np
from keras import callbacks
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.models import load_model as keras_load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model as keras_plot_model
from keras.utils.np_utils import to_categorical as keras_to_categorical
from sklearn.preprocessing import LabelEncoder

from app import MODELS_DIR
from app.load_data import parse_reviews
from app.word_embedding import GloveWordEmbedding


class StackedCNN:

    def __init__(self,
                 max_sequence_length=1000,
                 max_nb_words=20000,
                 embedding_dim=100,
                 validation_split=0.2,
                 loss='categorical_crossentropy',
                 optimizer='rmsprop',
                 batch_size=64,
                 epochs=10,
                 deep_activation='relu',
                 activation='sigmoid',
                 outfile='stacked_cnn',
                 plot_model=False,
                 load_model=False
                 ):
        """

        :param max_sequence_length:
        :param max_nb_words:
        :param embedding_dim:
        :param validation_split:
        :param loss:
        :param optimizer:
        :param batch_size:
        :param epochs:
        :param deep_activation:
        :param activation:
        :param outfile:
        :param plot_model:
        :param load_model:
        """

        self.max_sequence_length = max_sequence_length
        self.max_nb_words = max_nb_words
        self.embedding_dim = embedding_dim
        self.validation_split = validation_split
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.deep_activation = deep_activation
        self.activation = activation
        self.outfile = outfile
        self.plot_model = plot_model
        self.load_model = load_model

        self.X_train = None
        self.X_val = None
        self.X_test = None

        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.embedding_matrix = None
        self.word_index = None

        self.le = LabelEncoder()
        self.tokenizer = Tokenizer(num_words=self.max_nb_words)

        self.model = None

        self.load_trained_model()

    def load_trained_model(self):
        if self.load_model and self.outfile:
            try:
                model_path = os.path.join(MODELS_DIR, self.outfile + '.h5')
                self.model = keras_load_model(model_path)

                le_path = os.path.join(MODELS_DIR, "{}_le_classes.npy".format(self.outfile))
                self.le.classes_ = np.load(le_path)

                # loading tokenizer
                tokenizer_path = os.path.join(MODELS_DIR, "{}_tokenizer.pickle".format(self.outfile))
                with open(tokenizer_path, 'rb') as handle:
                    self.tokenizer = pickle.load(handle)


            except Exception:
                raise FileNotFoundError('Model Does not exist. Must train a model first.')

    def train_val_split(self, X, y, seed=200):
        """

        :param X:
        :param y:
        :param seed:
        :return:
        """

        np.random.seed(seed)

        # shuffling the training instances
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        train_data = X[indices]
        train_labels = y[indices]

        # calculating number of validation examples
        nb_validation_samples = int(self.validation_split * train_data.shape[0])

        # splitting in training and validation data
        x_train = train_data[:-nb_validation_samples]
        y_train = train_labels[:-nb_validation_samples]

        x_val = train_data[-nb_validation_samples:]
        y_val = train_labels[-nb_validation_samples:]

        print('Number of positive and negative reviews in training and validation set ')
        print(y_train.sum(axis=0))
        print(y_val.sum(axis=0))

        return {'x_train': x_train,
                'x_val': x_val,
                'y_train': y_train,
                'y_val': y_val}

    def prepare_word_embeddings(self):
        """

        :return:
        """
        word_index = self.tokenizer.word_index
        print('Found {} unique tokens.'.format(len(word_index)))

        # Instantiating the Glove embeddings
        gwe_obj = GloveWordEmbedding()
        embeddings_index = gwe_obj.get_word_embeddings(dimension=self.embedding_dim)
        print('Total {} word vectors in Glove 6B {}d.'.format(len(embeddings_index), self.embedding_dim))

        # constructing an embedding matrix
        embedding_matrix = np.random.random((len(word_index) + 1, self.embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        self.embedding_matrix = embedding_matrix
        self.word_index = word_index

    def preprocess_training_data(self, X_train, y_train, seed=200):
        """

        :param X_train:
        :param y_train:
        :param seed:
        :return:
        """

        y_train = y_train
        y_train_enc = self.le.fit_transform(y_train)

        train_labels = keras_to_categorical(y_train_enc)
        train_texts = list(X_train)

        self.tokenizer.fit_on_texts(train_texts)
        train_sequences = self.tokenizer.texts_to_sequences(train_texts)

        train_data = pad_sequences(train_sequences, maxlen=self.max_sequence_length)

        print('Shape of data tensor:', train_data.shape)
        print('Shape of label tensor:', train_labels.shape)

        data_splits = self.train_val_split(X=train_data, y=train_labels, seed=seed)

        x_train = data_splits['x_train']
        y_train = data_splits['y_train']

        x_val = data_splits['x_val']
        y_val = data_splits['y_val']

        self.X_train = x_train
        self.X_val = x_val

        self.y_train = y_train
        self.y_val = y_val

        return

    def preprocess_test_data(self, X_test, y_test):
        """

        :param X_test:
        :param y_test:
        :return:
        """

        y_test_enc = self.le.transform(y_test)
        y_test_hot = keras_to_categorical(y_test_enc)

        test_sequences = self.tokenizer.texts_to_sequences(list(X_test))
        test_padded_sequences = pad_sequences(test_sequences, maxlen=self.max_sequence_length)

        self.X_test = test_padded_sequences
        self.y_test = y_test_hot

        return

    def build_model(self):
        """

        :return:
        """
        # preparing word embeddings. (Creates self.word_index and self.embedding_matrix)
        self.prepare_word_embeddings()

        # Creating the Embedding layer using the predefined embedding matrix
        embedding_layer = Embedding(len(self.word_index) + 1,
                                    self.embedding_dim,
                                    weights=[self.embedding_matrix],
                                    input_length=self.max_sequence_length,
                                    trainable=True)

        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')

        embedded_sequences = embedding_layer(sequence_input)

        l_cov1 = Conv1D(128, 5, activation=self.deep_activation)(embedded_sequences)
        l_pool1 = MaxPooling1D(5)(l_cov1)
        l_cov2 = Conv1D(128, 5, activation=self.deep_activation)(l_pool1)
        l_pool2 = MaxPooling1D(5)(l_cov2)
        l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
        l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
        l_flat = Flatten()(l_pool3)
        l_dense = Dense(128, activation=self.deep_activation)(l_flat)
        preds = Dense(2, activation=self.activation)(l_dense)

        model = Model(sequence_input, preds)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['acc'])

        print("Model fitting - Simplified Convolutional Neural Network")
        print(model.summary(), end='\n\n')

        self.model = model

        return

    def fit(self, x_train, y_train):
        """

        :return:
        """
        # does all the preprocessing. Also splits in training and development data
        self.preprocess_training_data(X_train=x_train, y_train=y_train, seed=200)

        # construct model and set it to the constructor for easier access.
        self.build_model()

        print('Number of Epochs: {}, Batch Size: {}'.format(self.epochs, self.batch_size), end='\n\n')

        tbCallBack = callbacks.TensorBoard(log_dir='./Graph',
                                           histogram_freq=0,
                                           write_graph=True,
                                           write_images=True)

        history = self.model.fit(x=self.X_train,
                                 y=self.y_train,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(self.X_val, self.y_val),
                                 verbose=2,
                                 callbacks=[tbCallBack])

        if self.outfile:
            # saving keras model
            model_path = os.path.join(MODELS_DIR, self.outfile + '.h5')
            self.model.save(model_path)

            # saving label encoder
            le_path = os.path.join(MODELS_DIR, "{}_le_classes.npy".format(self.outfile))
            np.save(le_path, self.le.classes_)

            # saving keras tokenizer
            tokenizer_path = os.path.join(MODELS_DIR, "{}_tokenizer.pickle".format(self.outfile))

            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.outfile and self.plot_model:
            model_img_path = MODELS_DIR + self.outfile + '.png'
            keras_plot_model(self.model,
                             to_file=model_img_path,
                             show_shapes=True,
                             show_layer_names=True)

        return history

    def predict(self, X, y):
        """

        :return:
        """
        # preprocesses and sets self.X_test and self.y_test
        self.preprocess_test_data(X_test=X, y_test=y)

        scores = self.model.evaluate(x=self.X_test,
                                     y=self.y_test,
                                     batch_size=self.batch_size,
                                     verbose=2)

        predicted_classes = self.model.predict(self.X_test, batch_size=self.batch_size)

        print(predicted_classes)

        pred_scores = np.squeeze(predicted_classes)

        predicted_classes = list(map(lambda x: 1 if x > 0.5 else 0, list(pred_scores)))

        return {'scores': scores,
                'y_pred': predicted_classes,
                'y_pred_scores': pred_scores,
                'y_true': y}


if __name__ == "__main__":
    training_csv_df = parse_reviews(file_type='train', load_data=True)
    test_csv_df = parse_reviews(file_type='test', load_data=True)

    x_training = training_csv_df['text']
    y_training = training_csv_df['polarity']

    x_test = training_csv_df['text']
    y_test = training_csv_df['polarity']

    cnn_obj = StackedCNN(load_model=True, outfile='stacked_cnn')
    # cnn_obj.fit(x_training, y_training)
    meta = cnn_obj.predict(x_test.head(10), y_test.head(10))
    print(meta)
