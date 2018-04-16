from app.model_old import Model
from keras import layers
from keras import models


class CNN(Model):
    def __init__(self,
                 train_data,
                 train_labels,
                 layers,
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
                 activity_regularization_params=None,
                 dropout=None,
                 max_sequence_length=100,
                 max_nb_words=20000,
                 embedding_dim=100,
                 validation_split=0.2,
                 ):

        self.layers = layers
        self.batch_size = batch_size
        self.layers_num = len(layers)
        self.deep_activation = deep_activation
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.kernel_regularization_params = kernel_regularization_params
        self.dropout = dropout

        self.train_data = train_data
        self.train_labels = train_labels

        Model.__init__(self,
                       optimizer,
                       learning_rate,
                       decay,
                       momentum,
                       kernel_regularization_params,
                       activity_regularization_params,
                       epochs,
                       batch_size)

        # cnn related
        self.max_sequence_length = max_sequence_length
        self.max_nb_words = max_nb_words
        self.embedding_dim = embedding_dim
        self.validation_split = validation_split

    def build_model(self):
        """

        :return:
        """
        model = models.Sequential()
        model.add(layers.Embedding(10000, 32))

        for filter in self.layers[1:]:
            if filter != self.layers[:-1]:
                model.add(layers.Conv1D(filters=filter, kernel_size=7, activation=self.deep_activation))
                model.add(layers.MaxPooling1D(5))
            else:
                model.add(layers.Conv1D(filters=filter, kernel_size=7, activation=self.deep_activation))
                model.add(layers.GlobalMaxPooling1D())

        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model
