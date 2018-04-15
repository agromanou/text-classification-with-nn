from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from app.features import *
from app.load_data import parse_reviews

from keras import models
from keras import layers


def prepare_user_plus_vector_based_features():
    """

    :return:
    """
    # loading data (train and test)
    train_data = parse_reviews(load_data=False, file_type='train')
    test_data = parse_reviews(load_data=False, file_type='test')

    X_train = train_data.drop(['polarity'], axis=1)
    X_test = test_data.drop(['polarity'], axis=1)

    y_train = train_data['polarity']
    y_test = test_data['polarity']

    len_train = len(X_train)
    len_test = len(X_test)
    mapper = {'positive': 1, 'negative': 0}

    y_train_enc = y_train.map(mapper).values.reshape(len_train, 1)
    y_test_enc = y_test.map(mapper).values.reshape(len_test, 1)

    # Lemmatizing all X's.
    X_train_lemmatized = pd.DataFrame(LemmaExtractor(col_name='text').fit_transform(X_train))
    X_test_lemmatized = pd.DataFrame(LemmaExtractor(col_name='text').fit_transform(X_test))

    # Setting the best vector based features with their parameters
    vect_based_features = Pipeline([('extract', TextColumnExtractor(column='text')),
                                    ('contractions', ContractionsExpander()),
                                    ('vect', CountVectorizer(binary=True,
                                                             min_df=0.01,
                                                             max_features=None,
                                                             ngram_range=(1, 1),
                                                             stop_words=None)),
                                    ('tfidf', TfidfTransformer(norm='l2', use_idf=False)),
                                    ('to_dense', DenseTransformer())])

    # Setting the best parameters for the user based features
    user_based_features = FeatureUnion(transformer_list=[
        ('text_length', TextLengthExtractor(col_name='text', reshape=True)),
        ('avg_token_length', WordLengthMetricsExtractor(col_name='text', metric='avg', split_type='thorough')),
        ('std_token_length', WordLengthMetricsExtractor(col_name='text', metric='std', split_type='thorough')),
        ('contains_spc', ContainsSpecialCharactersExtractor(col_name='text')),
        ('n_tokens', NumberOfTokensCalculator(col_name='text')),
        ('contains_dots_bool', ContainsSequentialChars(col_name='text', pattern='..')),
        ('contains_excl_bool', ContainsSequentialChars(col_name='text', pattern='!!')),
        ('sentiment_positive', HasSentimentWordsExtractor(col_name='text', sentiment='positive', count_type='counts')),
        ('sentiment_negative', HasSentimentWordsExtractor(col_name='text', sentiment='negative', count_type='boolean')),
        ('contains_uppercase', ContainsUppercaseWords(col_name='text', how='count'))])

    # we also need the pipeline without the clf in order to feed it to different classifiers
    final_pipeline_without_clf = Pipeline([
        ('features', FeatureUnion(transformer_list=[
            ('vect_based_feat', vect_based_features),
            ('user_based_feat', user_based_features)])),
        ('scaling', StandardScaler())])

    X_train_features = final_pipeline_without_clf.fit_transform(X_train_lemmatized)
    X_test_features = final_pipeline_without_clf.transform(X_test_lemmatized)

    return {
        'X_train': X_train_features,
        'X_test': X_test_features,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_enc': y_train_enc,
        'y_test_enc': y_test_enc
    }


def prepare_embedding_based_features(emb_dim=100, emb_type='tfidf'):
    """

    :param emb_dim:
    :param emb_type:
    :return:
    """

    assert emb_type in ['tfidf', 'tf']

    # loading data (train and test)
    train_data = parse_reviews(load_data=False, file_type='train')
    test_data = parse_reviews(load_data=False, file_type='test')

    X_train = train_data.drop(['polarity'], axis=1)
    X_test = test_data.drop(['polarity'], axis=1)

    y_train = train_data['polarity']
    y_test = test_data['polarity']

    len_train = len(X_train)
    len_test = len(X_test)
    mapper = {'positive': 1, 'negative': 0}

    y_train_enc = y_train.map(mapper).values.reshape(len_train, 1)
    y_test_enc = y_test.map(mapper).values.reshape(len_test, 1)

    we_obj = GloveWordEmbedding()

    pre_loaded_we = {emb_dim: we_obj.get_word_embeddings(dimension=emb_dim)}

    print(X_test)

    final_pipeline = Pipeline([
        ('embedding_feat', SentenceEmbeddingExtractor(col_name='text',
                                                      word_embeddings_dict=pre_loaded_we,
                                                      embedding_type=emb_type,
                                                      embedding_dimensions=emb_dim)),
        ('scaling', StandardScaler())])

    X_train_features = final_pipeline.fit_transform(X_train)
    X_test_features = final_pipeline.transform(X_test)

    return {
        'X_train': X_train_features,
        'X_test': X_test_features,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_enc': y_train_enc,
        'y_test_enc': y_test_enc}


if __name__ == '__main__':
    # data = dict()
    data = prepare_user_plus_vector_based_features()

    x_train = data['X_train']
    y_train = data['y_train']

    x_test = data['X_test']
    y_test = data['y_test']

    print(type(x_train))
    print(x_train.shape)
    print(type(y_train))
    print(y_train.shape)

    print(type(x_test))
    print(x_test.shape)
    print(type(y_test))
    print(y_test.shape)
    print((x_train.shape[1],))

    # mlp = SimpleMLP()
    # mlp.fit(train_data=x_train, train_labels=y_train, epochs=3, batch_size=150)

    model = models.Sequential()
    model.add(layers.Dense(64,
                           activation='relu',
                           input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
