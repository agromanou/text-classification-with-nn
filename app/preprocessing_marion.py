from app.features import *
from app.load_data import parse_reviews
from bs4 import BeautifulSoup

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model

def load():
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

    # # Lemmatizing all X's.
    # X_train_lemmatized = pd.DataFrame(LemmaExtractor(col_name='text').fit_transform(X_train))
    # X_test_lemmatized = pd.DataFrame(LemmaExtractor(col_name='text').fit_transform(X_test))
    #
    # # Setting the best vector based features with their parameters
    # vect_based_features = Pipeline([('extract', TextColumnExtractor(column='text')),
    #                                 ('contractions', ContractionsExpander()),
    #                                 ('vect', CountVectorizer(binary=True,
    #                                                          min_df=0.01,
    #                                                          max_features=None,
    #                                                          ngram_range=(1, 1),
    #                                                          stop_words=None)),
    #                                 ('tfidf', TfidfTransformer(norm='l2', use_idf=False)),
    #                                 ('to_dense', DenseTransformer())])
    #
    # # Setting the best parameters for the user based features
    # user_based_features = FeatureUnion(transformer_list=[
    #     ('text_length', TextLengthExtractor(col_name='text', reshape=True)),
    #     ('avg_token_length', WordLengthMetricsExtractor(col_name='text', metric='avg', split_type='thorough')),
    #     ('std_token_length', WordLengthMetricsExtractor(col_name='text', metric='std', split_type='thorough')),
    #     ('contains_spc', ContainsSpecialCharactersExtractor(col_name='text')),
    #     ('n_tokens', NumberOfTokensCalculator(col_name='text')),
    #     ('contains_dots_bool', ContainsSequentialChars(col_name='text', pattern='..')),
    #     ('contains_excl_bool', ContainsSequentialChars(col_name='text', pattern='!!')),
    #     ('sentiment_positive', HasSentimentWordsExtractor(col_name='text', sentiment='positive', count_type='counts')),
    #     ('sentiment_negative', HasSentimentWordsExtractor(col_name='text', sentiment='negative', count_type='boolean')),
    #     ('contains_uppercase', ContainsUppercaseWords(col_name='text', how='count'))])
    #
    # # we also need the pipeline without the clf in order to feed it to different classifiers
    # final_pipeline_without_clf = Pipeline([
    #     ('features', FeatureUnion(transformer_list=[
    #         ('vect_based_feat', vect_based_features),
    #         ('user_based_feat', user_based_features)])),
    #     ('scaling', StandardScaler())])
    #
    # X_train_features = final_pipeline_without_clf.fit_transform(X_train_lemmatized)
    # X_test_features = final_pipeline_without_clf.transform(X_test_lemmatized)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_enc': y_train_enc,
        'y_test_enc': y_test_enc
    }


if __name__ == "__main__":
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    datasets = load()

    print(datasets.keys())

    texts = []

    for idx in range(datasets['X_train'].shape[0]):
        text = datasets['X_train'].iloc[[idx], [0]]
        texts.append(text)

    # texts = datasets['X_train']['text'].tolist()
    labels = datasets['y_train_enc']
    labels = labels.tolist()

    labels = sum(labels, [])

    print(texts)

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    print(sequences)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))


