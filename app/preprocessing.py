from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from app.features import *
from app.load_data import parse_reviews


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

    return {'X_train': X_train_features,
            'X_test': X_test_features,
            'y_train': y_train,
            'y_test': y_test}
