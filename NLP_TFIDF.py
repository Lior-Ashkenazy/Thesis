from sklearn.feature_extraction.text import TfidfVectorizer

def tfidfModel(df_train, df_test):
    #---Text Representation---
    tfidf = TfidfVectorizer(sublinear_tf=True, max_df=1.0, min_df=1, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    #for train
    features_train = tfidf.fit_transform(df_train['proposal'].values.astype('U')).toarray()
    labels_train = df_train.label
    #for test
    features_test = tfidf.transform(df_test['proposal'].values.astype('U')).toarray()
    labels_test = df_test.label
    return features_train, labels_train, features_test, labels_test