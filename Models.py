
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def RandomForestClassifier_(features_train, labels_train, features_test):
    # best model for RandomForestClassifier - Grid Search
    params_RF = {'n_estimators': [100,150,200],'max_depth': [2,3,4]}
    GridSearchCV_RF = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=params_RF, cv= 5)
    best_model_RF = GridSearchCV_RF.fit(features_train, labels_train)
    # predict
    prob_preds_RF = best_model_RF.predict_proba(features_test)
    print(prob_preds_RF)
    # y_preds_RF = [1 if z[1]>0.5 else 0 for z in prob_preds_RF]
    # metrics
    # acc = accuracy_score(labels_test.values, y_preds_RF)
    # f1Score = f1_score(labels_test, y_preds_RF, average='weighted')
    return prob_preds_RF

def MultinomialNB_(features_train, labels_train, features_test):
    #best model for MultinomialNB - Grid Search
    params_NB = {'alpha':np.linspace(0.5, 1.5, 6),'fit_prior':[True, False]}
    GridSearchCV_NB = GridSearchCV(estimator=MultinomialNB(), param_grid=params_NB, cv= 5)
    best_model_NB = GridSearchCV_NB.fit(features_train, labels_train)
    # predict
    prob_preds_NB = best_model_NB.predict_proba(features_test)
    print(prob_preds_NB)
    # y_preds_NB = [1 if z[1]>0.5 else 0 for z in prob_preds_NB]
    # metrics
    # acc = accuracy_score(labels_test.values, y_preds_NB)
    # f1Score = f1_score(labels_test, y_preds_NB, average='weighted')
    return prob_preds_NB

def LogisticRegression_(features_train, labels_train, features_test):
    #best model for LogisticRegression - Grid Search
    params_LR = {'C': np.logspace(-4, 2, 10), 'class_weight': ['balanced', 'None']}
    GridSearchCV_LR = GridSearchCV(estimator=LogisticRegression(random_state=0), param_grid=params_LR, cv= 10)
    best_model_LR = GridSearchCV_LR.fit(features_train, labels_train)
    # predict
    prob_preds_LR = best_model_LR.predict_proba(features_test)
    print(prob_preds_LR)
    # y_preds_LR = [1 if z[1]>0.5 else 0 for z in prob_preds_LR]
    # metrics
    # acc = accuracy_score(labels_test.values, y_preds_LR)
    # f1Score = f1_score(labels_test.values, y_preds_LR, average='weighted')
    return prob_preds_LR