import itertools
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

import SnapshotData
import KaggleData
import DeepDAOData
import DataExploration
import NLP_TFIDF
import Models
import BERTmodel
import GenerateVotes
import Sampling
import SystemEvaluation

# ----Choose Data----
# df = SnapshotData.get_SnapshotData() # Snapshot Data
# df = KaggleData.get_KaggleData() # Kaggle Data
# df = DeepDAOData.get_DeepDAOData() # DeepDAOD Data
#
#
# ----Data Exploration----
# df_train, df_test = DataExploration.dataExploration(df)
#
# df_train.to_csv('df_train_Snapshot.csv')
# df_test.to_csv('df_test_Snapshot.csv')
#
# df_train.to_csv('df_train_Kaggle.csv')
# df_test.to_csv('df_test_Kaggle.csv')
#
# df_train.to_csv('df_train_DeepDAO.csv')
# df_test.to_csv('df_test_DeepDAO.csv')

# # ----Read Dataset----
df_train = pd.read_csv('df_train_Snapshot.csv')
df_test = pd.read_csv('df_test_Snapshot.csv')

# df_train = pd.read_csv('df_train_Kaggle.csv')
# df_test = pd.read_csv('df_test_Kaggle.csv')
# df_train = df_train.loc[:len(df_train)/15,:]
# df_test = df_test.loc[:len(df_test)/15,:]
# print(len(df_train))
# print(len(df_test))

# # df_train = pd.read_csv('df_train_DeepDAO.csv')
# # df_test = pd.read_csv('df_test_DeepDAO.csv')
#
labels_test = df_test.label

# # ----Generate votes----
voters_number = 100
votes_forAll_proposals = GenerateVotes.get_votes(labels_test, voters_number)
# print(votes_forAll_proposals)
#
# # # Save votes
# # file = open("Votes_Snapshot.txt", "w+")
# # content = str(votes_forAll_proposals)
# # file.write(content)
# # file.close()
#
# # file = open("Votes_Kaggle.txt", "w+")
# # content = str(votes_forAll_proposals)
# # file.write(content)
# # file.close()
#
# # file = open("Votes_DeepDAO.txt", "w+")
# # content = str(votes_forAll_proposals)
# # file.write(content)
# # file.close()
#
#
# # # # # # ----Simple NLP Module----
# features_train, labels_train, features_test, labels_test = NLP_TFIDF.tfidfModel(df_train, df_test)
#
# ## !!!!Only for Snapshot - SMOTE (oversampling for imbalanced data
# sm = SMOTE(random_state=0)
# features_train, labels_train = sm.fit_resample(features_train, labels_train)
#
#
# # # --Random Forest Classifier--
# print('Random Forest Classifier')
# probs_RF = Models.RandomForestClassifier_(features_train, labels_train, features_test)
# # Save results
# probs_RF_df = pd.DataFrame(probs_RF)
# # #
# probs_RF_df.to_csv('Probs_Snapshot_RF_Smote.csv')
# # probs_RF_df.to_csv('Probs_Kaggle_RF.csv')
# # probs_RF_df.to_csv('Probs_DeepDAO_RF.csv')
#
# # --Multinomial Naive Bayes--
# print('Multinomial Naive Bayes')
# probs_NB = Models.MultinomialNB_(features_train, labels_train, features_test)
# print(probs_NB)
# # Save results
# probs_NB_df = pd.DataFrame(probs_NB)
# #
# probs_NB_df.to_csv('Probs_Snapshot_NB_Smote.csv')
# # probs_NB_df.to_csv('Probs_Kaggle_NB.csv')
# # probs_NB_df.to_csv('Probs_DeepDAO_NB.csv')
#
# # # --Logistic Regression--
# print('Logistic Regression')
# probs_LR = Models.LogisticRegression_(features_train, labels_train, features_test)
# print(probs_LR)
# # Save results
# probs_LR_df = pd.DataFrame(probs_LR)
#
# probs_LR_df.to_csv('Probs_Snapshot_LR_Smote.csv')
# # probs_LR_df.to_csv('Probs_Kaggle_LR.csv')
# # # probs_LR_df.to_csv('Probs_DeepDAO_LR.csv')


# # ----Language Model NLP Module----
# probs_BERT = BERTmodel.BERTmodel(df_train, df_test)
# print(probs_BERT)
# # Save results
# probs_BERT_df = pd.DataFrame(probs_BERT)
#
# # # probs_BERT_df.to_csv('Probs_Snapshot_BERT.csv')
# probs_BERT_df.to_csv('Probs_Kaggle_BERT.csv')
# # probs_BERT_df.to_csv('Probs_DeepDAO_BERT.csv')


# ----Sampling Module----
# Read votes
# file = open("Votes_Snapshot.txt", "r")
# votes = file.read().split() #puts the file into an array
# file.close()
# print(type(votes))

# file = open("Votes_Kaggle.txt", "r")
# votes = file.read()
# file.close()
#
# file = open("Votes_DeepDAO.txt", "r")
# votes = file.read()
# file.close()


# # Choosing probabilities
# probs_classes = pd.read_csv('Probs_Snapshot_RF_Smote.csv').to_numpy()
# probs_classes = pd.read_csv('Probs_Snapshot_NB_Smote.csv').to_numpy()
# probs_classes = pd.read_csv('Probs_Snapshot_LR_Smote.csv').to_numpy()
# probs_classes = pd.read_csv('Probs_Snapshot_BERT.csv').to_numpy()
# probs_classes = pd.read_csv('Probs_Kaggle_RF.csv').to_numpy()
# probs_classes = pd.read_csv('Probs_Kaggle_NB.csv').to_numpy()
# probs_classes = pd.read_csv('Probs_Kaggle_LR.csv').to_numpy()
# probs_classes = pd.read_csv('Probs_Kaggle_BERT.csv').to_numpy()
# probs_classes = pd.read_csv('Probs_DeepDAO_RF.csv').to_numpy()
# probs_classes = pd.read_csv('Probs_DeepDAO_NB.csv').to_numpy()
# probs_classes = pd.read_csv('Probs_DeepDAO_LR.csv').to_numpy()
# probs_classes = pd.read_csv('Probs_DeepDAO_BERT.csv').to_numpy()

z_arr = [arr[1] for arr in probs_classes]
# print("z_arr\n",z_arr)
# print("z_arr size\n",len(z_arr))

# Find functions
votes = votes_forAll_proposals

average_attentionSize_fixed_arr = [0.1, 0.2, 0.3, 0.4, 0.5]

c_arr = []
while len(c_arr) < 100:  # optionals c
    c = np.round(np.random.uniform(0.01, 1, 1), 2)
    if c not in c_arr: c_arr.append(c[0])
c_arr.sort()
# print(c_arr)

d_arr = []
while len(d_arr) < 100:  # optionals d
    d = np.round(np.random.uniform(0.01, 1, 1), 2)
    if d not in d_arr: d_arr.append(d[0])
d_arr.sort()
# print(d_arr)

combinations_arr = []
for c in c_arr:
    arr = [c]
    print(arr)
    combinations = list(itertools.product(arr, d_arr))
    # print(combinations)
    # print(len(combinations))
    combinations_arr.append(combinations)

df_results = pd.DataFrame(columns=['average_attentionSize_fixed', 'c', 'd', 'quality'])
final_c_arr = []
final_d_arr = []
final_quality_arr = []

for average_attentionSize_fixed in average_attentionSize_fixed_arr:
    c_arr = []
    d_arr = []
    average_quality_arr = []

    for combinations in combinations_arr:
        option_combination = Sampling.combination_search(combinations, z_arr, average_attentionSize_fixed)  # meets the requirements of the fixed average attentionSize fixed
        if option_combination != None:
            c = option_combination[0]
            d = option_combination[1]
            c_arr.append(c)
            d_arr.append(d)
            attentionSize_arr, average_attentionSize = Sampling.get_average_attentionSize(c, d, z_arr)
            attention_label_arr_ALL = Sampling.get_attention_labels(attentionSize_arr, votes, voters_number)
            average_quality = SystemEvaluation.get_quality(labels_test, attention_label_arr_ALL)
            average_quality_arr.append(average_quality)  # list of all average_quality
    # print('average_quality_arr\n', average_quality_arr)
    # print('c_arr\n', c_arr)
    # print('d_arr\n', d_arr)
    max_quality = max(average_quality_arr)
    max_quality_index = average_quality_arr.index(max_quality)
    final_c_arr.append(c_arr[max_quality_index])
    final_d_arr.append(d_arr[max_quality_index])
    final_quality_arr.append(max_quality)

df_results['average_attentionSize_fixed'], df_results['c'], df_results['d'], df_results['quality'] = average_attentionSize_fixed_arr, final_c_arr, final_d_arr, final_quality_arr
print(df_results)

# # ----Results and Plots----
# def metrics_results(datasetName, metric):
#     # Read real labels
#     df_test = pd.read_csv('df_test_'+ datasetName + '.csv')
#     if datasetName == 'Kaggle':
#         df_test = df_test.loc[:len(df_test) / 15, :]
#     labels_test = df_test.label.to_numpy()
#
#     # Read probabilities
#     Probs_datasetName_RF = pd.read_csv('Probs_'+ datasetName + '_RF.csv').to_numpy()
#     Probs_datasetName_NB = pd.read_csv('Probs_'+ datasetName + '_NB.csv').to_numpy()
#     Probs_datasetName_LR = pd.read_csv('Probs_'+ datasetName + '_LR.csv').to_numpy()
#     Probs_datasetName_BERT = pd.read_csv('Probs_'+ datasetName + '_BERT.csv').to_numpy()
#
#     # From probabilities to predictions
#     Preds_datasetName_RF = SystemEvaluation.predictions(Probs_datasetName_RF)
#     Preds_datasetName_NB = SystemEvaluation.predictions(Probs_datasetName_NB)
#     Preds_datasetName_LR = SystemEvaluation.predictions(Probs_datasetName_LR)
#     Preds_datasetName_BERT = SystemEvaluation.predictions(Probs_datasetName_BERT)
#     # print(Preds_datasetName_RF)
#     # print(Preds_datasetName_NB)
#     # print(Preds_datasetName_LR)
#     # print(Preds_datasetName_BERT)
#
#     # metric values
#     if metric == 'accuracy':
#         Acc_datasetName_RF = SystemEvaluation.accuracy(labels_test, Preds_datasetName_RF)
#         Acc_datasetName_NB = SystemEvaluation.accuracy(labels_test, Preds_datasetName_NB)
#         Acc_datasetName_LR = SystemEvaluation.accuracy(labels_test, Preds_datasetName_LR)
#         Acc_datasetName_BERT = SystemEvaluation.accuracy(labels_test, Preds_datasetName_BERT)
#         print('Acc_'+datasetName+'_RF', round(Acc_datasetName_RF,4))
#         print('Acc_'+datasetName+'_NB', round(Acc_datasetName_NB,4))
#         print('Acc_'+datasetName+'_LR', round(Acc_datasetName_LR,4))
#         print('Acc_'+datasetName+'_BERT', round(Acc_datasetName_BERT,4))
#
#         # SystemEvaluation.barPlot("Models", ["Random Forest", "Multinominal NB", "Logistic Regression", "BERT"],
#         #                          "Accuracy", [Acc_datasetName_RF, Acc_datasetName_NB, Acc_datasetName_LR, Acc_datasetName_BERT],
#         #                          "Models Accuracy - " + datasetName + " Data")
#     elif metric == 'f1 score':
#         f1_datasetName_RF = SystemEvaluation.f1Score(labels_test, Preds_datasetName_RF)
#         f1_datasetName_NB = SystemEvaluation.f1Score(labels_test, Preds_datasetName_NB)
#         f1_datasetName_LR = SystemEvaluation.f1Score(labels_test, Preds_datasetName_LR)
#         f1_datasetName_BERT = SystemEvaluation.f1Score(labels_test, Preds_datasetName_BERT)
#         print('f1_'+datasetName+'_RF', round(f1_datasetName_RF,4))
#         print('f1_'+datasetName+'_NB', round(f1_datasetName_NB,4))
#         print('f1_'+datasetName+'_LR', round(f1_datasetName_LR,4))
#         print('f1_'+datasetName+'_BERT', round(f1_datasetName_BERT,4))
#
#         # SystemEvaluation.barPlot("Models", ["Random Forest", "Multinominal NB", "Logistic Regression", "BERT"],
#         #                          "F1 Score",
#         #                          [f1_datasetName_RF, f1_datasetName_NB, f1_datasetName_LR, f1_datasetName_BERT],
#         #                          "Models F1 Score - " + datasetName + " Data")
#
#     elif metric == 'precision':
#         precision_datasetName_RF = SystemEvaluation.precision(labels_test, Preds_datasetName_RF)
#         precision_datasetName_NB = SystemEvaluation.precision(labels_test, Preds_datasetName_NB)
#         precision_datasetName_LR = SystemEvaluation.precision(labels_test, Preds_datasetName_LR)
#         precision_datasetName_BERT = SystemEvaluation.precision(labels_test, Preds_datasetName_BERT)
#         print('precision_'+datasetName+'_RF', round(precision_datasetName_RF,4))
#         print('precision_'+datasetName+'_NB', round(precision_datasetName_NB,4))
#         print('precision_'+datasetName+'_LR', round(precision_datasetName_LR,4))
#         print('precision_'+datasetName+'_BERT', round(precision_datasetName_BERT,4))
#
#         # SystemEvaluation.barPlot("Models", ["Random Forest", "Multinominal NB", "Logistic Regression", "BERT"],
#         #                          "Precision Score",
#         #                          [precision_datasetName_RF, precision_datasetName_NB, precision_datasetName_LR, precision_datasetName_BERT],
#         #                          "Models Precision Score - " + datasetName + " Data")
#
#     elif metric == 'recall':
#         recall_datasetName_RF = SystemEvaluation.recall(labels_test, Preds_datasetName_RF)
#         recall_datasetName_NB = SystemEvaluation.recall(labels_test, Preds_datasetName_NB)
#         recall_datasetName_LR = SystemEvaluation.recall(labels_test, Preds_datasetName_LR)
#         recall_datasetName_BERT = SystemEvaluation.recall(labels_test, Preds_datasetName_BERT)
#         print('recall_'+datasetName+'_RF', round(recall_datasetName_RF,4))
#         print('recall_'+datasetName+'_NB', round(recall_datasetName_NB,4))
#         print('recall_'+datasetName+'_LR', round(recall_datasetName_LR,4))
#         print('recall_'+datasetName+'_BERT', round(recall_datasetName_BERT,4))
#
#         # SystemEvaluation.barPlot("Models", ["Random Forest", "Multinominal NB", "Logistic Regression", "BERT"],
#         #                          "Recall Score",
#         #                          [recall_datasetName_RF, recall_datasetName_NB, recall_datasetName_LR, recall_datasetName_BERT],
#         #                          "Models Recall Score - " + datasetName + " Data")
#

# # --Accuracy bar plot for Kaggle data--
# metrics_results('Kaggle', 'accuracy')
#
# # --Accuracy bar plot for DeepDAO data--
# metrics_results('DeepDAO', 'accuracy')
#
# # F1-score bar plot for Snapshot data
# metrics_results('Snapshot', 'f1 score')

# # Kaggle
# metrics_results('Kaggle', 'accuracy')
# metrics_results('Kaggle', 'f1 score')
# metrics_results('Kaggle', 'precision')
# metrics_results('Kaggle', 'recall')

# # Snapshot
# metrics_results('Snapshot', 'accuracy')
# metrics_results('Snapshot', 'f1 score')
# metrics_results('Snapshot', 'precision')
# metrics_results('Snapshot', 'recall')

# # DeepDAO
# metrics_results('DeepDAO', 'accuracy')
# metrics_results('DeepDAO', 'f1 score')
# metrics_results('DeepDAO', 'precision')
# metrics_results('DeepDAO', 'recall')


# # Quality plot
# SystemEvaluation.quality_plot(['0.1','0.2','0.3','0.4','0.5'], ['0.2','0.4','0.6','0.8','1.0'], "Random Forest",
#                               [0.9160, 0.9320, 0.9480, 0.9507, 0.9547], [0.8747, 0.8960, 0.9173, 0.9240, 0.9307],
#                               [0.8747, 0.8960, 0.9173, 0.9240, 0.9307], 'Kaggle', 'Snapshot', 'DeepDAO')


