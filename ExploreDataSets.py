import pandas as pd

# Read data and concate train and test
Kaggle_data = pd.concat([pd.read_csv('df_train_Kaggle.csv'),pd.read_csv('df_test_Kaggle.csv')], ignore_index=True)
Snapshot_data = pd.concat([pd.read_csv('df_train_Snapshot.csv'),pd.read_csv('df_test_Snapshot.csv')], ignore_index=True)
DeepDAO_data = pd.concat([pd.read_csv('df_train_DeepDAO.csv'),pd.read_csv('df_test_DeepDAO.csv')], ignore_index=True)


# Find the mean number of words in the proposals
def mean_sentences_len(data):
    sum_len = 0
    for sentence in data:
        if isinstance(sentence, str):
            sum_len += len(sentence.split())
    mean = int(sum_len/len(data))
    return mean

mean_Kaggle = mean_sentences_len(Kaggle_data['proposal'])
print(len(Kaggle_data))
print('mean_Kaggle', mean_Kaggle)

mean_Snapshot = mean_sentences_len(Snapshot_data['proposal'])
print(len(Snapshot_data))
print('mean_Snapshot', mean_Snapshot)

mean_DeepDAO = mean_sentences_len(DeepDAO_data['proposal'])
print(len(DeepDAO_data))
print('mean_DeepDAO', mean_DeepDAO)

