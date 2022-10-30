#-------------------------Kaggle data from Kaggle site - https://www.kaggle.com/datasets/vetrirah/janatahack-independence-day-2020-ml-hackathon-------------------------

import pandas as pd

# df_KaggleData = pd.read_csv('Data Kaggle.csv')
# # # Examining the data balance for each category
# # print('Computer Science\n' ,df_KaggleData['Computer Science'].value_counts()) # The most balanced
# # print('Physics\n' ,df_KaggleData['Physics'].value_counts())
# # print('Mathematics\n' ,df_KaggleData['Mathematics'].value_counts())
# # print('Statistics\n' ,df_KaggleData['Statistics'].value_counts())
# # print('Quantitative Biology\n' ,df_KaggleData['Quantitative Biology'].value_counts())
# # print('Quantitative Finance\n' ,df_KaggleData['Quantitative Finance'].value_counts())

def get_KaggleData():
    df_KaggleData = pd.read_csv('Data Kaggle.csv')
    df_Kaggle = df_KaggleData.loc[:, ['ABSTRACT', 'Computer Science']]
    df_Kaggle.rename(columns = {'ABSTRACT':'proposal', 'Computer Science':'label'}, inplace = True)
    return df_Kaggle