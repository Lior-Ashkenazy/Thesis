import pandas as pd
from sklearn.model_selection import train_test_split

def dataExploration(df):
    data_ = df
    data_x = data_['proposal']
    data_y = data_['label']
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.15, random_state=42)
    df_train = pd.concat([X_train, y_train], join = 'outer', axis = 1)
    df_test = pd.concat([X_test, y_test], join = 'outer', axis = 1)
    # print(df_train)
    # print(df_test)
    return df_train, df_test