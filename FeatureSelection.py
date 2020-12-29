from sklearn.feature_selection import SelectKBest, f_regression
import statsmodels.api as sm
import pandas as pd
import numpy as np

def FeatureSelectionUsingCorrelation(data, correlation):
    #x_train, x_val, y, y_val = train_test_split(data_to_split, label_y, random_state=1, test_size=0.1, shuffle=True)
    cor = data.corr()
    columnsNamesArr = data.columns.values
    print(cor)
    cor_target = abs(cor["time_to_eruption"])
    ImportantFeatures = cor_target[cor_target > correlation]
    ColumnsNamesReturned=[]
    '''for i in range(len(columnsNamesArr)):
        if(cor_target[i] in cor_target[cor_target>correlation]):
            ColumnsNamesReturned.append(columnsNamesArr[i])'''
    print(ImportantFeatures)
    return ImportantFeatures
def FeatureSelectionWrapper(data):
    #x_train, x_val, y, y_val = train_test_split(data_to_split, label_y, random_state=1, test_size=0.1, shuffle=True)
    X = data.drop("time_to_eruption", 1)  # Feature Matrix
    X = X.drop("segment_id", 1)
    y = data["time_to_eruption"]  # Target Variable
    print(X.head())
    cols = list(X.columns)
    pmax = 1
    X.astype('float64')
    X_1 = X[cols].astype('float64')
    X_1 = sm.add_constant(X)
    model = sm.OLS(y, X_1).fit()
    print(model.pvalues)
    while (len(cols) > 0):
        p = []
        X_1 = X[cols].astype('float64')
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y,X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if (pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    print(len(selected_features_BE))

    return 0



path_local = '/home/fervn98/PycharmProjects/DATASETCI'