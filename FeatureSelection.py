from sklearn.feature_selection import SelectKBest, f_regression
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV


def FeatureSelectionUsingCorrelation(data, correlation):
    #x_train, x_val, y, y_val = train_test_split(data_to_split, label_y, random_state=1, test_size=0.1, shuffle=True)
    cor = data.corr()
    columnsNamesArr = data.columns.values
    cor_target = abs(cor["time_to_eruption"])
    ImportantFeatures = cor_target[cor_target > correlation]
    ColumnsNamesReturned=[]
    '''for i in range(len(columnsNamesArr)):
        if(cor_target[i] in cor_target[cor_target>correlation]):
            ColumnsNamesReturned.append(columnsNamesArr[i])'''
    print(ImportantFeatures.index)
    newDataset=data[ImportantFeatures.index].copy()
    newDataset=newDataset.drop("time_to_eruption", 1)
    return newDataset
def FeatureSelectionWrapper(data,threshold):
    #x_train, x_val, y, y_val = train_test_split(data_to_split, label_y, random_state=1, test_size=0.1, shuffle=True)
    X = data.drop("time_to_eruption", 1)  # Feature Matrix
    X = X.drop("segment_id", 1)
    y = data["time_to_eruption"]  # Target Variable
    cols = list(X.columns)
    pmax = 1
    X.astype('float64')
    X_1 = X[cols].astype('float64')
    X_1 = sm.add_constant(X)
    model = sm.OLS(y, X_1).fit()
    while (len(cols) > 0):
        p = []
        X_1 = X[cols].astype('float64')
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y,X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if (pmax > threshold):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    newDataset=X[selected_features_BE].copy()

    return selected_features_BE,newDataset

def FeatureSelectionEmbedded(data, threshold):
    X = data.drop("time_to_eruption", 1)  # Feature Matrix
    X = X.drop("segment_id", 1)
    y = data["time_to_eruption"]  # Target Variable

    X.astype('float64')
    model = LassoCV()
    model.fit(X, y)
    coef = pd.Series(model.coef_, index=X.columns)
    imp_coef = coef.sort_values()
    ImportantFeatures = imp_coef[imp_coef > threshold]
    newDataset = data[ImportantFeatures.index].copy()
    return newDataset

path_local = '/home/fervn98/PycharmProjects/DATASETCI'
