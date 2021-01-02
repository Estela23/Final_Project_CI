from sklearn import model_selection
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LassoCV


def FeatureSelectionUsingCorrelation(data, correlation):

    cor = data.corr()
    cor_target = abs(cor["time_to_eruption"])
    ImportantFeatures = cor_target[cor_target > correlation]
    newDataset = data[ImportantFeatures.index].copy()

    corr_train, corr_val, corr_y, corr_y_val = model_selection.train_test_split(
        newDataset[newDataset.columns[:-1]],
        newDataset[newDataset.columns[-1]],
        test_size=0.2, shuffle=True)

    test_set = pd.read_csv("data/test_final_data_complete.csv")
    test_set_reduced = test_set[ImportantFeatures.index[:-1]]

    return corr_train, corr_val, corr_y, corr_y_val, test_set_reduced


def FeatureSelectionWrapper(data, threshold):
    # x_train, x_val, y, y_val = train_test_split(data_to_split, label_y, random_state=1, test_size=0.1, shuffle=True)
    X_train = data.drop(["time_to_eruption"], 1)  # Feature Matrix
    y_train = data["time_to_eruption"]  # Target Variable
    cols = list(X_train.columns)
    while len(cols) > 0:
        X_1 = sm.add_constant(X_train[cols].astype('float64'))
        model = sm.OLS(y_train, X_1).fit()
        pvalues = pd.Series(model.pvalues.values[1:], index=cols)
        if max(pvalues) > threshold:
            cols.remove(pvalues.idxmax())
        else:
            break
    newDataset = data[cols+["time_to_eruption"]].copy()

    wrapp_train, wrapp_val, wrapp_y, wrapp_y_val = model_selection.train_test_split(
        newDataset[newDataset.columns[:-1]],
        newDataset[newDataset.columns[-1]],
        test_size=0.2, shuffle=True)

    test_set = pd.read_csv("data/test_final_data_complete.csv")
    test_set_reduced = test_set[cols]

    return wrapp_train, wrapp_val, wrapp_y, wrapp_y_val, test_set_reduced


def FeatureSelectionEmbedded(data, threshold):
    X = data.drop(["time_to_eruption", "segment_id"], 1)  # Feature Matrix
    y = data["time_to_eruption"]  # Target Variable

    X.astype('float64')
    model = LassoCV()
    model.fit(X, y)
    coef = pd.Series(model.coef_, index=X.columns)
    imp_coef = coef.sort_values()
    ImportantFeatures = imp_coef[imp_coef > threshold]
    newDataset = data[ImportantFeatures.index].copy()
    newDataset["time_to_eruption"] = data["time_to_eruption"]
    return ImportantFeatures.index, newDataset
