import argparse
from sklearn import model_selection
from aplying_nn import apply_NN
import pandas as pd
from FeatureSelection import FeatureSelectionUsingCorrelation,FeatureSelectionWrapper,FeatureSelectionEmbedded
from model_application import apply_LGBM


def MethodSelection(methodselected, train, val, y, y_val, test):
    if methodselected == "LGBM":
       apply_LGBM(train, val, y, y_val, test)

    elif methodselected == "NN":
        apply_NN(train, val, y, y_val, test)


def FeatureSelection(methodselected, dataframe, threshold):
    if methodselected == "Correlation":
        return FeatureSelectionUsingCorrelation(dataframe, threshold)
    elif methodselected == "Wrapper":
        return FeatureSelectionWrapper(dataframe, threshold)
    else:
        train, val, y, y_val = model_selection.train_test_split(dataframe[dataframe.columns[:-1]],
                                                               dataframe[dataframe.columns[-1]],
                                                               test_size=0.2, shuffle=True)

        test_set = pd.read_csv("data/test_final_data_complete.csv")

        return train, val, y, y_val, test_set

    # elif methodselected == "Embedded":
    #    return FeatureSelectionEmbedded(dataframe, threshold)


def load_data():
    df_log = pd.read_csv("data/train_final_data_complete.csv")
    df_log2 = pd.read_csv("data/test_final_data_complete.csv")

    return df_log, df_log2


def main():
    parser = argparse.ArgumentParser(description="parameters")
    parser.add_argument("--FeatureSelection", type=str, default=None,
                        choices=['Correlation', 'Wrapper'])  # , 'Embedded'
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--method", type=str,default=None,
                        choices=["LGBM", "NN"])

    args = parser.parse_args()

    traindataframe, testdataframe = load_data()

    train, val, y, y_val, test = FeatureSelection(args.FeatureSelection, traindataframe, args.threshold)
    MethodSelection(args.method, train, val, y, y_val, test)


if __name__ == '__main__':
    main()
