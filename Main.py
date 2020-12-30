import argparse
import os

from sklearn import model_selection

import evaluation
from load_data import create_matrix, save_data_to_csv
import pandas as pd
from FeatureSelection import FeatureSelectionUsingCorrelation,FeatureSelectionWrapper,FeatureSelectionEmbedded
from model_application import apply_LGBM_regression

path_local = '/home/fervn98/PycharmProjects/DATASETCI'
def MethodSelection(methodselected, dataframe,y_train):
    if methodselected=="LGBM":
        msefinal=0
        rmsefinal=0
        maefinal=0
        for i in range(10):
            train, val, y, y_val = model_selection.train_test_split(dataframe[dataframe.columns[:-1]],
                                                                    dataframe[dataframe.columns[-1]],
                                                                    test_size=0.2, shuffle=True)
            prediction=apply_LGBM_regression(train,y, val)
            mse, rmse, mae = evaluation.all_errors(y_val, prediction)
            msefinal+=mse
            rmsefinal+=rmse
            maefinal+=mae
        msefinal=msefinal/10
        rmsefinal=rmsefinal/10
        maefinal=maefinal/10
        print(msefinal)
        print(rmsefinal)
        print(maefinal)
    return 0
def FeatureSelection(methodselected,dataframe, threshold):
    if methodselected=="Correlation":
        return FeatureSelectionUsingCorrelation(dataframe,threshold)
    elif methodselected=="Wrapper":
        return FeatureSelectionWrapper(dataframe, threshold)
    elif methodselected=="Embedded":
        return FeatureSelectionEmbedded(dataframe, threshold)
    return 0
def load_data():
    df_log = pd.read_csv(os.path.join(path_local, "train_final_data.csv"))
    df_log2 = pd.read_csv(os.path.join(path_local, "test_final_data.csv"))
    return df_log, df_log2
def main():
    parser = argparse.ArgumentParser(description="parameters")
    parser.add_argument("--loadData", type=str, default="Yes",
                        choices=["Yes", "No"])
    parser.add_argument("--FeatureSelection", type=str, default=None,
                        choices=['Correlation', 'Wrapper', 'Embedded'])
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--method", type=str,default=None,
                        choices=["LGBM"])

    args = parser.parse_args()
    if args.loadData == "Yes":
        traindataframe, testdataframe=load_data()
    elif args.loadData == "No":
        create_matrix("INTRODUCIRPATH")
        save_data_to_csv("","","")
    FeaturesSelected, Traindataframe=FeatureSelection(args.FeatureSelection,traindataframe,args.threshold)
    MethodSelection(args.method,Traindataframe, testdataframe)

if __name__ == '__main__':
    main()
