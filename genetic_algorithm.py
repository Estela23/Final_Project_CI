import pandas as pd
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import evaluation
import os

# fitness function is rMSE

def apply_ga():
    varbound = np.array([[0, 10]] * 3)

    model = ga(function=evaluation.rMSE(), dimension=3, variable_type='real', variable_boundaries=varbound)

    model.run()

    convergence=model.report
    solution=model.ouput_dict


path_local = 'C:/Users/Tair/Documents/MAI Semester1/CI/Project'
file_name = 'train_final_data.csv'
df_data = pd.read_csv(os.path.join(path_local, file_name))
preds = apply_LGBM_regression(train, y, val)
