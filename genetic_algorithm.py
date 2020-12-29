import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import model_application
from keras.layers import LSTM, Input, Dense
from keras.models import Model
import evaluation
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
from lightgbm import LGBMRegressor



#we will use ga to choose the best n_estimators and number_leaves
np.random.seed(1120)


def train_evaluate(ga_individual_solution):
    # Decode genetic algorithm solution to integer for window_size and num_units
    n_estimators_bits = BitArray(ga_individual_solution[0:6])
    n_leaves_bits = BitArray(ga_individual_solution[6:])
    n_estimators = n_estimators_bits.uint
    n_leaves = n_leaves_bits.uint
    print('\nn_estimators: ', n_estimators, ', n_leaves: ', n_leaves)

    # Return fitness score of 100 if window_size or num_unit is zero
    if n_estimators == 0 or n_leaves == 0:
        return 100,

    file_name = 'train_final_data.csv'
    df_data = pd.read_csv(file_name)
    df_data = df_data.drop(columns=["segment_id"])
    # split into train and validation (80/20)
    X_train, X_val, y_train, y_val = model_selection.train_test_split(df_data[df_data.columns[:-1]],
                                                                      df_data[df_data.columns[-1]],
                                                                      test_size=0.20, random_state=1120)

    # Train LGBM model and predict on validation set
    lgb = LGBMRegressor(random_state=600, num_leaves=n_leaves, n_estimators=n_estimators)
    lgb.fit(X_train, y_train)
    preds = lgb.predict(X_val)
    mse, rmse, mae = evaluation.all_errors(y_val, preds)

    return rmse


def apply_ga():
    population_size = 4
    num_generations = 4
    gene_length = 10

    # Our goal is to minimize the RMSE score, that's why using -1.0.
    creator.create('FitnessMax', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('binary', bernoulli.rvs, 0.5)  # random variates
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n=gene_length)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('mate', tools.cxOrdered)
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.6)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('evaluate', train_evaluate)

    population = toolbox.population(n=population_size)
    # r = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.1, ngen=num_generations, verbose=False)

    best_individuals = tools.selBest(population, k=1)

    for bi in best_individuals:
        n_estimators_bits = BitArray(bi[0:6])
        n_leaves_bits = BitArray(bi[6:])
        best_n_estimators = n_estimators_bits.uint
        best_n_leaves = n_leaves_bits.uint
        print('\nnumber of estimators: ', best_n_estimators, ',  number of leaves: ', best_n_leaves)


apply_ga()

