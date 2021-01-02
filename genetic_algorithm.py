import numpy as np
import pandas as pd
from sklearn import model_selection
import evaluation
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from lightgbm import LGBMRegressor
from sympy.combinatorics.graycode import GrayCode, gray_to_bin, bin_to_gray


# we will use ga to find the best n_estimators, number_leaves and maximum depth of the tree got LGM

def clean_bins(lst):
    result = ''.join(str(e) for e in lst)
    return result


def train_evaluate(ga_individual_solution):
    # Decode genetic algorithm solution to integer for window_size and num_units
    n_estimators_bits = gray_to_bin(clean_bins(ga_individual_solution[0:12]))
    n_leaves_bits = gray_to_bin(clean_bins(ga_individual_solution[12:22]))
    max_depth_bits = gray_to_bin(clean_bins(ga_individual_solution[22:]))
    n_estimators = int(n_estimators_bits, 2)
    n_leaves = int(n_leaves_bits, 2)
    max_depth = int(max_depth_bits, 2)
    print('\nn_estimators: ', n_estimators, ', n_leaves: ', n_leaves, ', max_depth: ', max_depth)

    # Return fitness score of in case of zeros
    if n_estimators == 0 or n_leaves == 0 or max_depth == 0:
        return 100,

    file_name = 'data/train_final_data_complete.csv'
    df_data = pd.read_csv(file_name)
    df_data = df_data.drop(columns=["segment_id"])
    # split into train and validation (80/20)
    X_train, X_val, y_train, y_val = model_selection.train_test_split(df_data[df_data.columns[:-1]],
                                                                      df_data[df_data.columns[-1]],
                                                                      test_size=0.20, random_state=1120)

    # Train LGBM model and predict on validation set
    lgb = LGBMRegressor(random_state=600, num_leaves=n_leaves, n_estimators=n_estimators,
                        max_depth=max_depth)
    lgb.fit(X_train, y_train)
    preds = lgb.predict(X_val)
    mse, rmse, mae = evaluation.all_errors(y_val, preds, verbose=True)

    return mse, rmse, mae


def apply_ga(population_size, num_generations, gene_length, mu, _lambda):
    np.random.seed(120)
    creator.create('FitnessMin', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('binary', bernoulli.rvs, 0.5)  # random variates
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n=gene_length)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.3)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('evaluate', train_evaluate)

    population = toolbox.population(n=population_size)

    algorithms.eaMuCommaLambda(population, toolbox, mu, _lambda, cxpb=0.9, mutpb=0.01, ngen=num_generations,
                               verbose=True)

    best_individuals = tools.selBest(population, k=1, fit_attr='fitness')

    for bi in best_individuals:
        n_estimators_bits = gray_to_bin(clean_bins(bi[0:12]))
        n_leaves_bits = gray_to_bin(clean_bins(bi[12:22]))
        max_depth_bits = gray_to_bin(clean_bins(bi[22:]))
        best_n_estimators = int(n_estimators_bits, 2)
        best_n_leaves = int(n_leaves_bits, 2)
        best_max_depth = int(max_depth_bits, 2)

        print('\nnumber of estimators: ', best_n_estimators, ',  number of leaves: ', best_n_leaves, ', max_depth: ',
              best_max_depth)

#############################3
population_size = 10
num_generations = 100
gene_length = 26
mu = population_size
_lambda = mu
apply_ga(population_size, num_generations, gene_length, mu, _lambda)
