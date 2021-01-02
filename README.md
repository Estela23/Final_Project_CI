# INGV - Volcanic Eruption Prediction

**Authors:**

Estela Vázquez-Monjardín Lorenzo

Tair Tahar

Alberto Soutullo Rendo

Fernando Vázquez Novoa

## Project Structure

In this project there is several files used to study the Kaggle competition.

* **data**: Examples of data submission files, and features created discarding volcanoes and with entire data. This are `_final_data` and `_final_data_complete` respectively.

* **results**: Different result files we uploaded to Kaggle.

* **aplying_nn.py**: Here we perform building, K-FOLD and grid search of the Neural Network.

* **aplying_lgbm.py**: Here we perform the building, K-FOLD of the LGBM.

* **build_features.py**: Here, we read the entire data of Kaggle (not uploaded in this repository), and we create the features used.

* **evaluation.py**: Several metric functions.

* **feature_selection.py**: Here we perform the selection of features with different algorithms.

* **genetic_algorithm.py**: Here we try to optimize the parameters of LGBM with genetic algorithms.

* **main.py**: Main execution, explained below.

## Execution: 

In order to execute this program to obtain the results, there are several flags to put when calling main:

* --FeatureSelection: ['Correlation', 'Wrapper']. Algorithm used for feature selection. With None, original data is applied. default: None.

* --threshold: Threshold used in the feature selection algorithms. default: 0.01.

* --method: ['LGBM', 'NN']. Models used in this competition. Default: None. Needs to be passed.


Example 1: Neural Network results with Wrapper as feature selection.
 ```shell script
 python main.py --method NN --FeatureSelection Wrapper
 ```

Example 2: LGBM results with all data
 ```shell script
 python main.py --method LGBM
 ```

This will print different error metrics, and will write a file in this path called `submission.csv`.
This is the file we used to submit to Kaggle.

## Kaggle results:

Kaggle ranking with best score:
![Alt text](results/kaggle_ranking.jpg?raw=true "Ranking")

Kaggle submissions:
![Alt text](results/kaggle_submissions.jpg?raw=true "Ranking")