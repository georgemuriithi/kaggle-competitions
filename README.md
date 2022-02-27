# Kaggle Competitions

<a href="https://github.com/georgemuriithi/kaggle-competitions/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/georgemuriithi/kaggle-competitions.svg?color=blue&cachedrop">
</a>

Participated in **House Prices Prediction** and **Credit Default Risk Prediction** competitions.

https://www.kaggle.com/c/house-prices-advanced-regression-techniques

https://www.kaggle.com/c/home-credit-default-risk

In both, **advanced decision tree-based models** for regression and classification are used.

In House Prices Prediction, performance evaluation is based on **RMSLE (Root Mean Squared Logarithmic Error),** while in Credit Default Risk Prediction, it is based on **AUROC (Area Under Receiver Operating Characteristic).**

In House Prices Prediction, I ranked **816/5011,** with an error of **0.12549,** compared to the best one of **0.00000.**

![Screenshot 2022-01-24 115000](https://user-images.githubusercontent.com/21691211/150891215-38bbc1d8-543d-4b7d-94af-4414af37bdd6.png)

In Credit Default Risk Prediction, I scored **0.73610,** compared to the best score of **0.81724.** Ranking was unavailable.

![Screenshot 2022-01-26 220705](https://user-images.githubusercontent.com/21691211/151168378-80fbd534-1054-4a55-9d5f-9175b9ce3bdf.png)

My submissions can be accessed from the *submissions* folder.

## Problem Description
The problems are well detailed in the kaggle links provided above.

## Solution Approach
### <a href="https://github.com/georgemuriithi/kaggle-competitions/blob/main/House-Prices-Prediction.ipynb">House Prices Prediction</a>
<a href="https://colab.research.google.com/drive/1S1iZ_7c9rMUBq7pxDLEIuCWKvlFgCxod?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

After **Feature engineering,** the following models for regression are tested:

- `Ridge`
- `BaggingRegressor`
  - `n_estimators=50`
- `RandomForestRegressor`
  - `n_estimators=50`
- `XGBRegressor`
  - `max_depth=5`
  - `objective='reg:squarederror'`
- `LGBMRegressor`
- `VotingRegressor`
  - `estimators=[ridge, bagging, random_forest, xgb, lgbm]`
  - `n_jobs=-1`
- `StackingRegressor`
  - `estimators=[ridge, bagging, random_forest, xgb, lgbm]`
  - `final_estimator=Ridge`
  - `n_jobs=-1`

**Hyperparameters:**

- `train_test_split(test_size=0.2, random_state=0)`
- `kfold = KFold(n_splits=5, shuffle=True, random_state=0)`
- `cross_val_score(cv=kfold)`

`VotingRegressor` turns out as the best performing, with the best combined **Validation R<sup>2</sup> score, RMSLE and Cross validation R<sup>2</sup> mean score.**

### <a href="https://github.com/georgemuriithi/kaggle-competitions/blob/main/Credit-Default-Risk-Prediction.ipynb">Credit Default Risk Prediction</a>
<a href="https://colab.research.google.com/drive/1HqGHxtlR9y3O98MZx5kM3vAdohcATKHU?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

After **Feature engineering,** the following models for classification are tested:

- `XGBClassifier`
  - `tree_method='gpu_hist'`
  - `gpu_id=0`
- `LGBMClassifier`
  - `device='gpu'`
- `RandomForestClassifier`
  - `n_estimators=50`
- `StackingClassifier`
  - `estimators=[xgb, lgbm, random_forest]`
  - `final_estimator=LGBMClassifier`
  - `n_jobs=-1`

**Hyperparameter:** `train_test_split(test_size=0.2, random_state=42)`

***GPU** is leveraged. Classification task takes a lot of computation power and time.*

`LGBMClassifier` turns out as the best performing, with the maximum **Validation AUROC score.**
