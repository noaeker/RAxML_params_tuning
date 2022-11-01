from side_code.config import *
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, accuracy_score,precision_score, recall_score
import numpy as np
import os
import lightgbm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV,GroupKFold
from sklearn.feature_selection import RFECV

def classifier(X, groups, y, n_jobs, path="error", do_rfe = False):
    param_grid = {
        'learning_rate': [0.005, 0.01],
        'n_estimators': [8, 16, 24],
        'num_leaves': [6, 8, 12, 16],  # large num_leaves helps improve accuracy but might lead to over-fitting
        'boosting_type': ['gbdt', 'dart'],  # for better accuracy -> try dart
        'objective': ['binary'],
        'max_bin': [255, 510],  # large max_bin helps improve accuracy but might slow down training progress
        'random_state': [500],
        'colsample_bytree': [0.64, 0.65, 0.66],
        'subsample': [0.7, 0.75],
        'reg_alpha': [1, 1.2],
        'reg_lambda': [1, 1.2, 1.4],
        'metric': ['auc']
    }
    model = lightgbm.LGBMClassifier()
    if False:  # LOCAL_RUN:
        param_grid = {}
    group_splitter = list(GroupKFold(n_splits=3).split(X, y.ravel(), groups=groups))
    if do_rfe:
        selector = RFECV(model, step=3, cv=group_splitter, n_jobs=n_jobs)
        selector = selector.fit(X, y.ravel())
        model = selector.estimator
        feature_names = selector.feature_names_in_
        X = X[feature_names]
        logging.info(f"Number of features after feature selection: {len(selector.support_)}")
    # grid_search = RandomizedSearchCV(estimator= model, param_distributions=param_grid,
    #                           cv=group_splitter, n_jobs=n_jobs, pre_dispatch='1*n_jobs', verbose=2)
    # grid_search.fit(X_train, y_train.ravel())
    # best_classifier = grid_search.best_estimator_
    best_classifier = model.fit(X, y.ravel())
    pickle.dump(best_classifier, open(path, "wb"))
    return best_classifier


def regressor(X, groups, y, n_jobs, path="gbm_time", do_rfe = False):
    param_grid = {
        'learning_rate': [0.005, 0.01],
        'n_estimators': [8, 16, 24],
        'num_leaves': [6, 8, 12, 16],  # large num_leaves helps improve accuracy but might lead to over-fitting
        'boosting_type': ['gbdt', 'dart'],  # for better accuracy -> try dart
        'max_bin': [255, 510],  # large max_bin helps improve accuracy but might slow down training progress
        'random_state': [500],
        'colsample_bytree': [0.64, 0.65, 0.66],
        'subsample': [0.7, 0.75],
        'reg_alpha': [1, 1.2],
        'reg_lambda': [1, 1.2, 1.4]
    }

    model = lightgbm.LGBMRegressor()

    if False:
        param_grid = {}
    group_splitter = list(GroupKFold(n_splits=3).split(X, y.ravel(), groups=groups))
    if do_rfe:
        selector = RFECV(model, step=3, cv=group_splitter, n_jobs=n_jobs)
        selector = selector.fit(X, y.ravel())
        model = selector.estimator
        feature_names = selector.feature_names_in_
        X = X[feature_names]
        logging.info(f"Number of features after feature selection: {len(selector.support_)}")

    # grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
    #                           cv=group_splitter, n_jobs=n_jobs, pre_dispatch='1*n_jobs', verbose=2)
    # grid_search.fit(X_train, y_train.ravel())
    # best_regressor = grid_search.best_estimator_
    best_regressor = model.fit(X, y.ravel())
    # Calculate the absolute errors
    pickle.dump(best_regressor, open(path, "wb"))
    return best_regressor

def print_model_statistics(model, train_data,test_data,y_test,is_classification, vi_path, name):
    var_impt = variable_importance(train_data,model)
    var_impt.to_csv(vi_path, sep=CSV_SEP)
    logging.info(f"{name} variable importance: \n {var_impt}")
    predicted = model.predict(test_data)
    test_metrics = model_metrics(y_test, predicted, is_classification= is_classification)
    logging.info(f"{name} metrics: \n {test_metrics}")



def model_metrics(y_test, predictions, is_classification):
    if is_classification:
        return {'AUC' :roc_auc_score(y_test, predictions),'accuracy_score':accuracy_score(y_test, predictions),'precision':roc_auc_score(y_test, predictions),'recall':recall_score(y_test, predictions)}
    return {"r2": r2_score(y_test, predictions), "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mean_squared_error(y_test, predictions)
            }


def train_test_validation_splits(full_data, test_pct, val_pct, msa_col_name = "msa_name"):
    np.random.seed(SEED)
    msa_names = list(np.unique(full_data[msa_col_name]))
    val_and_test_msas = np.random.choice(msa_names, size=int(len(msa_names) * (test_pct + val_pct)), replace=False)
    np.random.seed(SEED + 1)
    test_msas = np.random.choice(val_and_test_msas,
                                 size=int(len(val_and_test_msas) * (test_pct / (test_pct + val_pct))), replace=False)
    validation_msas = [name for name in val_and_test_msas if name not in test_msas]
    train_data = full_data[~full_data[msa_col_name].isin(val_and_test_msas)]
    test_data = full_data[full_data[msa_col_name].isin(test_msas)]
    validation_data = full_data[full_data[msa_col_name].isin(validation_msas)]
    logging.info(f"Train data size is {len(train_data.index)} Test data size is{len(test_data.index)} Validation data size is {len(validation_data.index)}")
    return train_data, test_data, validation_data



def variable_importance(X_train, rf_model):
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(X_train.columns, rf_model.feature_importances_):
        feats[feature] = importance  # add the name/value pair

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances.sort_values(by='Gini-importance', inplace=True)
    return importances