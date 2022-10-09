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
from sklearn.model_selection import GridSearchCV

def classifier(X_train, y_train, n_jobs, path="error", use_lightgbm=False):
    if use_lightgbm:
        path = path + "_lightgbm"
    if os.path.exists(path):
        logging.info(f"Using existing classifier model in {path}")
        return pickle.load(open(path, "rb"))

    logging.info(f"Building a new classifier model")
    if not use_lightgbm:
        param_grid = {
            'bootstrap': [True, False],
            'max_depth': [80, 90, 100, 110],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300, 1000]
        }
        error_model = RandomForestClassifier()
    else:
        param_grid = {'boosting_type': ['gbdt', 'dart', 'rf', 'goss'],
                      'reg_lambda': [2000.0, 100.0, 500.0, 10000.0, 1200.0, 0.0, 0.5, 1.5, 3],
                      'reg_alpha': [10.0, 100.0, 0., 1.0, 0.5, 3.0],
                      'num_leaves': [100, 50],
                      'bagging_fraction': [0.1, 0.3, 0.5, 0.75, 0.99],
                      'subsample': [0.1, 0.3, 0.5, 0.75, 0.99]}
        error_model = lightgbm.LGBMClassifier()
    if LOCAL_RUN:
        param_grid = {}
    grid_search = GridSearchCV(estimator=error_model, param_grid=param_grid,
                               cv=5, n_jobs=n_jobs, pre_dispatch='1*n_jobs', verbose=2)
    grid_search.fit(X_train, y_train.ravel())
    best_classifier = grid_search.best_estimator_

    # gbm_classifier = lightgbm.LGBMClassifier()
    # GS_res_gbm = GridSearchCV(gbm_classifier, lgbm_grid_param_space).fit(X_train, y_train.ravel())
    # clf_gbm = GS_res_gbm.best_estimator_
    # gbm_classifier.fit(X_train, y_train)
    pickle.dump(best_classifier, open(path, "wb"))
    return best_classifier


def regressor(X_train, y_train, n_jobs, path="gbm_time", use_lightgbm=False):
    if use_lightgbm:
        path = path + "_lightgbm"
    if os.path.exists(path):
        logging.info(f"Using existing regressor model in {path}")
        return pickle.load(open(path, "rb"))
    logging.info(f"Building a new regressor model")
    if not use_lightgbm:
        param_grid = {
            'bootstrap': [True, False],
            'max_depth': [80, 90, 100, 110],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300, 1000]
        }
        model = RandomForestRegressor()
    else:
        param_grid = {'boosting_type': ['gbdt', 'dart', 'rf', 'goss'],
                      'reg_lambda': [2000.0, 100.0, 500.0, 10000.0, 1200.0, 0.0, 0.5, 1.5, 3],
                      'reg_alpha': [10.0, 100.0, 0., 1.0, 0.5, 3.0],
                      'num_leaves': [100, 50],
                      # 'bagging_fraction': [0.1, 0.3, 0.5, 0.75, 0.99],
                      'subsample': [0.1, 0.3, 0.5, 0.75, 0.99]}

        model = lightgbm.LGBMRegressor()

    if LOCAL_RUN:
        param_grid = {}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=3, n_jobs=n_jobs, pre_dispatch='1*n_jobs', verbose=2)
    grid_search.fit(X_train, y_train.ravel())
    best_regressor = grid_search.best_estimator_
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