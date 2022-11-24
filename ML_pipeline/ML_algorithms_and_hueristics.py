from side_code.config import *
from ML_pipeline.ML_config import *
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, accuracy_score,precision_score, recall_score
import numpy as np
import os
import lightgbm
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV,GroupKFold
from sklearn.feature_selection import RFECV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve




def RFE(model,X,y,group_splitter,n_jobs):
    selector = RFECV(model, step=3, cv=group_splitter, n_jobs=n_jobs,min_features_to_select=X.shape[1] )#
    selector = selector.fit(X, y.ravel())
    model = selector.estimator
    X_new =  selector.transform(X)
    logging.info(f"Number of features after feature selection: {X_new.shape[1]} out of {(X.shape[1])}")
    return selector,X_new, model

def ML_model(X_train, groups, y_train, n_jobs, path, classifier):
    if os.path.exists(path):
        model = pickle.load(open(path,"rb"))
        return model
    else:
        if classifier:
            model = lightgbm.LGBMClassifier()
            param_grid = CLASSIFICATION_PARAM_GRID
        else:
            model = lightgbm.LGBMRegressor()
            param_grid = REGRESSION_PARAM_GRID
        group_splitter = list(GroupKFold(n_splits=5).split(X_train, y_train.ravel(), groups=groups))
        selector, X_train, model = RFE(model, X_train, y_train, group_splitter, n_jobs)
        grid_search = GridSearchCV(estimator= model, param_grid=param_grid,
                                   cv=group_splitter, n_jobs=n_jobs, pre_dispatch='1*n_jobs', verbose=2)
        grid_search.fit(X_train, y_train.ravel())
        best_model = grid_search.best_estimator_
    if classifier:
        calibrated_model = CalibratedClassifierCV(base_estimator=best_model, cv=group_splitter, method = 'isotonic')
        calibrated_model.fit(X_train, y_train.ravel())
    else:
        calibrated_model = None
    model = {'best_model': best_model,'calibrated_model': calibrated_model, 'selector': selector}
    pickle.dump(model, open(path, "wb"))
    return model


def calibration_plot(model, test_data, y_test):
    calibrated_prediction = model['calibrated_model'].predict_proba((model['selector']).transform(test_data))[:, 1]
    uncalibrated_prediction = model['best_model'].predict_proba((model['selector']).transform(test_data))[:, 1]
    fop_uncalibrated, mpv_uncalibrated = calibration_curve(y_test, uncalibrated_prediction, n_bins=10, normalize=True)
    fop_calibrated, mpv_calibrated = calibration_curve(y_test, calibrated_prediction, n_bins=10)
    # plot perfectly calibrated
    pyplot.plot([0, 1], [0, 1], linestyle='--', color='black')
    # plot model reliabilities
    pyplot.plot(mpv_uncalibrated, fop_uncalibrated, marker='.')
    pyplot.plot(mpv_calibrated, fop_calibrated, marker='.')
    pyplot.show()


def print_model_statistics(model, test_X, y_test, is_classification, vi_path, name):
    var_impt = variable_importance(test_X.columns,model['best_model'])
    var_impt.to_csv(vi_path, sep=CSV_SEP)
    logging.info(f"{name} variable importance: \n {var_impt}")
    predicted = model['best_model'].predict((model['selector']).transform(test_X))
    test_metrics = model_metrics(y_test, predicted, is_classification= is_classification)
    logging.info(f"{name} metrics: \n {test_metrics}")
    #if is_classification:
    #    calibration_plot(model, test_X, y_test)



def model_metrics(y_test, predictions, is_classification):
    if is_classification:
        return {'AUC' :roc_auc_score(y_test, predictions),'accuracy_score':accuracy_score(y_test, predictions),'precision':precision_score(y_test, predictions),'recall':recall_score(y_test, predictions)}
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



def variable_importance(columns, model):
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(columns, model.feature_importances_):
        feats[feature] = importance  # add the name/value pair

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances.sort_values(by='Gini-importance', inplace=True)
    return importances