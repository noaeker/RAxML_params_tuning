from side_code.config import *
from ML_pipeline.ML_config import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, accuracy_score, precision_score, \
    recall_score, PrecisionRecallDisplay
import numpy as np
import os
import lightgbm
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.feature_selection import RFECV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV



def regression_per_group(df):
    y_pred = list(df.y_pred)
    y = list(df.y)
    return r2_score(y, y_pred)


def AUC_per_group(df):
    y_pred = list(df.y_pred)
    y = list(df.y)
    return roc_auc_score(y, y_pred)


def AUPRC_per_group(df):
    y_pred = list(df.y_pred)
    y = list(df.y)
    return average_precision_score(y, y_pred)


def score_func(y, y_pred, classification, groups_data):
    all_grouping = []
    for group in groups_data:
        df = pd.DataFrame({'y': y, 'y_pred': y_pred, 'grouping_col': groups_data[group]})
        if classification:
            df = df.groupby('grouping_col').apply(AUC_per_group).reset_index(name='AUC')
        else:
            df = df.groupby('grouping_col').apply(regression_per_group).reset_index(name='R2')
        df["grouping_col_name"] = group
        all_grouping.append(df)
    return pd.concat(all_grouping)


def RFE(model, X, y, group_splitter, n_jobs, scoring, do_RFE):
    if do_RFE:
        min_features = 3
    else:
        min_features = X.shape[1]
    selector = RFECV(model, step=1, cv=group_splitter, n_jobs=n_jobs, min_features_to_select=min_features,
                     scoring=scoring)  # min_features_to_select= 30,X.shape[1] X.shape[1]
    selector = selector.fit(X, y.ravel())
    model = selector.estimator
    X_new = X[X.columns[selector.get_support(indices=True)]]
    logging.info(f"Number of features after feature selection: {X_new.shape[1]} out of {(X.shape[1])}")
    return selector, X_new, model




def ML_model(X_train, groups, y_train, n_jobs, path, classifier=False, model='lightgbm', calibrate=True, name="", large_grid = False, do_RFE = False, n_cv_folds = 3):
    path = path + name
    logging.info(f"Building a {name} model and saving to {path}")
    if path and os.path.exists(path):
        logging.info(f"Using existing model in {path}")
        model = pickle.load(open(path, "rb"))
        return model
    else:
        group_splitter = list(GroupKFold(n_splits=n_cv_folds).split(X_train, y_train.ravel(), groups=groups))
        if classifier:
            if model=='lightgbm':
                model =lightgbm.LGBMClassifier(importance_type='gain')
                param_grid = LIGHTGBM_CLASSIFICATION_PARAM_GRID
                if large_grid:
                    param_grid.update(GENERAL_PARAM_GRID)
            elif model=='sgd':
                model = make_pipeline(StandardScaler(),SGDClassifier(loss='modified_huber'))
                param_grid = {}
            elif model == 'rf':
                model = RandomForestClassifier()
                param_grid = {}
        else:
            model = lightgbm.LGBMRegressor()
            param_grid = REGRESSION_PARAM_GRID
            if large_grid:
                REGRESSION_PARAM_GRID.update(GENERAL_PARAM_GRID)
        if classifier:
            scoring = 'roc_auc'
        else:
            scoring = 'r2'
        selector, X_train, model = RFE(model, X_train, y_train, group_splitter, n_jobs, scoring, do_RFE)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                   cv=group_splitter, n_jobs=n_jobs, pre_dispatch='1*n_jobs', verbose=2,
                                   scoring=scoring)
        grid_search.fit(X_train, y_train.ravel())
        best_model = grid_search.best_estimator_
    if classifier and calibrate:
        logging.info("Generating calibrated model for classification model")
        calibrated_model = CalibratedClassifierCV(base_estimator=best_model, cv=group_splitter, method='isotonic')
        calibrated_model.fit(X_train, y_train.ravel())
    else:
        calibrated_model = best_model
    model = {'best_model': best_model, 'calibrated_model': calibrated_model, 'selector': selector}
    if path:
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

def enrich_with_single_feature_metrics(var_impt, train_X, y_train, test_X, y_test):
    auc_scores = []
    auprc_scores = []
    mcc_scores = []
    for feature in var_impt.index:
        try:
            lg = LogisticRegression(random_state=0).fit(train_X[[feature]], y_train)
            #lg = lightgbm.LGBMClassifier(importance_type='gain').fit(train_X[[feature]], y_train)
            proba = lg.predict_proba(test_X[[feature]])[:, 1]
            pred = lg.predict(test_X[[feature]])
            auc = roc_auc_score(y_test, proba)
            auprc = average_precision_score(y_test, proba)
            mcc = matthews_corrcoef(y_test, pred)

        except:
            auc = -1
            auprc = -1
            mcc = -1
        auc_scores.append(auc)
        auprc_scores.append(auprc)
        mcc_scores.append(mcc)
    var_impt["AUC"] = auc_scores
    var_impt["AUPRC"] = auprc_scores
    var_impt["mcc"] = mcc_scores



def print_model_statistics(model, train_X, test_X,val_X, y_train, y_test,y_val, is_classification, vi_path, metrics_path,
                           group_metrics_path, name, sampling_frac, test_MSAs, feature_importance=True):
    if feature_importance:
        try:
            var_impt = variable_importance(train_X.columns[model['selector'].get_support(indices=True)], model['best_model'])
            if vi_path and (sampling_frac==1 or sampling_frac==-1):
                enrich_with_single_feature_metrics(var_impt, train_X, y_train, test_X, y_test)
                var_impt.to_csv(vi_path, sep=CSV_SEP)
                    #logging.info(f"AUC for feature {feature} is {auc}" )
            logging.info(f"{name} variable importance: \n {var_impt}")
        except:
            logging.info("No existing feature importance procedure for this  model")
    predicted_train = model['best_model'].predict((model['selector']).transform(train_X))
    predicted_test = model['best_model'].predict((model['selector']).transform(test_X))
    predicted_val = model['best_model'].predict((model['selector']).transform(val_X))
    if is_classification:
        predicted_proba_train = model['best_model'].predict_proba((model['selector']).transform(train_X))[:, 1]
        predicted_proba_test = model['best_model'].predict_proba((model['selector']).transform(test_X))[:, 1]
        predicted_proba_val = model['best_model'].predict_proba((model['selector']).transform(val_X))[:, 1]
    else:
        predicted_proba_train = predicted_train
        predicted_proba_test = predicted_test
        predicted_proba_val = predicted_val
    groups_data_test = test_X[
        ["feature_msa_n_seq", "feature_msa_n_loci","feature_msa_pypythia_msa_difficulty"]]#

    groups_dict_test = {'msa_difficulty_group': pd.qcut(groups_data_test["feature_msa_pypythia_msa_difficulty"], 4),
                        "n_seq_group": pd.qcut(groups_data_test["feature_msa_n_seq"], 4),
                        "feature_msa_n_loci": pd.qcut(groups_data_test["feature_msa_n_loci"], 4)}

    train_metrics = model_metrics(y_train, predicted_train, predicted_proba_train, group_metrics_path, sampling_frac,
                                  is_classification=is_classification,
                                  groups_data=None)
    test_metrics = model_metrics(y_test, predicted_test, predicted_proba_test, group_metrics_path, sampling_frac,
                                 is_classification=is_classification, groups_data=groups_dict_test)

    validation_metrics = model_metrics(y_val, predicted_val, predicted_proba_val, group_metrics_path, sampling_frac,
                                 is_classification=is_classification, groups_data=None)

    logging.info(f"{name} train metrics: \n {train_metrics}")
    logging.info(f"{name} test metrics: \n {test_metrics}")
    logging.info(f"{name} validation metrics: \n {validation_metrics}")

    train_metrics.update(test_metrics)
    train_metrics = pd.DataFrame.from_dict([train_metrics])
    train_metrics["sample_fraction"] = sampling_frac
    add_to_csv(csv_path=metrics_path, new_data=train_metrics)

    #if is_classification:
    #    calibration_plot(model, test_X, y_test)


def add_to_csv(csv_path, new_data):
    if not os.path.exists(csv_path):
        new_data.to_csv(csv_path, sep=CSV_SEP, index=False)
    else:
        curr_metrics_df = pd.read_csv(csv_path, sep=CSV_SEP, index_col=False)
        metric_df = pd.concat([curr_metrics_df, new_data])
        metric_df.to_csv(csv_path, sep=CSV_SEP)


def model_metrics(y_test, predictions, prob_predictions, metrics_path, sampling_frac, is_classification, groups_data):
    if is_classification:
        if groups_data and (sampling_frac==1 or sampling_frac==-1):
                auc_per_group = score_func(y_test, prob_predictions, classification=True, groups_data=groups_data)
                auc_per_group["sampling_frac"] = sampling_frac
                add_to_csv(metrics_path, auc_per_group)
                logging.info(auc_per_group)
        # PrecisionRecallDisplay.from_predictions(y_test, prob_predictions)
        # plt.show()
        return {'AUC': roc_auc_score(y_test, prob_predictions),
                'average_precision': average_precision_score(y_test, prob_predictions),
                'accuracy_score': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions), 'recall': recall_score(y_test, predictions),
                'mcc': matthews_corrcoef(y_test, predictions)}
    else:
        if groups_data and (sampling_frac==1 or sampling_frac==-1):
            r2_per_group = score_func(y_test, predictions, classification=False, groups_data=groups_data)
            r2_per_group["sampling_frac"] = sampling_frac
            add_to_csv(metrics_path, r2_per_group)
            logging.info(r2_per_group)
        return {"r2": r2_score(y_test, predictions), "MAE": mean_absolute_error(y_test, predictions),
                "MSE": mean_squared_error(y_test, predictions)
                }


def train_test_validation_splits(full_data, test_pct, val_pct, msa_col_name="msa_name", subsample_train=False,
                                 subsample_train_frac=-1,remove_unrelaible_times = False):
    logging.info(f"Original number of MSAs in full data is {len(full_data.msa_path.unique())}")
    if remove_unrelaible_times:
        logging.info("Remove unrelaible times from data")
        full_data = full_data.loc[~full_data.non_reliable_timings]
        logging.info(f"New Number of MSAs in full data is {len(full_data.msa_path.unique())}")
    validation_data = full_data.loc[(full_data[msa_col_name].str.contains("Single_gene_PROTEIN") | full_data[msa_col_name].str.contains("Single_gene_DNA"))]
    logging.info(f"Number of MSAs in validation data is {len(validation_data.msa_path.unique())}")
    #logging.info(f"Number of overall positive samples in validation: {len(validation_data.loc[validation_data.is_global_max == 1].index)}, Number of overall negative samples in validation is {len(validation_data.loc[validation_data.is_global_max == 0].index)}")
    full_data = full_data.loc[~(full_data[msa_col_name].str.contains("Single_gene_PROTEIN") | full_data[msa_col_name].str.contains("Single_gene_DNA"))]
    np.random.seed(SEED)
    logging.info("Partitioning MSAs according to number of sequences")
    msa_n_seq_group = pd.qcut(full_data["feature_msa_n_seq"], 4)
    msas = full_data[msa_col_name]
    full_sampling_data = pd.DataFrame({'msa_n_seq': msa_n_seq_group, 'msa': msas}).drop_duplicates().sample(frac=1,random_state = SEED)
    full_sampling_data  = full_sampling_data.sort_values('msa') # Sort according to MSAs
    test_msas = full_sampling_data.groupby('msa_n_seq').sample(frac=test_pct, random_state= SEED)
    train_msas = full_sampling_data.loc[~full_sampling_data['msa'].isin(test_msas['msa'])]

    if subsample_train:
        logging.info(f"Subsampling training data to {subsample_train_frac}")
        train_msas = train_msas.groupby('msa_n_seq').sample(frac=subsample_train_frac, random_state=SEED)
    train_data = full_data[full_data[msa_col_name].isin(train_msas['msa'])]
    logging.info(f"Number of MSAs in training data is {len(train_data.msa_path.unique())}")
    #logging.info(f"Number of overall positive samples in train: {len(train_data.loc[train_data.is_global_max == 1].index)}, Number of overall negative samples in test {len(train_data.loc[train_data.is_global_max == 0].index)}")
    test_data = full_data[full_data[msa_col_name].isin(test_msas['msa'])]
    logging.info(f"Number of MSAs in test data is {len(test_data.msa_path.unique())}")
    #logging.info(f"Number of overall positive samples in test: {len(test_data.loc[test_data.is_global_max == 1].index)}, Number of overall negative samples in test {len(test_data.loc[test_data.is_global_max == 0].index)}")
    return train_data, test_data, validation_data


def variable_importance(columns, model):
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(columns, model.feature_importances_):
        feats[feature] = importance  # add the name/value pair

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances.sort_values(by='Gini-importance', inplace=True)
    return importances
