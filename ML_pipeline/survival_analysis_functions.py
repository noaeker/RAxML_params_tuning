from sksurv.ensemble import RandomSurvivalForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import set_config
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.pipeline import make_pipeline
from sksurv.preprocessing import OneHotEncoder, encode_categorical
from sksurv.metrics import as_cumulative_dynamic_auc_scorer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)


def KM_estimator(validation_edited_df):
    # Kaplan meir estimator
    from sksurv.nonparametric import kaplan_meier_estimator
    for starting_tree_type in ("rand", "pars"):
        mask_type = validation_edited_df["starting_tree_type"] == starting_tree_type
        time_starting_tree_type, survival_prob_starting_tree_type = kaplan_meier_estimator(
            validation_edited_df["status"].astype(bool)[mask_type],
            validation_edited_df["n_trees_used"][mask_type])

        plt.step(time_starting_tree_type, survival_prob_starting_tree_type, where="post",
                 label="Starting tree type = %s" % starting_tree_type)

    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.legend(loc="best")
    plt.show()

def fit_and_score_features_cph(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X[:, j:j + 1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores


def feature_importance(X_test, Y_test, rsf):
    result = permutation_importance(
        rsf, X_test, Y_test, n_repeats=15, random_state=0
    )
    res = pd.DataFrame(
        {k: result[k] for k in ("importances_mean", "importances_std",)},
        index=X_test.columns
    ).sort_values(by="importances_mean", ascending=False)
    print(res)

def GB_survival_pipeline(X_train, Y_train, X_test, Y_test, va_times):
    gb = GradientBoostingSurvivalAnalysis(loss="coxph")
    gb.fit(X_train, Y_train)
    c = (gb.score(X_test, Y_test))
    print(f'GB C index={c}')
    feature_importance(X_test, Y_test, gb)
    #gb_chf_funcs = gb.predict_cumulative_hazard_function(
    #    X_test, return_array=False)
    #gb_risk_scores = np.row_stack([chf(va_times) for chf in gb_chf_funcs])

    #gb_auc, gb_mean_auc = cumulative_dynamic_auc(
    #    Y_train, Y_test, gb_risk_scores, va_times
    #)
    return gb


def RSF_survival_pipeline(X_train, Y_train, X_test, Y_test, va_times):
    rsf = RandomSurvivalForest()
    cv = KFold(n_splits=3, shuffle=True, random_state=1)
    cv_param_grid = {
        "estimator__max_depth": np.arange(1, 10, dtype=int),
        "estimator__min_samples_split": np.arange(3, 8, dtype=int),
    }
    rsf_optimized = GridSearchCV(
        as_cumulative_dynamic_auc_scorer(rsf, times=np.arange(1,20)),
        param_grid=cv_param_grid,
        cv=cv,
    ).fit(X_train, Y_train).best_estimator_
    #rsf.fit(X_train, Y_train)
    c = (rsf_optimized.score(X_test, Y_test))
    print(f'RSF C index={c}')
    feature_importance(X_test, Y_test, rsf_optimized)

    rsf_chf_funcs = rsf_optimized.predict_cumulative_hazard_function(
        X_test)
    rsf_risk_scores = np.row_stack([chf(va_times) for chf in rsf_chf_funcs])

    rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
        Y_train, Y_test, rsf_risk_scores, va_times
    )

    #plt.plot(va_times, cph_auc, "o-", label="CoxPH (mean AUC = {:.3f})".format(cph_mean_auc))
    return rsf_optimized, rsf_auc,rsf_mean_auc


def COX_survival_pipeline( X_train, Y_train, X_test, Y_test, va_times ):
    estimator = CoxPHSurvivalAnalysis(ties = 'efron')
    estimator.fit(X_train, Y_train)
    pd.Series(estimator.coef_, index=X_train.columns)
    train_c_score = estimator.score(X_train, Y_train)
    test_c_score = estimator.score(X_test, Y_test)
    print(f'train_c_score: {train_c_score},test_c_score: {test_c_score} ')
    scores = fit_and_score_features_cph(X_train.values, Y_train)
    print(scores)
    pd.Series(scores, index=X_train.columns).sort_values(ascending=False)
    cph_risk_scores = estimator.predict(X_test)
    cph_auc, cph_mean_auc = cumulative_dynamic_auc(
        Y_train, Y_test, cph_risk_scores, va_times
    )


    return estimator, cph_auc,cph_mean_auc


def transform_y_to_surv(data):
    data = data[["status", "n_trees_used"]].to_numpy()
    aux = [(e1, e2) for e1, e2 in data]
    ready_data = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    return ready_data


def get_train_test_X_y_survival(train, test, X_cols=["mean_success_prob", "feature_mds_False_stress_3_spr_enriched",
                                                     "feature_msa_pypythia_msa_difficulty"]):
    Y_cols = ["status", "n_trees_used"]
    X_train = train[X_cols]
    Y_train = transform_y_to_surv(train[Y_cols])
    X_test = test[X_cols]
    Y_test = transform_y_to_surv(test[Y_cols])
    return X_train, Y_train, X_test, Y_test

