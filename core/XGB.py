from core.util import n_iter, kfold, random_state
import pandas as pd
import core.util as util
from time import perf_counter
import shap
from datetime import timedelta
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder as LE
import core.FeatureImportance as FI
import core.DrawPic as DrawPic
def run_xgb(X_new, X_test_new, Y_train, Y_test,Y_name,groups,model_type):
    sample_weight = compute_sample_weight(class_weight='balanced', y=Y_train)
    print("Running XGBoost regression")
# Set up model
    model_name = "XGB"
    if(model_type == "regr"):
        model = xgb.XGBRegressor(random_state=random_state)
        grid_pipe_xgb = {'n_estimators': list(range(100, 1100, 100)),
               'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3],
               'gamma': [i/10 for i in range(0,6)],
               'max_depth': list(range(2, 16)),
               'min_child_weight': list(range(1,11)),
               'subsample': [x/10 for x in range(2, 11)],
               'colsample_bytree': [x/10 for x in range(2, 11)],
               'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5],
               'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]}
        scoring='neg_mean_squared_error'  # Default scoring for regression
    elif(model_type == "class"):
        model = xgb.XGBClassifier(random_state=random_state)
        grid_pipe_xgb = {'n_estimators': list(range(100, 1100, 100)),
           'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3],
           'gamma': [i/10 for i in range(0,6)],
           'max_depth': list(range(2, 16)),
           'min_child_weight': list(range(1,11)),
           'subsample': [x/10 for x in range(2, 11)],
           'colsample_bytree': [x/10 for x in range(2, 11)],
           'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5],
           'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5],
           'scale_pos_weight': [1, 2, 5]}
        scoring = 'balanced_accuracy'  # Default scoring for classification
        le = LE()
        Y_train = pd.Series(le.fit_transform(Y_train), index=Y_train.index, name="target")
        Y_test = pd.Series(le.transform(Y_test), index=Y_test.index, name="target")
        
    else:
        raise ValueError("model_type must be 'regr' or 'class'")

    # Define parameter grid 
   

    grid_xgb_debug={'n_estimators': list(range(100, 1100, 100)),'max_depth': list(range(2, 15))}
    xgb = RandomizedSearchCV(estimator = model, param_distributions = grid_pipe_xgb, 
                                  scoring = scoring, n_iter = n_iter, 
                                  cv = util.PseudoGroupCV(kfold,groups), 
                                  verbose = 0, random_state = random_state, n_jobs = -1)
    # Randomized search:
    start = perf_counter()
    print("Fitting XGBoost model")
    xgb.fit(X_new, Y_train.values.ravel(),sample_weight=sample_weight)
    print("Best params: ", xgb.best_params_)
    stop = perf_counter()
    print("Time: ", timedelta(seconds = stop-start))

    # Save results
    best_results = util.save_results_cv_pipe(xgb, model_name, model_type, scoring,Y_name,X_new)
    print("Best results:")
    print(best_results)

    # Initialize model
    xgb_mod = xgb.best_estimator_

    # Fits the model on the data
    if model_type == "class":
        xgb_mod.fit(X_new, Y_train.values.ravel(), sample_weight=sample_weight)
    else:
        xgb_mod.fit(X_new, Y_train.values.ravel())
    joblib.dump(xgb_mod, f'xgb_{Y_name}.pkl')
    ### Run model on test set
    y_pred_test = xgb_mod.predict(X_test_new)
    if model_type == "class":
        performance = util.calc_performance(Y_test, y_pred_test, model_name,Y_name,X_test_new,model_type,X_new=X_new,Y_train=Y_train.values.ravel(),model=xgb_mod)
    else:
        performance = util.calc_performance(Y_test, y_pred_test, model_name,Y_name,X_test_new,model_type)
    print(performance)

    ### Calculate feature_importances

    FI.calc_permutation_feature_importance(xgb_mod,X_test_new,Y_test,model_name,model_type,Y_name)
    FI.calc_tree_gini_feature_importance(xgb_mod,model_name,model_type,Y_name)
    shap_value=FI.calc_shap_feature_importances(xgb_mod,X_test_new,X_new,Y_name,model_type,model_name)
    DrawPic(shap_value,X_test_new,Y_name)