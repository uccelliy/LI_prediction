from core.util import n_iter, kfold, random_state
import core.util as util
import pandas as pd
from time import perf_counter
import shap
from datetime import timedelta
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import core.FeatureImportance as FI
import core.DrawPic as DrawPic
### Random forest regression
def run_rf(X_new, X_test_new, Y_train, Y_test,Y_name,groups,model_type):
    print("Running RF regression")
    model_name = "RF"
    if(model_type == "regr"):
        model = RandomForestRegressor(random_state = random_state)
        grid_rf = {'n_estimators': list(range(100, 1100, 100)), # Nr of trees
              'max_features': list(range(4, 32)), # Number of features to consider at every split
              'max_depth': list(range(2, 15)), # Max depth of tree
              'min_samples_split': list(range(2 ,11)), # Minimum number of samples required to split a node
              'min_samples_leaf': list(range(1 ,11))} # Minimum number of samples required at each leaf node
        scoring = 'neg_mean_squared_error'  # Default scoring for regression
    elif(model_type == "class"):
        model = RandomForestClassifier(random_state = random_state,class_weight='balanced')
        grid_rf = {'n_estimators': list(range(100, 1100, 100)), # Nr of trees
              'max_features': list(range(4, 32)), # Number of features to consider at every split
              'max_depth': list(range(2, 15)), # Max depth of tree
              'min_samples_split': list(range(2 ,11)), # Minimum number of samples required to split a node
              'min_samples_leaf': list(range(1 ,11)),
              'class_weight': [None, 'balanced']} 
        scoring = 'balanced_accuracy'  # Default scoring for classification  试了三种accuracy f1 balanced_accuracy最好的时balanced_accuracy 这里都可以写什么问一下chat
    else:
        raise ValueError("model_type must be 'regr' or 'class'")
    
    # Define parameter grid
    

    grid_rf_debug={'n_estimators': list(range(100, 1100, 100)),'max_depth': list(range(2, 15))}
    # Randomized search:
    rf_regr_pipe_test = RandomizedSearchCV(estimator = model, param_distributions = grid_rf, scoring = scoring, 
                                           n_iter = n_iter, cv = util.PseudoGroupCV(kfold,groups), verbose = 0, 
                                           random_state = random_state, n_jobs = -1)
    start = perf_counter()
    print("Fitting RF model")
    rf_regr_pipe_test.fit(X_new, Y_train.values.ravel())
    stop = perf_counter()
    print("Time: ", timedelta(seconds = stop -start))

    # Save results
    best_results_rf = util.save_results_cv_pipe(rf_regr_pipe_test, model_name, model_type, scoring,Y_name,X_new)
    print("Best results:")
    print(best_results_rf)

    # Initialize model
    rf = rf_regr_pipe_test.best_estimator_

    # Fits the model on the data
    rf.fit(X_new, Y_train.values.ravel())

    joblib.dump(rf, f'rf_{Y_name}.pkl')

    ### Run model on test set
    y_pred_test = rf.predict(X_test_new)
    if model_type == "class":
        performance = util.calc_performance(Y_test, y_pred_test, model_name,Y_name,X_test_new,model_type,X_new=X_new,Y_train=Y_train.values.ravel(),model=rf)
    else:
        performance = util.calc_performance(Y_test, y_pred_test, model_name,Y_name,X_test_new,model_type)
    print(performance)

    ### Calculate feature_importances
    FI.calc_permutation_feature_importance(rf,X_test_new,Y_test,model_name,model_type,Y_name)
    FI.calc_tree_gini_feature_importance(rf,model_name,model_type,Y_name)
    shap_value=FI.calc_shap_feature_importances(rf,X_test_new,X_new,Y_name,model_type,model_name)
    DrawPic(shap_value,X_test_new,Y_name)
