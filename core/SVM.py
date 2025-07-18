from core.util import n_iter, kfold, random_state
import pandas as pd
import core.util as util
from time import perf_counter
import shap
from datetime import timedelta
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC, SVR
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import core.FeatureImportance as FI
import core.DrawPic as DrawPic
## Support vector machines

def run_svm( X_new, X_test_new, Y_train, Y_test,Y_name,groups,model_type):
    print("Running SVM regression")
    model_name = "SVM"
    grid_svm_debug={'C': [0.01, 0.1, 1, 10], 'kernel': ["linear"]}

    # Set up model
    if(model_type == "regr"):
        model = SVR()
        # Define parameter grid
        grid_pipe_svm = {'C': [0.01, 0.1, 1, 10],
               'kernel': ["linear", "poly", "rbf", "sigmoid"],
              'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, "scale", "auto"]}
        scoring = 'neg_mean_squared_error'  # Default scoring for regression
    elif(model_type == "class"):
        grid_pipe_svm = {'C': [0.1, 1, 10, 100],
               'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
               'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # RBF/poly/sigmoid用
               'degree': [2, 3, 4],  # 仅对 kernel='poly' 有效
                'class_weight': [None, 'balanced']}
        model = SVC(probability=True,class_weight='balanced')
        scoring = 'balanced_accuracy'  # Default scoring for classification
    else:
        raise ValueError("model_type must be 'regr' or 'class'")
    

    # Randomized search:
    svm_regr = RandomizedSearchCV(estimator = model, param_distributions = grid_pipe_svm, scoring = scoring, 
                                  n_iter = n_iter, cv = util.PseudoGroupCV(kfold,groups), verbose = 2, 
                                  random_state = random_state, n_jobs = -1)
    start = perf_counter()
    print("Fitting SVM model")
    svm_regr.fit(X_new, Y_train.values.ravel())
    stop = perf_counter()
    print("Time: ", timedelta(seconds = stop -start))

    # Save results
    best_results = util.save_results_cv_pipe(svm_regr, model_name, model_type, scoring, Y_name,X_new)
    print("Best results:")
    print(best_results)
    stop = perf_counter()
    print("Time: ", timedelta(seconds = stop-start))

    # Initialize model
    svm = svm_regr.best_estimator_

    # Fits the model on the data
    svm.fit(X_new, Y_train.values.ravel())

    joblib.dump(svm, f'svm_{Y_name}.pkl')

    ### Run model on test set
    y_pred_test = svm.predict(X_test_new)
    if model_type == "class":
        performance = util.calc_performance(Y_test, y_pred_test, model_name, Y_name,X_test_new,model_type,X_new=X_new,Y_train=Y_train.values.ravel(),model=svm)
    else:
        performance = util.calc_performance(Y_test, y_pred_test, model_name, Y_name,X_test_new,model_type)
    print(performance)

    ### Calculate feature_importances
    FI.calc_permutation_feature_importance(svm,X_test_new,Y_test,model_name,model_type,Y_name)
    #shap_value=FI.calc_shap_feature_importances(svm,X_test_new,X_new,Y_name,model_type,model_name)
    #DrawPic(shap_value,X_test_new,Y_name)