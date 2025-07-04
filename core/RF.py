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

### Random forest regression
def run_rf(X_new, X_test_new, Y_train, Y_test,Y_name,groups,model_type):
    print("Running RF regression")
  


    model_name = "RF"
    if(model_type == "regr"):
        model = RandomForestRegressor(random_state = random_state)
#        grid_rf = {'n_estimators': list(range(100, 1100, 100)), # Nr of trees
#               'max_features': list(range(4, 32)), # Number of features to consider at every split
#               'max_depth': list(range(2, 15)), # Max depth of tree
#               'min_samples_split': list(range(2 ,11)), # Minimum number of samples required to split a node
#               'min_samples_leaf': list(range(1 ,11))} # Minimum number of samples required at each leaf node
        scoring = 'neg_mean_squared_error'  # Default scoring for regression
    elif(model_type == "class"):
        model = RandomForestClassifier(random_state = random_state)
#        grid_rf = {'n_estimators': list(range(100, 1100, 100)), # Nr of trees
#               'max_features': list(range(4, 32)), # Number of features to consider at every split
#               'max_depth': list(range(2, 15)), # Max depth of tree
#               'min_samples_split': list(range(2 ,11)), # Minimum number of samples required to split a node
#               'min_samples_leaf': list(range(1 ,11)),
#               'class_weight': [None, 'balanced']} 
        scoring = 'accuracy'  # Default scoring for classification
    else:
        raise ValueError("model_type must be 'regr' or 'class'")
    
    # Define parameter grid
    

    grid_rf_debug={'n_estimators': list(range(100, 1100, 100)),'max_depth': list(range(2, 15))}
    # Randomized search:
    rf_regr_pipe_test = RandomizedSearchCV(estimator = model, param_distributions = grid_rf_debug, scoring = scoring, 
                                           n_iter = n_iter, cv = util.PseudoGroupCV(kfold,groups), verbose = 0, 
                                           random_state = random_state, n_jobs = -1)
    start = perf_counter()
    print("Fitting RF model")
    rf_regr_pipe_test.fit(X_new, Y_train.values.ravel())
    print("Best params: ", rf_regr_pipe_test.best_params_)
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
    performance = util.calc_performance(Y_test, y_pred_test, model_name,Y_name,X_test_new,model_type)
    print(performance)

    # ### Calculate feature_importances
    # start = perf_counter()
    # util.calc_feature_importance(rf, X_test_new, Y_test, model_name, model_type,Y_name)
    # stop = perf_counter()
    # print("Time: ", timedelta(seconds = stop -start))

    # #num_features = X_new.shape[1]
    # #max_evals = max(2 * num_features + 1, 1500)  # 或你想设的上限

    # # Calculate SHAP values
    # explainer = shap.Explainer(rf, X_test_new)
    # shap_values = explainer(X_test_new)
    # shap_values.values

    # # Save SHAP values per person
    # shap_pp_df_rf = pd.DataFrame(shap_values.values, columns = X_test_new.columns)
    # shap_pp_df_rf.to_csv(f"../results/shap_rf_pp_{Y_name}.csv")

    # # Average over all participants
    # importances = []
    # for i in range(shap_values.values.shape[1]):
    #     importances.append(np.mean(np.abs(shap_values.values[:, i])))

    # feature_importances = {fea: imp for imp, fea in zip(importances, X_new.columns.to_list())}

    # # Save averages
    # df_shap = pd.DataFrame.from_dict(feature_importances, orient = 'index')
    # df_shap.to_csv(f"../results/shap_rf_{Y_name}.csv")

    # feature_names = [
    #     a + ": " + str(b) for a,b in zip(X_test_new.columns, np.abs(shap_values.values).mean(0).round(3))
    # ]

    # # Plot top 15 features
    # plt.clf()
    # shap.summary_plot(shap_values, max_display=15,feature_names=feature_names)
    # plt.gcf().set_size_inches(15, 10)
    # plt.savefig(f'../results/SHAP_rf_top15_ntr_{Y_name}.png', bbox_inches="tight", dpi=300)
    # plt.close()
    # # Plot top 15 features (bargraph)
    # plt.clf()
    # shap.summary_plot(shap_values, plot_type="bar", max_display=15,
    #                   feature_names=feature_names)
    # plt.gcf().set_size_inches(18, 10)
    # plt.savefig(f'../results/SHAP_rf_top15_bar_ntr_{Y_name}.png', bbox_inches="tight", dpi=500)
    # plt.close()
