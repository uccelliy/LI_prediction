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
    start = perf_counter()
    util.calc_feature_importance(svm, X_test_new, Y_test, model_name, model_type, Y_name)
    stop = perf_counter()
    print("Time: ", timedelta(seconds = stop-start))

    num_features = X_new.shape[1]
    max_evals = max(2 * num_features + 1, 1500)  # 或你想设的上限

    # Calculate SHAP values
    #explainer = shap.Explainer(svm.predict, X_test_new, max_evals)
    #shap_values = explainer(X_test_new)
    background = shap.sample(X_new, 100)  # 或 X_train[:100]

# 创建 KernelExplainer
    explainer = shap.KernelExplainer(svm.predict, background)

# 计算 SHAP 值，使用你期望的 max_evals 控制近似精度
    shap_values = explainer.shap_values(X_test_new, nsamples=max_evals)

    # Save SHAP values per person
    shap_pp_df_svm = pd.DataFrame(shap_values.values, columns = X_test_new.columns)
    shap_pp_df_svm.to_csv(f"../results/shap_svm_pp_{Y_name}.csv")

    # Average over all participants
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))

    feature_importances = {fea: imp for imp, fea in zip(importances, X_new.columns.to_list())}

    # Save averages
    df_shap = pd.DataFrame.from_dict(feature_importances, orient = 'index')
    df_shap.to_csv(f"../results/shap_svm_{Y_name}.csv")

    feature_names = [
        a + ": " + str(b) for a,b in zip(X_test_new.columns, np.abs(shap_values.values).mean(-1).round(3))
    ]


    # Plot top 14 features
    plt.clf()
    shap.summary_plot(shap_values, max_display=14,feature_names=feature_names)
    plt.gcf().set_size_inches(14, 10)
    plt.savefig(f'../results/SHAP_svm_top14_ntr_{Y_name}.png', bbox_inches="tight", dpi=300)
    plt.close()
    # Plot top 14 features (bargraph)
    plt.clf()
    shap.summary_plot(shap_values, plot_type="bar", max_display=14,
                      feature_names=feature_names)
    plt.gcf().set_size_inches(17, 10)
    plt.savefig(f'../results/SHAP_svm_top14_bar_ntr_{Y_name}.png', bbox_inches="tight", dpi=500)
    plt.close()
