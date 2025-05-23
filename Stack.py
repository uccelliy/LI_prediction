import util
from util import n_iter, kfold, random_state, scoring_regr
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from mlxtend.regressor import StackingCVRegressor
from time import perf_counter
from datetime import timedelta
import joblib
## Stacked model
def run_stack(X_new, X_test_new, Y_train, Y_test,Y_name,groups):
    print("Running Stacked model")
# Set up model
    model_name = "Stack"
    f_clf = xgb.XGBRegressor(random_state=random_state)

    rf=joblib.load(f'rf_{Y_name}.pkl')
    svr=joblib.load(f'svr_{Y_name}.pkl')
    xgb_mod=joblib.load(f'xgb_{Y_name}.pkl')

# Define parameter grid
    grid_pipe_xgb = {'n_estimators': list(range(100, 1100, 100)),
                'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3],
                'gamma': [i/10 for i in range(0,6)],
                'max_depth': list(range(2, 16)),
                'min_child_weight': list(range(1,11)),
                'subsample': [x/10 for x in range(2, 11)],
                'colsample_bytree': [x/10 for x in range(2, 11)],
                'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5],
                'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]}

# Randomized search:
    grid_search_meta = RandomizedSearchCV(estimator = f_clf,
                                param_distributions  = grid_pipe_xgb,
                                scoring = scoring_regr,
                                cv = util.PseudoGroupCV(kfold,groups),
                                verbose = 2, random_state = random_state,
                                n_jobs = -1, n_iter=n_iter)

    stack_pipeline = StackingCVRegressor(
    regressors = (svr, rf, xgb_mod),
    meta_regressor = grid_search_meta,
    refit =False,
    verbose=2, n_jobs = 1)

    start = perf_counter()
    print("Fitting Stacked model")
    stack_pipeline.fit(X_new, Y_train.values.ravel())
    print("Best params: ", stack_pipeline.meta_regressor.best_params_)
    stop = perf_counter()
    print("Time: ", timedelta(seconds = stop -start))

    # Save results
    best_results_stack = util.save_results_cv_pipe(grid_search_meta, model_name, "regr", scoring_regr,Y_name,X_new)
    print("Best results:")
    print(best_results_stack)

    ### Run model on test set
    y_pred_test = stack_pipeline.predict(X_test_new)
    performance = util.calc_performance_regression2(Y_test, y_pred_test, model_name,Y_name,X_test_new)
    print(performance)

