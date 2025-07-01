from sklearn.decomposition import KernelPCA
from core.util import n_iter, kfold, random_state
import core.util as util
import pandas as pd
from time import perf_counter
from sklearn.linear_model import ElasticNet
from datetime import timedelta
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.feature_selection import SelectFromModel

def feature_selection(X_train, Y_train, groups,model_type="regr",method="KPCA"):
    """
    Perform feature selection using Elastic Net regression.
    
    Parameters:
    - X_train: Training features.
    - Y_train: Training target variable.
    - kfold: K-fold cross-validation object.
    - groups: Groups for cross-validation.
    - scoring_regr: Scoring metric for regression.
    - n_iter: Number of iterations for RandomizedSearchCV.
    - random_state: Random state for reproducibility.
    - model_type: Type of model ('regr' or 'class').
    
    Returns:
    - None
    """
    if(model_type == "regr" and method =="EN"):
        model = ElasticNet()
        scoring='neg_mean_squared_error'  # Default scoring for regression
        grid_pipe_en = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1,  1.0, 10.0, 100.0],
                    'l1_ratio': np.arange(0.0, 1.0, 0.01)}

        en_regr_pipe = RandomizedSearchCV(estimator = model, param_distributions = grid_pipe_en, 
                                      scoring = scoring, n_iter = n_iter, cv = util.PseudoGroupCV(kfold, groups), 
                                      verbose = 0, random_state = random_state, n_jobs = -1)
        start = perf_counter()
        en_regr_pipe.fit(X_train, Y_train.values.ravel())
        stop = perf_counter()
        print("Time: ", timedelta(seconds = stop -start))
        util.save_results_cv_pipe(en_regr_pipe, method, model_type, scoring)
        
        # Get selected features
        model_fs = SelectFromModel(en_regr_pipe.best_estimator_, prefit=True)
        support = model_fs.get_support()
        X_EN = X_train.loc[:,support]  # Select only the features that are supported
  
        ### Save whether or not feature is selected (also for later use)
        support = pd.DataFrame(support.reshape(1,-1), columns=X_train.columns.to_list(),
                                               index=[f"{method}_{model_type}"])
        support.to_csv(f"../results/support_EN_{model_type}.csv")
        ### end elastic net
        return X_EN, model_fs  # 返回选择的特征和变换器
    elif(method == "KPCA"):
        print("Running KernelPCA for nonlinear dimensionality reduction...")
        kpca = KernelPCA(n_components=min(50, X_train.shape[1]), kernel='rbf', random_state=random_state)

        start = perf_counter()
        X_kpca = kpca.fit_transform(X_train)
        stop = perf_counter()
        print("KPCA time:", timedelta(seconds=stop - start))
        print(f"Reduced from {X_train.shape[1]} to {X_kpca.shape[1]} dimensions.")
        support = pd.DataFrame(X_kpca, index=X_train.index, columns=[f"PC{i+1}" for i in range(X_kpca.shape[1])])
        support.to_csv(f"../results/support_KPCA_{model_type}.csv")
        return support, kpca  # 返回降维后的X 和变换器
    elif(method == "None"):
        print("No feature selection method specified, returning original features.")
        return X_train, None
    else:
        raise ValueError("Invalid input")   
    # Tuning elastic net
  
    