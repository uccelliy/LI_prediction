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
        scoring=' neg_mean_squared_error'  # Default scoring for regression
        grid_pipe_en = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
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
        feature_names = en_regr_pipe.feature_names_in_
        selected_feat = np.array(feature_names)[support]
     
  
        ### Save whether or not feature is selected (also for later use)
        df_support = pd.read_csv(f"results/support_{model_type}.csv", index_col=0)
        support = pd.DataFrame(support.reshape(1,-1), columns=X_train.columns.to_list(),
                                               index=[f"{method}_{model_type}"])
        df_support = pd.concat([df_support, support])
        df_support.to_csv(f"results/support_{model_type}.csv")
        ### end elastic net
        return selected_feat, model_fs  # 返回选择的特征和变换器
    elif(method == "KPCA"):
        print("Running KernelPCA for nonlinear dimensionality reduction...")
        kpca = KernelPCA(n_components=min(50, X_train.shape[1]), kernel='rbf', random_state=random_state)

        start = perf_counter()
        X_kpca = kpca.fit_transform(X_train)
        stop = perf_counter()
        print("KPCA time:", timedelta(seconds=stop - start))
        print(f"Reduced from {X_train.shape[1]} to {X_kpca.shape[1]} dimensions.")

        return X_kpca, kpca  # 返回降维后的X 和变换器
    elif(method == "None"):
        print("No feature selection method specified, returning original features.")
        return X_train, None
    else:
        raise ValueError("Invalid input")   
    # Tuning elastic net
  
    