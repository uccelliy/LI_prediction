# Import packages
# ! Note: not all packages used !
import pandas as pd
from sklearn import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold,KFold
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score, roc_auc_score, f1_score,balanced_accuracy_score
from imblearn.under_sampling import RandomUnderSampler
# Define parameters random search + resampling
n_iter = 100
cv = 10
random_state = 42
kfold = GroupKFold(n_splits=cv)
# This whole script is regression


# Save results from randomized search
def save_results_cv_pipe(model_random, model_name, model_type, scoring,Y_name,X_new):
    # Save all results
    cv_results = pd.DataFrame(model_random.cv_results_)
    file = f"../results/cv_results_{model_name}_{model_type}_{Y_name}.csv"
    cv_results.to_csv(file, index=False)

    # Save best
    best_score = model_random.best_score_
    best_params = model_random.best_params_
    best = pd.DataFrame({"n_iter": [n_iter], "cv": [cv], "scoring": [scoring], "best score": [best_score],
                         "best params": [best_params]}, index=[f"{model_name}_{model_type}"])
    df_best = pd.read_csv(f"../results/best_tuning_{Y_name}.csv", index_col=0)
    df_best = pd.concat([df_best, best])
    df_best.to_csv(f"../results/best_tuning_{Y_name}.csv")

    # Save gini feature importances for the tree based models
    if model_name.startswith("RF") or model_name.startswith("XGB"):
        if model_name.endswith("_noprs"):
            if model_name.endswith("_child_noprs"):
                file_extension = "_child_noprs"
            else:
                file_extension = "_noprs"
        elif model_name.endswith("_child"):
            file_extension = "_child"
        else:
            file_extension = ""
        gini_imp = model_random.best_estimator_.feature_importances_
        gini_imp_scaled = (gini_imp - np.min(gini_imp)) / (np.max(gini_imp) - np.min(gini_imp))
        df_featimp = pd.read_csv(f"../results/feature_importances_{model_type}{file_extension}_{Y_name}.csv", index_col=0)
        feature_importances = pd.DataFrame([gini_imp, gini_imp_scaled], columns=X_new.columns.to_list(),
                                           index=[f"{model_name}_{model_type}_gini",
                                                  f"{model_name}_{model_type}_gini_scaled"])
        df_featimp = pd.concat([df_featimp, feature_importances])
        df_featimp.to_csv(f"../results/feature_importances_{model_type}{file_extension}_{Y_name}.csv")
        
    return best

# Function to calculate 95% bootstrap interval
def bootstrap_CI2(X_test_new,y_true, y_pred, function1, function2,function3,model_type ,n_times=10000,threshold=5):
    bs_replicates1 = np.empty(n_times)
    bs_replicates2 = np.empty(n_times)
    bs_replicates3 = np.empty(n_times)
    bs_replicates4 = np.empty(n_times)
    
    # Create bootstrap replicates as much as size
    for i in range(n_times):
        idx_bs = np.random.choice(np.arange(len(y_pred)), size=len(y_pred))
        y_true_bs = y_true.to_numpy()[idx_bs].ravel()
        y_pred_bs = y_pred[idx_bs].ravel()
        
        if model_type=="regr":
            if function1 == mean_squared_error:
                bs_replicates1[i] = function1(y_true_bs, y_pred_bs)
            else:
                bs_replicates1[i] = function1(y_true_bs, y_pred_bs)
                bs_replicates3[i] = 1 - ((1 - bs_replicates1[i]) * (X_test_new.shape[0] - 1)) / (X_test_new.shape[0] - X_test_new.shape[1] - 1)

            if function2 == mean_squared_error:
                bs_replicates2[i] = function2(y_true_bs, y_pred_bs)
            else:
                bs_replicates2[i] = function2(y_true_bs, y_pred_bs)

            if function2 == mean_squared_error:
                bs_replicates4[i] = function3(y_true_bs, y_pred_bs)
            else:
                bs_replicates4[i] = function3(y_true_bs, y_pred_bs)[0,1]
        elif model_type == "class":
            bs_replicates1[i] = function1(y_true_bs, y_pred_bs)
            bs_replicates2[i] = function2(y_true_bs, y_pred_bs)
            bs_replicates3[i] = balanced_accuracy_score(y_true_bs, y_pred_bs)
            bs_replicates4[i] = function3(y_true_bs, y_pred_bs)

    # Get 95% confidence interval
    ci_lower1 = np.percentile(bs_replicates1, threshold)
    ci_upper1 = np.percentile(bs_replicates1, (100 - threshold))

    ci_lower2 = np.percentile(bs_replicates2, threshold)
    ci_upper2 = np.percentile(bs_replicates2, (100 - threshold))

    ci_lower3 = np.percentile(bs_replicates3, threshold)
    ci_upper3 = np.percentile(bs_replicates3, (100 - threshold))
    
    ci_lower4 = np.percentile(bs_replicates4, threshold)
    ci_upper4 = np.percentile(bs_replicates4, (100 - threshold))
    
    result_list = [ci_lower1, ci_upper1, ci_lower2, ci_upper2, ci_lower3, ci_upper3,ci_lower4, ci_upper4]

    return result_list

def permutation_Pvalue(X_test_new,y_true, y_pred, score1,score2,score3,score4,function1, function2,function3,model_type ,X_new=None,Y_train=None,model=None,n_times=10000):
    bs_replicates1 = np.empty(n_times)
    bs_replicates2 = np.empty(n_times)
    bs_replicates3 = np.empty(n_times)
    bs_replicates4 = np.empty(n_times)
    
    # Create bootstrap replicates as much as size
    for i in range(n_times):
        
        if model_type=="regr":
            idx_bs = np.random.permutation(np.arange(len(y_pred)))
            y_true_bs = y_true.to_numpy()[idx_bs].ravel()
            y_pred_bs = y_pred.ravel()
            if function1 == mean_squared_error:
                bs_replicates1[i] = function1(y_true_bs, y_pred_bs)
            else:
                bs_replicates1[i] = function1(y_true_bs, y_pred_bs)
                bs_replicates3[i] = 1 - ((1 - bs_replicates1[i]) * (X_test_new.shape[0] - 1)) / (X_test_new.shape[0] - X_test_new.shape[1] - 1)

            if function2 == mean_squared_error:
                bs_replicates2[i] = function2(y_true_bs, y_pred_bs)
            else:
                bs_replicates2[i] = function2(y_true_bs, y_pred_bs)

            if function2 == mean_squared_error:
                bs_replicates4[i] = function3(y_true_bs, y_pred_bs)
            else:
                bs_replicates4[i] = function3(y_true_bs, y_pred_bs)[0,1]
        elif model_type == "class":
            idx_bs = np.random.permutation(np.arange(len(y_pred)))
            y_true_bs = y_true.to_numpy()[idx_bs].ravel()
            y_pred_bs = y_pred.ravel()
            # model_perm = clone(model)
            # y_perm = np.random.permutation(Y_train)
            # y_pred_bs = model_perm.fit(X_new, y_perm).predict(X_test_new)
            # y_true_bs = y_true.to_numpy().ravel()
            bs_replicates1[i] = function1(y_true_bs, y_pred_bs)
            bs_replicates2[i] = function2(y_true_bs, y_pred_bs)
            bs_replicates3[i] = balanced_accuracy_score(y_true_bs, y_pred_bs)
            bs_replicates4[i] = function3(y_true_bs, y_pred_bs, average='weighted')

    p1 = np.mean(bs_replicates1 >= score1)
    p2 = np.mean(bs_replicates2 >= score2)
    p3 = np.mean(bs_replicates3 >= score3)
    p4 = np.mean(bs_replicates4 >= score4)
    # Get 95% confidence interval
    
    
    result_list = [p1,p2,p3,p4]

    return result_list

def rmse_func(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Define function to calculate performance measures regression
def calc_performance(y_test, y_pred, model_name, Y_name,X_test_new,model_type,X_new=None,Y_train=None,model=None):
    if model_type not in ["regr", "class"]:
        raise ValueError("model_type must be 'regr' or 'class'")
    elif model_type == "regr":
        r2 = r2_score(y_test, y_pred)
        adjusted_r2 = 1 - ((1 - r2) * (X_test_new.shape[0] - 1)) / (X_test_new.shape[0] - X_test_new.shape[1] - 1)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = rmse_func(y_test, y_pred)
        corr =  np.corrcoef(np.ravel(y_test), np.ravel(y_pred))[0, 1]
        result_list_tmp = permutation_Pvalue(X_test_new,y_test, y_pred, r2,rmse,adjusted_r2,corr,r2_score, rmse_func,np.corrcoef,"regr")
        performance = [r2, result_list_tmp[0], mae, rmse,result_list_tmp[1], adjusted_r2, result_list_tmp[2],corr, result_list_tmp[3]]
    elif model_type == "class":
        accuracy = accuracy_score(y_test, y_pred)
        cohen_kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        balanced_accuracy= balanced_accuracy_score(y_test, y_pred)
        result_list_tmp = permutation_Pvalue(X_test_new,y_test, y_pred, accuracy,cohen_kappa,balanced_accuracy,
                                          f1,accuracy_score, cohen_kappa_score,f1_score,"class",X_new=X_new,Y_train=Y_train,model=model)
        performance = [accuracy, result_list_tmp[0] ,cohen_kappa, result_list_tmp[1], balanced_accuracy, result_list_tmp[2],f1, result_list_tmp[3]]
    # save performance
    df_perf = pd.read_csv(f"../results/performance_{model_type}_{Y_name}.csv", index_col=0)
    perf = pd.DataFrame([performance], columns=df_perf.columns.tolist(), index=[f"{model_name}_{model_type}"])
    df_perf = pd.concat([df_perf, perf])
    df_perf.to_csv(f"../results/performance_{model_type}_{Y_name}.csv")

    # save predictions
    if model_name.endswith("_child") or model_name.endswith("_child_noprs"):
        file_extension = "_child"
    else:
        file_extension = ""

    df_pred = pd.read_csv(f"../results/predictions_{model_type}{file_extension}_{Y_name}.csv", index_col=0)
    pred = pd.DataFrame(y_pred, columns=[f"{model_name}_{model_type}"], index=df_pred.index.tolist())
    df_pred = pd.concat([df_pred, pred], axis=1)
    df_pred.to_csv(f"../results/predictions_{model_type}{file_extension}_{Y_name}.csv")

    return perf

# function that calculates permutation importances
def calc_feature_importance(estimator, X, y, model_name, model_type,Y_name):
    if model_type == "class":
        scoring = 'balanced_accuracy'
    elif model_type == "regr":
        scoring = "neg_root_mean_squared_error"
    else:
        scoring = None

    feature_imp = permutation_importance(estimator, X, y, scoring=scoring, n_jobs = -1, random_state=42).importances_mean
    feature_imp_scaled = (feature_imp - np.min(feature_imp)) / (np.max(feature_imp) - np.min(feature_imp))

    # Save feature importances
    if model_name.endswith("_noprs"):
        if model_name.endswith("_child_noprs"):
            file_extension = "_child_noprs"
        else:
            file_extension = "_noprs"
    elif model_name.endswith("_child"):
        file_extension = "_child"
    else:
        file_extension = ""

    df_perm_featimp = pd.read_csv(f"../results/perm_feature_importances_{model_type}{file_extension}_{Y_name}.csv", index_col=0)
    feature_importances = pd.DataFrame([feature_imp, feature_imp_scaled], columns=df_perm_featimp.columns.tolist(),
                                       index=[f"{model_name}_{model_type}", f"{model_name}_{model_type}_scaled"])
    df_perm_featimp = pd.concat([df_perm_featimp, feature_importances])
    df_perm_featimp.to_csv(f"../results/perm_feature_importances_{model_type}{file_extension}_{Y_name}.csv")

    return feature_importances

# setting up the group split
from sklearn.model_selection import  train_test_split

class PseudoGroupCV:
    def __init__(self, cv_obj, groups):
        self.cv = cv_obj
        self.groups=groups
    def split(self, X,y, groups=None):
        return self.cv.split(X,y, groups=self.groups)
    def get_n_splits(self, X, y, groups):
        return self.cv.get_n_splits(X,y, groups)
    
def prepare_data(data1,data2,name,model_type="regr"):
     # Define random_state
    data1 = data1.reset_index(drop=True)
    data2 = data2.reset_index(drop=True)

    # Step 2: Check IID 不重复（如果你的代码逻辑假设 IID 唯一）
    assert not data1['IID'].duplicated().any(), "Duplicate IID in data1"
    assert not data2['IID'].duplicated().any(), "Duplicate IID in data2"
    common_ids = pd.Series(np.intersect1d(data1['IID'], data2['IID']))

    data1_a = data1[data1['IID'].isin(common_ids)].sort_values(by='IID').reset_index(drop=True)
    data2_a = data2[data2['IID'].isin(common_ids)].sort_values(by='IID').reset_index(drop=True) 
    
    X_train, X_test, Y_train, Y_test = train_test_split(data1_a, data2_a, test_size=0.2, random_state=42)
    groups = X_train['FID'].reset_index(drop=True)

    # 保留你要预测的标签列（假设行为名称为 behav_name）
    Y_train = Y_train[[name]].reset_index(drop=True)
    Y_test = Y_test[[name]].reset_index(drop=True)

    # 去掉前两列（IID, FID），只保留特征
    X_train = X_train.iloc[:, 2:].reset_index(drop=True)
    X_test = X_test.iloc[:, 2:].reset_index(drop=True)

    # 对分类任务进行欠采样
    # if model_type == "class":
    #     rus = RandomUnderSampler(random_state=42)
    #     X_train, Y_train_resampled = rus.fit_resample(X_train, Y_train[name])  # 注意只传一列
    #     Y_train = pd.DataFrame({name: Y_train_resampled})  # 转回DataFrame
    #     groups = groups.loc[rus.sample_indices_].reset_index(drop=True)  # 同步更新 groups
    
    # predictions_regr.csv
    df_pred_init = pd.DataFrame(Y_test.values, index = X_test.index.tolist(), columns = ["y_test"])
    df_pred_init.to_csv(f"../results/predictions_{model_type}_{name}.csv")
    
    ## ### Initialize dataframes
    ##  support_regr.csv
    df_support_init = pd.DataFrame(columns = X_train.columns.tolist())
    df_support_init.to_csv(f"../results/support_{model_type}_{name}.csv")
    
    # initialize permutation feature importances with selected features

    
    return X_train, X_test, Y_train, Y_test,groups

def result_file_init(X_train,model_type, name):
    df_perm_featimp_init = pd.DataFrame(columns = X_train.columns.tolist())
    df_perm_featimp_init.to_csv(f"../results/perm_feature_importances_{model_type}_{name}.csv")

    # # initialize feature_importances_regr.csv
    # these are based on training set (within model)
    df_featimp_init = pd.DataFrame(columns = X_train.columns.tolist())
    df_featimp_init.to_csv(f"../results/feature_importances_{model_type}_{name}.csv")
    
def result_file_init_best(behav_name):
    df_best_init = pd.DataFrame(columns = ["n_iter", "cv", "scoring", "best score", "best params"])
    df_best_init.to_csv(f'../results/best_tuning_{behav_name}.csv')


def result_file_init_performance(behav_name, model_type):
    if model_type == "regr":
        df_perf_init = pd.DataFrame(columns = ["r2", "p1", "mae", "rmse", "p2", "adj_r2", "p3","r", "p4"])
        df_perf_init.to_csv(f'../results/performance_regr_{behav_name}.csv')
    elif model_type == "class":
        df_perf_init = pd.DataFrame(columns = ["accuracy", "p1", "cohen_kappa","p2","balance_acc","p3","f1","p4"])
        df_perf_init.to_csv(f'../results/performance_class_{behav_name}.csv')

    

#不能在这里分训练集和测试集，应该是分好，传入模型,传入的Y也是一个矩阵，我们要对每一列分别处理，并且把每一列的结果分别保存下来






