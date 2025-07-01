
import pandas as pd
import core.util as util
import core.RF as RF
import core.SVM as SVM
import core.XGB as XGB
import core.Stack as Stack
import os
import core.FeatureSelection as FS


# Read in the data
model_type = "class"
os.chdir(os.path.dirname(__file__))
input_file = 'C:/Users/77260/Desktop/111/dti_5e6.txt'
X = pd.read_csv(input_file, sep="\t", decimal='.', encoding='cp1252')
print(X.shape)
input_file4 = 'C:/Users/77260/Desktop/111/class_test.txt'
Y_all = pd.read_csv(input_file4, sep="\t", decimal='.', encoding='cp1252')
print(Y_all.shape)
behaviors = Y_all.columns[2:]  # 跳过前两列（假设前两列非目标变量）

for behav_name in behaviors:

    print("Running model for behavior: ", behav_name)
    
    Y=Y_all[["IID","FID",behav_name]]
    X_train, X_test, Y_train, Y_test, groups = util.prepare_data(X, Y, behav_name,model_type)
    X_train_new, transform = FS.feature_selection(X_train, Y_train, groups, model_type=model_type, method="None")
    if transform is None:
        print(f"No features selected for {behav_name}. Skipping...")
        X_test_new = X_test # 如果没有特征选择，直接使用原始测试集
    else:
        X_test_new = transform.transform(X_test)
    


    # # best_tuning.csv
    df_best_init = pd.DataFrame(columns = ["n_iter", "cv", "scoring", "best score", "best params"])
    df_best_init.to_csv(f'../results/best_tuning_{behav_name}.csv')


    # performance_regr.csv
    if model_type == "regr":
        df_perf_init = pd.DataFrame(columns = ["r2", "p1", "mae", "rmse", "p2", "adj_r2", "p3","r", "p4"])
        df_perf_init.to_csv(f'../results/performance_regr_{behav_name}.csv')
    elif model_type == "class":
        df_perf_init = pd.DataFrame(columns = ["accuracy", "p1", "roc_auc","p2","balance_acc","p3","f1","p4"])
        df_perf_init.to_csv(f'../results/performance_class_{behav_name}.csv')


    RF.run_rf(X_train_new, X_test_new, Y_train, Y_test, behav_name, groups,model_type)
    SVM.run_svm(X_train_new, X_test_new, Y_train, Y_test, behav_name, groups,model_type)
    XGB.run_xgb(X_train_new, X_test_new, Y_train, Y_test, behav_name, groups,model_type)


