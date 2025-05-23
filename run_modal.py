
import pandas as pd
import util
import RF
import SVM
import XGB
import Stack


# Read in the data
input_file = '../X.txt'
X = pd.read_csv(input_file, sep="\t", decimal='.', encoding='cp1252')
print(X.shape)

input_file4 = '../Y.txt'
Y_all = pd.read_csv(input_file4, sep="\t", decimal='.', encoding='cp1252')
print(Y_all.shape)

for i in range(Y_all.shape[1]-2):

    print("Running model for behavior: ", Y_all.columns[i+2])
    Y=Y_all.iloc[:,i+2]
    behav_name=Y_all.columns[i+2]
    X_train,X_test,Y_train,Y_test,groups = util.prepare_data(X,Y,behav_name)
    


    # # best_tuning.csv
    df_best_init = pd.DataFrame(columns = ["n_iter", "cv", "scoring", "best score", "best params"])
    print(df_best_init)
    df_best_init.to_csv(f'../results/best_tuning_{behav_name}.csv')


    # performance_regr.csv
    df_perf_init = pd.DataFrame(columns = ["r2", "ci_lower1", "ci_upper1", "mae", "rmse", "ci_lower2", "ci_upper2", "adj_r2", "ci_lower3", "ci_upper3"])
    print(df_perf_init)
    df_perf_init.to_csv(f'../results/performance_regr_{behav_name}.csv')


    RF.run_rf(X_train, X_test, Y_train, Y_test,behav_name,groups)
    SVM.run_svm(X_train, X_test, Y_train, Y_test,behav_name,groups)
    XGB.run_xgb(X_train, X_test, Y_train, Y_test,behav_name,groups)
    Stack.run_stack(X_train, X_test, Y_train, Y_test,behav_name,groups)

    # Save results
    #df_best = pd.read_csv(f'../results/{behav_name}_best_tuning.csv', index_col=0)
    #df_perf = pd.read_csv(f'../results/{behav_name}_performance_regr.csv', index_col=0)
#
    ## Save predictions
    #df_pred = pd.read_csv(f"../results/{behav_name}_predictions_regr.csv", index_col=0)
#

### Read in file to create new data
#support_file = '../results/support_regr.csv'
#sel_feat = pd.read_csv(support_file, sep=",", decimal='.', encoding='cp1252')
#sel_feat = sel_feat.iloc[:, 1:]
#sel_feat = sel_feat.to_numpy()
#sel_feat = sel_feat.reshape(-1)

# Create new smaller dataset (only selected features)
#X_new = X_train.loc[:, sel_feat]
#X_test_new = X_test.loc[:, sel_feat]


# Check number of missings
#print(X_train.shape, X_test.shape)
#print("nr of NaN X_train:", X_train.isnull().sum().sum())
#print("nr of NaN X_test:", X_test.isnull().sum().sum())
