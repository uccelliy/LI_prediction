from sklearn.model_selection import GroupKFold
import pandas as pd
# Define parameters random search + resampling
n_iter = 100
cv = 10
scoring_class = 'roc_auc'
scoring_regr = 'neg_root_mean_squared_error'
random_state = 42
kfold = GroupKFold(n_splits=cv)
# This whole script is regression
model_type = "regr"

# Read in the data
input_file = '../X_train.txt'
df = pd.read_csv(input_file, sep="\t", decimal='.', encoding='cp1252')
groups = df['IID']
X_train = df.iloc[:, 2:]
print(df.shape, X_train.shape)

input_file2 = '../X_test.txt'
df2 = pd.read_csv(input_file2, sep="\t", decimal='.', encoding='cp1252')
X_test = df2.iloc[:, 2:]
print(df2.shape, X_test.shape)

input_file3 = '../card_test.txt'
df3 = pd.read_csv(input_file3, sep="\t", decimal='.', encoding='cp1252')
Y_test = df3.iloc[:, 2:]
print(df3.shape, Y_test.shape)

input_file4 = '../card_train.txt'
df4 = pd.read_csv(input_file4, sep="\t", decimal='.', encoding='cp1252')
Y_train = df4.iloc[:, 2:]
print(df4.shape, Y_train.shape)

# # best_tuning.csv
df_best_init = pd.DataFrame(columns = ["n_iter", "cv", "scoring", "best score", "best params"])
print(df_best_init)
df_best_init.to_csv("../results/best_tuning.csv")

# ### Initialize dataframes
#  support_regr.csv
df_support_init = pd.DataFrame(columns = X_train.columns.tolist())
print(df_support_init)
df_support_init.to_csv("../results/support_regr.csv")

# performance_regr.csv
df_perf_init = pd.DataFrame(columns = ["r2", "ci_lower1", "ci_upper1", "mae", "rmse", "ci_lower2", "ci_upper2", "adj_r2", "ci_lower3", "ci_upper3"])
print(df_perf_init)
df_perf_init.to_csv("../results/performance_regr.csv")

# predictions_regr.csv
df_pred_init = pd.DataFrame(Y_test.values, index = X_test.index.tolist(), columns = ["y_test"])
print(df_pred_init)
df_pred_init.to_csv("../results/predictions_regr.csv")

### Read in file to create new data
#support_file = '../results/support_regr.csv'
#sel_feat = pd.read_csv(support_file, sep=",", decimal='.', encoding='cp1252')
#sel_feat = sel_feat.iloc[:, 1:]
#sel_feat = sel_feat.to_numpy()
#sel_feat = sel_feat.reshape(-1)

# Create new smaller dataset (only selected features)
#X_new = X_train.loc[:, sel_feat]
#X_test_new = X_test.loc[:, sel_feat]
X_new = X_train
X_test_new= X_test

print(X_new.shape)
print(X_test_new.shape)

# initialize permutation feature importances with selected features
df_perm_featimp_init = pd.DataFrame(columns = X_new.columns.tolist())
print(df_perm_featimp_init)
df_perm_featimp_init.to_csv("../results/perm_feature_importances_regr.csv")

# # initialize feature_importances_regr.csv
# these are based on training set (within model)
df_featimp_init = pd.DataFrame(columns = X_new.columns.tolist())
print(df_featimp_init)
df_featimp_init.to_csv("../results/feature_importances_regr.csv")

# Check number of missings
print(X_train.shape, X_test.shape)
print("nr of NaN X_train:", X_train.isnull().sum().sum())
print("nr of NaN X_test:", X_test.isnull().sum().sum())
