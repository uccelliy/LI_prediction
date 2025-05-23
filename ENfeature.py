
from sklearn.linear_model import ElasticNet

# Tuning elastic net
model_name = "EN"
model = ElasticNet()

grid_pipe_en = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
                'l1_ratio': np.arange(0.0, 1.0, 0.01)}

en_regr_pipe = RandomizedSearchCV(estimator = model, param_distributions = grid_pipe_en, 
                                  scoring = scoring_regr, n_iter = n_iter, cv = PseudoGroupCV(kfold, groups), 
                                  verbose = 0, random_state = random_state, n_jobs = -1)
start = perf_counter()
en_regr_pipe.fit(X_train, Y_train.values.ravel())
stop = perf_counter()
print("Time: ", timedelta(seconds = stop -start))

best_results_pipe = save_results_cv_pipe(en_regr_pipe, model_name, "regr", scoring_regr)
print("Best results:")
print(best_results_pipe)
print(en_regr_pipe.best_estimator_)

# Get selected features
model_fs = SelectFromModel(en_regr_pipe.best_estimator_, prefit=True)
support = model_fs.get_support()
feature_names = en_regr_pipe.feature_names_in_
selected_feat = np.array(feature_names)[support]
print(selected_feat.shape)
print(selected_feat)

### Save whether or not feature is selected (also for later use)
df_support = pd.read_csv("results/support_regr.csv", index_col=0)
support = pd.DataFrame(support.reshape(1,-1), columns=X_train.columns.to_list(),
                                       index=[f"{model_name}_{model_type}"])
df_support = pd.concat([df_support, support])
df_support.to_csv(f"results/support_{model_type}.csv")
### end elastic net