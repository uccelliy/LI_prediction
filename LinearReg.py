

### Linear regression

start = perf_counter()

# Set up model
model_name = "Lin" # "Lin" / "Lin_noprs" / "Lin_child" / "Lin_child_noprs"
lin_regr = LinearRegression()

# Train model
lin_regr.fit(X_new, Y_train)
y_pred_train = lin_regr.predict(X_new)

# Performance in training set
rmse_train = root_mean_squared_error(Y_train, y_pred_train)
r2_train = r2_score(Y_train, y_pred_train)
stop = perf_counter()
print("Time: ", timedelta(seconds = stop -start))
print("RMSE train:", rmse_train)
print("R2 train:", r2_train)

# Run model on test set
y_pred_test = lin_regr.predict(X_test_new)
performance = calc_performance_regression2(Y_test, y_pred_test, model_name)
print(performance)

### Calculate feature_importances
start = perf_counter()
feat_imp = calc_feature_importance(lin_regr, X_test_new, Y_test, model_name, "regr")
stop = perf_counter()
print("Time: ", timedelta(seconds = stop -start))
print(feat_imp)