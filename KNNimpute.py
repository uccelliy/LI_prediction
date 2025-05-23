


## KNN imputation
start = time.perf_counter()
cols = X_train.columns.tolist()
indx_train = X_train.index.tolist()
indx_test = X_test.index.tolist()
k = round(math.sqrt(X_train.shape[0]))

# Impute train
imputer = KNNImputer(n_neighbors = k, weights = 'distance')
imputed = imputer.fit_transform(X_train) # becomes np array
X_train = pd.DataFrame(imputed, columns = cols)
# X_train = pd.DataFrame(imputed, columns = cols, index = indx_train)
print("Train imputed \n")

# Impute test
imputed_test = imputer.fit_transform(X_test) # becomes np array
X_test = pd.DataFrame(imputed_test, columns = cols)
# X_test = pd.DataFrame(imputed_test, columns = cols, index = indx_test)
print("Test imputed \n")

stop = time.perf_counter()
time = stop -start

print("time knn:", time, "\n")

## END KNN imputation