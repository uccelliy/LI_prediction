import pandas as pd
from joblib import Parallel, delayed
import os
from tqdm import tqdm  # 可选：进度条
import util
import RF
import SVM
import XGB
import Stack

def run_models_for_behavior(behav_name, X, Y_all):
    """运行单个 behavior 的所有模型（RF/SVM/XGB 并行，Stack 最后）"""
    Y = Y_all
    X_train, X_test, Y_train, Y_test, groups = util.prepare_data(X, Y, behav_name)

    # 初始化结果文件
    df_best_init = pd.DataFrame(columns = ["n_iter", "cv", "scoring", "best score", "best params"])
    df_best_init.to_csv(f'../results/best_tuning_{behav_name}.csv')


    # performance_regr.csv
    df_perf_init = pd.DataFrame(columns = ["r2", "ci_lower1", "ci_upper1", "mae", "rmse", "ci_lower2", "ci_upper2", "adj_r2", "ci_lower3", "ci_upper3"])
    df_perf_init.to_csv(f'../results/performance_regr_{behav_name}.csv')

    # 并行运行 RF/SVM/XGB
    Parallel(n_jobs=3)(
        delayed(model_func)(X_train, X_test, Y_train, Y_test, behav_name, groups)
        for model_func in [RF.run_rf, SVM.run_svm, XGB.run_xgb]
    )

    # Stacking 必须等待前三个模型完成
    Stack.run_stack(X_train, X_test, Y_train, Y_test, behav_name, groups)

# 主循环（并行处理所有 behaviors）
def main():
    # Read in the data 可以修改路径来读目标文件
    os.chdir(os.path.dirname(__file__))
    input_file = '../dti_5e6_aligned.txt'
    X = pd.read_csv(input_file, sep="\t", decimal='.', encoding='cp1252')
    print(X.shape)

    input_file4 = '../cog_5e6_aligned.txt'
    Y_all = pd.read_csv(input_file4, sep="\t", decimal='.', encoding='cp1252')
    print(Y_all.shape)
    behaviors = Y_all.columns[2:]  # 跳过前两列（假设前两列非目标变量）
   
    # 并行运行所有 behaviors
    Parallel(n_jobs=os.cpu_count() - 1)(
        delayed(run_models_for_behavior)(behav_name, X, Y_all[["IID","FID",behav_name]])
        for behav_name in tqdm(behaviors, desc="Processing behaviors",ncols=100, ascii=True)  # 可选：进度条
    )

if __name__ == "__main__":
    main()
