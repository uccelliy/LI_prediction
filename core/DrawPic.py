import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def draw_pic(shap_values,X_test_new,Y_name):
    feature_names = [
        a + ": " + str(b) for a,b in zip(X_test_new.columns, np.abs(shap_values.values).mean(0).round(3))
    ]
    # Plot top 15 features
    plt.clf()
    shap.summary_plot(shap_values, max_display=15,feature_names=feature_names)
    plt.gcf().set_size_inches(15, 10)
    plt.savefig(f'../results/SHAP_rf_top15_ntr_{Y_name}.png', bbox_inches="tight", dpi=300)
    plt.close()
    # Plot top 15 features (bargraph)
    plt.clf()
    shap.summary_plot(shap_values, plot_type="bar", max_display=15,
                      feature_names=feature_names)
    plt.gcf().set_size_inches(18, 10)
    plt.savefig(f'../results/SHAP_rf_top15_bar_ntr_{Y_name}.png', bbox_inches="tight", dpi=500)
    plt.close()