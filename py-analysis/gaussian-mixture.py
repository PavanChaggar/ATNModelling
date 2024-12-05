import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import norm 
import pandas as pd

temporal_regions = np.genfromtxt("temporal-rois.csv", delimiter='')

df =  pd.DataFrame(columns = ["region","C0_mean","C1_mean","C0_cov","C1_cov", "w0", "w1"])

for i, roi in enumerate(temporal_regions):
    print(roi)
    data = np.genfromtxt(f"roi-data/data-{int(roi)}.csv", delimiter='')
    model1 = GaussianMixture(n_components=1,random_state=123)
    model1.fit(data.reshape(-1,1))

    model2 = GaussianMixture(n_components=2,random_state=123)
    model2.fit(data.reshape(-1,1))
    aic1 = model1.aic(data.reshape(-1,1))
    print(aic1)
    aic2 = model2.aic(data.reshape(-1,1)) 
    print(aic2)
    if aic1 > aic2: 
        mean_1, mean_2 =  model2.means_[0][0], model2.means_[1][0]
        var_1, var_2 = model2.covariances_[0][0][0], model2.covariances_[1][0][0]
        w1, w2 = model2.weights_[0], model2.weights_[1]
        if w1 > w2 and w2 > 0.05:
            df.loc[i] = [int(roi), mean_1, mean_2, var_1, var_2, w1, w2]
        elif w2 > w1 and w1 > 0.05:
            df.loc[i] = [int(roi), mean_2, mean_1, var_2, var_1, w2, w1]

df.to_csv("../output/analysis-derivatives/tau-derivatives/pypart.csv")