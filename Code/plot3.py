Attempt = '11th'
Run = 'test_1'
gridsearch = 'yes'
Res = '05'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from matplotlib import pyplot as plt
# from science.extraTree_model_base_coslon import cal_statistic

Data = pd.read_csv('../lib/result_xgb_aus.csv', sep=',')

actual = np.array(Data['hf'][0:133], dtype=np.float64).reshape((-1, 1))
pred = np.array(Data['hf_pred'][0:133], dtype=np.float64).reshape((-1, 1))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots()

scatter1 = ax.scatter(actual, pred, c='blue', edgecolor='white', label = 'XGBoost')

ax.set(xlim=(-10, 180), xticks=np.arange(0, 180, 25),
       ylim=(-10, 180), yticks=np.arange(0, 180, 25))

plt.xlabel('Actual SHF Value (mW/m$^2$)', fontdict={'size': 12, 'color': 'black'})
plt.ylabel('Predicted SHF Value (mW/m$^2$)', fontdict={'size': 12, 'color': 'black'})
plt.plot([0, 175], [0, 175], linestyle='dashed', c='black', label = 'ideal predictor')
plt.grid(ls = ":",color = "gray",alpha = 0.5)
plt.text(90, -5, 'R$^2$=0.46, RMSE/Mean=0.19', fontdict={'size': 12, 'color': 'black'})
plt.legend()

plt.savefig('../result/aus-XGBoost.jpg', dpi = 300)
plt.show()
