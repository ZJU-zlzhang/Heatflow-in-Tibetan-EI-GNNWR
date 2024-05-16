Attempt = '11th'
Run = 'test_1'
gridsearch = 'yes'
Res = '05'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from matplotlib import pyplot as plt
from science.extraTree_model_base_coslon import cal_statistic

Data = pd.read_csv('../lib/gnnwr_result_tibet.csv', sep=',')
Data_al = pd.read_csv('../lib/gnnwr_result.csv', sep=',')
actual = np.array(Data['hf'], dtype=np.float64).reshape((-1, 1))
pred = np.array(Data['hf_pred'], dtype=np.float64).reshape((-1, 1))

actual2 = np.array(Data_al['hf'], dtype=np.float64).reshape((-1, 1))
pred2 = np.array(Data_al['hf_pred'], dtype=np.float64).reshape((-1, 1))

fig, ax = plt.subplots()

scatter1 = ax.scatter(actual, pred, c='blue', edgecolor='black')
scatter2 = ax.scatter(actual2, pred2, c='green', edgecolor='black', marker='s')

ax.set(xlim=(-10, 180), xticks=np.arange(0, 180, 25),
       ylim=(-10, 180), yticks=np.arange(0, 180, 25))

plt.xlabel('Actual [mW/m$^2$]')
plt.ylabel('Predicted [mW/m$^2$]')
plt.plot([0, 175], [0, 175], linestyle='dashed', c='black')
plt.text(90, -5, 'R$^2$=0.91, RMSE/Mean=0.07', fontdict={'size': 12, 'color': 'darkblue'})
plt.text(90, 6, 'R$^2$=0.73, RMSE/Mean=0.19', fontdict={'size': 12, 'color': 'darkgreen'})
# plt.savefig('01thAttempt/heatflow/plotTest.jpg', dpi = 300)

plt.legend([scatter2, scatter1], ['Only Tibet measurements', 'Tibet with ground microplates measurements'],
           loc="upper left", framealpha=0)
plt.show()
