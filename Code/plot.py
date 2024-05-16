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
Data_al = pd.read_csv('../lib/gnnwr_result_tibet_ground.csv', sep=',')
actual = np.array(Data['hf'], dtype=np.float64).reshape((-1, 1))
pred = np.array(Data['hf_pred'], dtype=np.float64).reshape((-1, 1))

actual2 = np.array(Data_al['hf'], dtype=np.float64).reshape((-1, 1))
pred2 = np.array(Data_al['hf_pred'], dtype=np.float64).reshape((-1, 1))

actual3 = []
pred3 = []
for i in range(len(actual2)):
    if actual2[i] not in actual:
        actual3.append(actual2[i])
        pred3.append(pred2[i])

fig, ax = plt.subplots()

a = cal_statistic(np.array(actual3), np.array(pred3))
RMSE = a[1] / actual.mean()

scatter1 = ax.scatter(actual3, pred3, c='blue', edgecolor='black', marker='s')
scatter2 = ax.scatter(actual, pred, c='green', edgecolor='black')

ax.set(xlim=(-10, 180), xticks=np.arange(0, 180, 25),
       ylim=(-10, 180), yticks=np.arange(0, 180, 25))

plt.xlabel('Actual [mW/m$^2$]')
plt.ylabel('Predicted [mW/m$^2$]')
plt.plot([0, 175], [0, 175], linestyle='dashed', c='black')
plt.text(90, -5, 'R$^2$=0.91, RMSE/Mean=0.07', fontdict={'size': 12, 'color': 'black'})
# plt.text(90, 6, 'R$^2$=0.73, RMSE/Mean=0.1', fontdict={'size': 12, 'color': 'darkgreen'})
# plt.savefig('01thAttempt/heatflow/plotTest.jpg', dpi = 300)
# plt.savefig('allTibet_R2_0.87.jpg', dpi=300)

plt.legend([scatter1, scatter2], ['tibet_inner', 'tibet_ground'],
           loc="upper left", framealpha=0)
plt.show()
