# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 13:29:34 2021

@author: DELL
"""

Attempt='11th'
Run='test_1'
gridsearch='yes'
Res='05'
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import _regression
import xgboost as xgb
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

Data_o = pd.read_csv('C:/Users/DELL/Desktop/Combined_%s.txt' % (Res), sep=' ')
Data_o.loc[(Data_o['Lon']==180) & (Data_o['Lat']==90), 'HF'] = np.nan
Data_o.loc[(Data_o['Lon']==-180) & (Data_o['Lat']==90), 'HF'] = np.nan
Data_o= Data_o.round({'Quality': 0, 'Tectonics': 0, 'Geology': 0})

Data_o = Data_o.dropna()
Data_o = Data_o.drop([11984,12244],axis=0)
Data = Data_o[['HF','Lon','Lat', 'IsoCorrAnomaly', 'MeanCurv', 'SI_TopoIso',
               'LAB', 'LAB_LitMod', 'LAB_LitMod_Aus17', 'LAB`_AN1', 'Moho', 'Moho_AN1',
               'Moho_AN1_Aus17_Afr', 'Moho_LitMod_Aus17_Afr', 'Moho_Lloyd_Aus17_Afr',
               'Moho_AN1-Shen_Aus17_Afr', 'Topo', 'Tectonics', 'Ridge', 'Transform', 'Trench',
               'YoungRift', 'Volcanos', 'MagAnomaly', 'Sus', 'Sus_AN1', 'Sus_AN1_Aus17_Afr',
               'Sus_LitMod_Aus17_Afr', 'Sus_Lloyd_Aus17_Afr', 'Sus_AN1-Shen_Aus17_Afr',
               'Bz5', 'IceTopography', 'Curie']]

Data_Tib = Data.loc[(Data['Lat']<=40) & (Data['Lon']>=73) & (Data['Lat']>=26) & (Data['Lon']<=105)]
Data_Except_Tib = Data.loc[((Data['Lat']<26) | (Data['Lat']>40)) | ((Data['Lon']<73) | (Data['Lon']>105))]

Features = ['Tectonics', 'Moho_LitMod_Aus17_Afr', 'Trench', 'LAB_LitMod', 'Ridge',
            'Transform','MeanCurv','YoungRift','Sus_LitMod_Aus17_Afr', 'Bz5',
            'Topo','Volcanos']
#全球的数据
Data_global=Data.loc[(Data['Lat']<=100)]
X_Global = pd.DataFrame(Data_global, columns=Features)
y2=[]
for j in range(len(Data)):
    y2.append(Data.iloc[j,0])

X_Except_Tib=pd.DataFrame(Data_Except_Tib, columns=Features)
X_Tib=pd.DataFrame(Data_Tib, columns=Features)


y = []
y1 = []
for j in range(len(Data_Tib)):
    y.append(Data_Tib.iloc[j, 0])
for j in range(len(Data_Except_Tib)):
    y1.append(Data_Except_Tib.iloc[j,0])
Mean = sum(y)/len(y)
print('Mean Heat Flow:', Mean)
print('Choice of HF measurements:', len(X_Tib))

x_train, x_test, y_train, y_test = train_test_split(X_Except_Tib,y1,test_size=0.2,random_state=42,shuffle=True)


s=[]
for i in range(50,501,50):    
    #params = {'objective':'reg:squarederror','learning_rate':0.01,'max_depth':11,
    #      'n_estimators':i,'subsample':0.7, 'gamma': 0}
    model = xgb.XGBRegressor(n_estimators=500,max_depth=5,random_state=42,subsample=0.8,eta=0.01)
    b=cross_val_score(model,X_Except_Tib,y1,cv=5)
    print(b)
    score=b.mean()
#    score=model.score(x_test,y_test)
    s.append(score)
    
plt.plot(np.arange(50,501,50),s,color='red',label='xgboost')
plt.legend()
plt.show()

model = xgb.XGBRegressor(n_estimators=150,max_depth=10,random_state=42,subsample=0.8,eta=0.1)
model.fit(x_train,y_train)
print(model.score(x_test,y_test),model.score(x_train,y_train))

names = ['Tectonics', 'Moho', 'Trench', 'LAB', 'Ridge','Transform','Mean Curvature',
         'Young Rift','Susceptibility', 'Bz', 'Topography', 'Volcanoes']

name = []
feature_importance = model.feature_importances_
#feature_importance = 100*(feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

fig = plt.figure(figsize=(9, 7))
plt.barh(pos, feature_importance[sorted_idx], align='center',color='silver')
for i in range(len(sorted_idx)):
    name.append(names[sorted_idx[i]])
plt.yticks(pos, name[:],fontsize=20)
plt.xlabel('Importance',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Variable Importance',fontsize=20)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
#fig.savefig('%sAttempt/Plots/Importance_Deviance_%s.jpg' % (Attempt,Run), dpi=300)
plt.show()

def plotPredictedTest(x,y,Attempt,Run,value=None,d=None,e=None):
    fig, ax = plt.subplots(figsize=(9,7))
    if not value is None:
        cmap = plt.get_cmap('gist_earth')
        plt.scatter(x,y,c=value,cmap=cmap,edgecolors=(0,0,0),vmin=d,vmax=e)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=15)
        cb.set_label('[m]',fontsize=15, rotation=270)
    else:
        plt.scatter(x,y, edgecolors=(0,0,0), color='darkblue')
    plt.plot([min(x),max(x)], [min(x),max(x)], 'k--', lw=2)
    plt.xlabel('Actual'+ '\n' + '[mW/m$^2$]',fontsize=20)
    plt.ylabel('Predicted'+ '\n' + '[mW/m$^2$]',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title('Ground Truth vs Predicted',fontsize=20)
    plt.tight_layout()
    #fig.savefig('%sAttempt/Plots/Test-Pedicted_%s.jpg' % (Attempt,Run), dpi=300)
    plt.show()

y_pred=model.predict(x_test)

e=abs(np.matrix(y_test)-np.matrix(y_pred)).A1
ee=np.dot(e,e)
RMSE=np.sqrt(ee/len(y_test))

plotPredictedTest(y_test,y_pred,Attempt,Run)


ToPredict = pd.read_csv('C:/Users/DELL/Desktop/Grid_%s.txt' % (Res), sep=' ')
ToPredict=ToPredict.round({'Tectonics': 0})
TP = ToPredict[['Lon','Lat', 'IsoCorrAnomaly', 'MeanCurv', 'SI_TopoIso', 
               'LAB', 'LAB_LitMod', 'LAB_LitMod_Aus17', 'LAB_AN1', 'Moho', 'Moho_AN1',
               'Moho_AN1_Aus17_Afr', 'Moho_LitMod_Aus17_Afr', 'Moho_Lloyd_Aus17_Afr',
               'Moho_AN1-Shen_Aus17_Afr', 'Topo', 'Tectonics', 'Ridge', 'Transform', 'Trench',
               'YoungRift', 'Volcanos', 'MagAnomaly', 'Sus', 'Sus_AN1', 'Sus_AN1_Aus17_Afr',
               'Sus_LitMod_Aus17_Afr', 'Sus_Lloyd_Aus17_Afr', 'Sus_AN1-Shen_Aus17_Afr',
               'Bz5', 'IceTopography', 'Curie']] 
TP.dropna()
Data_Tibet=TP.loc[(TP['Lon']<=105)&(TP['Lon']>=74)&(TP['Lat']>=26)&(TP['Lat']<=40)]
X_new=pd.DataFrame(Data_Tibet, columns=Features)
Lon_p=[]
Lat_p=[]
for i in range(len(Data_Tibet)):
    Lon_p.append(Data_Tibet.iloc[i,0])
    Lat_p.append(Data_Tibet.iloc[i,1])
new_pred = model.predict(X_new)

#在底图上标记上原有的实测点
Lon_Tibet=[]
Lat_Tibet=[]
for j in range(len(Data_Tib)):
    Lon_Tibet.append(Data_Tib.iloc[j,1])
    Lat_Tibet.append(Data_Tib.iloc[j,2])

def plottingTibet(Lon, Lat, value, unit, name, a, b, Attempt, Run):
    fig = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(111, projection=ccrs.PlateCarree(central_longitude=65))
    ax1.set_global()  # 使得轴域（Axes 即两条坐标轴围城的区域）适应地图的大小
    ax1.coastlines()  # 画出海岸线
    with open(r'C:/Users/DELL/Desktop/CN-border-L1.dat') as src:
        context = ''.join([line for line in src if not line.startswith('#')])
        blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
        borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]
    for line in borders:
        ax1.plot(line[0::2], line[1::2], '-', lw=1, color='k', transform=ccrs.Geodetic())
    # 标注坐标轴
    extent = [73, 105, 26, 40]
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    ax1.set_xticks([75, 81, 87, 93, 99, 105], crs=ccrs.PlateCarree())  # 设置显示的经度
    ax1.set_yticks([26, 28, 30, 32, 34, 36, 38, 40], crs=ccrs.PlateCarree())  # 设置显示的纬度
    # zero_direction_label 用来设置经度的0度加不加E和W
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
#     ax1.set_title("Heat Flow in Tibetan", loc="left", fontsize=10)
    plt.scatter(Lon, Lat, c=value, cmap='jet', marker='s', s=50, vmin=a, vmax=b)
    cb = plt.colorbar(fraction=0.021, pad=0.04)
    cb.set_label('%s' % unit, labelpad=30, fontsize=15, rotation=270)
    plt.title('%s' % name, fontsize=15)
    cb.ax.tick_params(labelsize=15)
    plt.show()
def plotting(Lon,Lat,value,unit,name,a,b,Attempt,Run):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    plt.scatter(Lon,Lat,c=value,cmap='jet', marker='s', s=100, vmin=a, vmax=b,zorder=1)
    cb = plt.colorbar(fraction=0.021, pad=0.04)
    cb.set_label('%s' % unit, labelpad=30, fontsize=15, rotation=270) 
    plt.title('%s' % name, fontsize=15)
    cb.ax.tick_params(labelsize=15)
    plt.scatter(Lon_Tibet , Lat_Tibet ,facecolors='none',s=5,zorder=10,edgecolors='k')
    plt.show()

##保存预测结果
M=np.vstack((Lon_p,Lat_p, new_pred)).T
np.savetxt('C:/Users/DELL/Desktop/result.txt', M, fmt='%.3f')

plottingTibet(Lon_p,Lat_p,new_pred,'[mW/m$^2$]','Predicted_Heat_Flux_Tibetan',0,120,Attempt,Run)
plotting(Lon_p,Lat_p,new_pred,'[mW/m$^2$]','Predicted_Heat_Flux',0,120,Attempt,Run)

