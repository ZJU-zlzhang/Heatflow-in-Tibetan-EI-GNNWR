from osgeo import gdal
import matplotlib as mpl

mpl.use('TkAgg')

import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import matplotlib.colors as cor
import matplotlib.patches as mpatches
import cartopy.io.shapereader as sr
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
from matplotlib.colors import ListedColormap

values = gdal.Open('../result/beta_result_tibet/Transform.tif')
x_ = values.RasterXSize
y_ = values.RasterYSize
adfGeoTransform = values.GetGeoTransform()
values = values.ReadAsArray()

x = []
for i in range(x_):
    x.append(adfGeoTransform[0] + i * adfGeoTransform[1])
y = []
for i in range(y_):
    y.append(adfGeoTransform[3] + i * adfGeoTransform[5])

crs = ccrs.PlateCarree()

fig, ax1 = plt.subplots(figsize=(8, 6), dpi=130)

aus_shp = gpd.read_file('../shp/tibet_new.shp')
hf_data = pd.read_csv('../lib/Combine_AUS.csv', sep=",")

cm = plt.cm.get_cmap('Spectral_r')

vmin = hf_data["HF"].min()
vmax = hf_data["HF"].max()

# 去掉边框(坐标轴)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# 去掉边框上的数字
plt.xticks([])
plt.yticks([])

aus_shp.plot(fc="none", ec="black", ax=ax1, lw=.5, zorder=2)

# 截取部分色带
cmap = mpl.cm.get_cmap('Spectral_r')
newcolors = cmap(np.linspace(0, 1, 256))
newcolor = []
for i in range(256):
    if i < 60:
        newcolor.append(newcolors[i])
    elif i > 60 and i % 3 == 0:
        newcolor.append(newcolors[i])

newcmap = ListedColormap(newcolor[:])

# ax1.set_extent([112.2, 155, -42, -10])
c = ax1.contourf(x, y, values, cmap=newcmap, levels=np.arange(0, 1.05, 0.1), projection=crs)
cax = fig.add_axes([ax1.get_position().x1 + 0.03, ax1.get_position().y0, 0.02, ax1.get_position().height])
# gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='k', alpha=0.5, linestyle='--')

plt.colorbar(c, cax=cax)

# scatter = ax1.scatter(hf_data['Lon'], hf_data['Lat'], c=hf_data['HF'], s=10, ec="k", lw=.5, cmap=cm, vmin=vmin,
#                      vmax=vmax)

# scatter_bar = plt.colorbar(scatter, shrink=0.75, label="mW/m^2")
# scatter_bar.outline.set_edgecolor('none')


plt.show()
