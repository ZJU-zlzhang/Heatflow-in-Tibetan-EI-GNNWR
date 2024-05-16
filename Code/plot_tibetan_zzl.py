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

# 打开栅格数据集
dataset = gdal.Open('../result/result_tibet_4.tif')

# 获取栅格的大小
x_size = dataset.RasterXSize
y_size = dataset.RasterYSize
adfGeoTransform = dataset.GetGeoTransform()
values = dataset.ReadAsArray()

# 定义边界掩码（例如，假设边界定义为非零像素）
boundary_mask = values != 0  # 根据实际情况调整边界定义
#
# 定义值修改掩码：在边界内部且值低于0或高于120的像素
value_modify_mask = boundary_mask & ((values < 0) | (values > 120))

# 应用修改：只修改边界内的像素
values[value_modify_mask] = np.clip(values[value_modify_mask], 0, 120)
values[~boundary_mask] = np.nan

# # 设置边界外的值为-1
# values[~boundary_mask] = -20
# # 仅调整边界内的值
# values[boundary_mask] = np.clip(values[boundary_mask], -20, 120)


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
x = []
for i in range(x_size):
    x.append(adfGeoTransform[0] + i * adfGeoTransform[1])
y = []
for i in range(y_size):
    y.append(adfGeoTransform[3] + i * adfGeoTransform[5])

crs = ccrs.PlateCarree()

fig, ax1 = plt.subplots(figsize=(14, 6), dpi=130)

aus_shp = gpd.read_file('../shp/tibet_new.shp')

hf_data = pd.read_csv('../lib/Tibet_boundary_point.csv', sep=",")


# cm = plt.cm.get_cmap('RdYlBu_r')

# vmin和vmax定义了色带的最小和最大值
vmin = 0  # 色带的最小值改为0
vmax = 120  # 色带的最大值改为120

aus_shp.plot(fc="none", ec="black", ax=ax1, lw=.8, zorder=2)

# 截取部分色带
cmap = mpl.cm.get_cmap('RdYlBu_r')
# cmap = mpl.cm.get_cmap('GnBu_r')
newcolors = cmap(np.linspace(0, 1, 256))
newcolor = []
for i in range(256):
    if i < 60:
        newcolor.append(newcolors[i])
    elif i > 60 and i % 3 == 0:
        newcolor.append(newcolors[i])

newcmap = ListedColormap(newcolor[:])

# ax1.set_extent([112.2, 155, -42, -10])
# ax1.set(xlim=(73.9, 105.1), ylim=(26.5, 40.5))
# 在调用contourf时，使用新的vmin和vmax作为等高线的级别
# c = ax1.contourf(x, y, values, cmap=newcmap, levels=np.linspace(vmin, vmax, num=121), vmin=vmin, vmax=vmax, projection=crs)
# c = ax1.contourf(x, y, values, cmap=newcmap, levels=np.arange(0, 120, 0.1), projection=crs)
# 确保在使用新的色带时不包括NaN值的区域
# c = ax1.contourf(x, y, values, cmap=newcmap, levels=np.linspace(vmin, vmax, num=121), vmin=vmin, vmax=vmax, extend='neither', projection=crs)
c = ax1.contourf(x, y, values, cmap=newcmap, levels=np.linspace(vmin, vmax, num=121), vmin=vmin, vmax=vmax, extend='neither', projection=crs)
cax = fig.add_axes([ax1.get_position().x1 + 0, ax1.get_position().y0, 0.01, ax1.get_position().height])
# gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='k', alpha=0.5, linestyle='--')

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_xlabel('Longitude(°E)', fontsize=14)
ax1.set_ylabel('Latitude(°N)', fontsize=14)
cbar = plt.colorbar(c, cax=cax)
cbar.set_label('mW/m²', fontsize=12)

# scatter = ax1.scatter(hf_data['Lon'], hf_data['Lat'], c=hf_data['HF'], s=10, ec="k", lw=.5, cmap=cm, vmin=vmin,
#                      vmax=vmax)

# scatter_bar = plt.colorbar(scatter, shrink=0.75, label="mW/m^2")
# scatter_bar.outline.set_edgecolor('none')
# plt.savefig('../result/aus-gnnwr.jpg', dpi = 300)

# aus_shp = aus_shp[(aus_shp.geometry.type == 'Polygon') | (aus_shp.geometry.type == 'MultiPolygon')]
# # 转换为matplotlib路径
# boundary_path = Path.make_compound_path(*[Path(np.array(geom.xy).T) for geom in aus_shp.geometry])
#
# # 创建一个反向的PathPatch
# patch = PathPatch(boundary_path, transform=ax1.transData, facecolor='none', edgecolor='none', lw=0, zorder=2, invert=True)
#
# # 将patch添加到ax中
# ax1.add_patch(patch)
#
# # 设置patch的背景色为白色
# ax1.set_clip_path(patch)
# ax1.set_facecolor('white')
# 保存SVG文件
plt.savefig('../result/tibetan.svg', format='svg', dpi=300)
plt.show()
