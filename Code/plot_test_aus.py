from osgeo import gdal
import matplotlib as mpl
import cmaps

mpl.use('TkAgg')

import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import matplotlib.colors as cor
import matplotlib.patches as mpatches
# import cartopy.io.shapereader as sr
# import cartopy.feature as cfeature
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
from matplotlib.colors import ListedColormap


# -----------函数：添加比例尺--------------
def add_scalebar(ax, lon0, lat0, length, size=0.45):
    '''
    ax: 坐标轴
    lon0: 经度
    lat0: 纬度
    length: 长度
    size: 控制粗细和距离的
    '''
    # style 3
    ax.hlines(y=lat0, xmin=lon0, xmax=lon0 + length / 111, colors="black", ls="-", lw=1, label='%d km' % (length))
    ax.vlines(x=lon0, ymin=lat0 - size, ymax=lat0 + size, colors="black", ls="-", lw=1)
    ax.vlines(x=lon0 + length / 2 / 111, ymin=lat0 - size, ymax=lat0 + size, colors="black", ls="-", lw=1)
    ax.vlines(x=lon0 + length / 111, ymin=lat0 - size, ymax=lat0 + size, colors="black", ls="-", lw=1)
    ax.text(lon0 + length / 111, lat0 + size + 0.05, '%d' % (length), horizontalalignment='center')
    ax.text(lon0 + length / 2 / 111, lat0 + size + 0.05, '%d' % (length / 2), horizontalalignment='center')
    ax.text(lon0, lat0 + size + 0.05, '0', horizontalalignment='center')
    ax.text(lon0 + length / 111 / 2 * 3, lat0 + size + 0.05, 'km', horizontalalignment='center')


def add_north(ax, labelsize=18, loc_x=0.12, loc_y=0.95, width=0.06, height=0.09, pad=0.14):
    """
    画一个比例尺带'N'文字注释
    主要参数如下
    :param ax: 要画的坐标区域 Axes实例 plt.gca()获取即可
    :param labelsize: 显示'N'文字的大小
    :param loc_x: 以文字下部为中心的占整个ax横向比例
    :param loc_y: 以文字下部为中心的占整个ax纵向比例
    :param width: 指南针占ax比例宽度
    :param height: 指南针占ax比例高度
    :param pad: 文字符号占ax比例间隙
    :return: None
    """
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen * (loc_x - width * .5), miny + ylen * (loc_y - pad)]
    right = [minx + xlen * (loc_x + width * .5), miny + ylen * (loc_y - pad)]
    top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
    center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * .4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.text(s='N',
            x=minx + xlen * loc_x,
            y=miny + ylen * (loc_y - pad + height),
            fontsize=labelsize,
            horizontalalignment='center',
            verticalalignment='bottom')
    ax.add_patch(triangle)


values = gdal.Open('../result/xgb_aus_3.tif')
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

aus_shp = gpd.read_file('../shp/ne_110m_admin_0_countries.shp')
hf_data = pd.read_csv('../lib/Combine_AUS.csv', sep=",")

# cm = plt.cm.get_cmap('Spectral_r')

vmin = hf_data["HF"].min()
vmax = hf_data["HF"].max()

aus_shp.plot(fc="none", ec="black", ax=ax1, lw=.5, zorder=2)

# 截取部分色带
# cmap = cmaps.Spectral_r RdYlBu_r
cm = plt.cm.get_cmap('RdYlBu_r')
# newcolors = cmap(np.linspace(0, 1, 256))
# newcolor = []
# for i in range(256):
#     if i < 60:
#         newcolor.append(newcolors[i])
#     elif i > 60 and i % 3 == 0:
#         newcolor.append(newcolors[i])
#
# newcmap = ListedColormap(newcolor[:])

# ax1.set_extent([112.2, 155, -42, -10])
c = ax1.contourf(x, y, values, cmap=cm, levels=np.arange(20, 111, 5), projection=crs, alpha=0.9)
cax = fig.add_axes([ax1.get_position().x1 + 0.03, ax1.get_position().y0, 0.02, ax1.get_position().height])
# gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='k', alpha=0.5, linestyle='--')
# add_north(ax1)
# add_scalebar(ax1, 112, -42, 1000, size=0.2)

ax1.set_xlabel('Longitude(°E)')
ax1.set_ylabel('Latitude(°S)')
cbar = plt.colorbar(c, cax=cax)
cbar.set_label('mW/m²')

scatter = ax1.scatter(hf_data['Lon'], hf_data['Lat'], c=hf_data['HF'], s=45, alpha=1, ec="k", lw=.5, cmap=cm, vmin=20,
                      vmax=110)
# plt.colorbar(scatter, cax=cax)

# scatter_bar = plt.colorbar(scatter, shrink=0.75, label="mW/m^2")
# scatter_bar.outline.set_edgecolor('none')


plt.show()
