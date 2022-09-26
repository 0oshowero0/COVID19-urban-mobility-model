import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from math import floor
import seaborn as sns
from scipy.stats import pearsonr,spearmanr
from seaborn_regression_util import *
from math import ceil,floor
import geopandas as gpd
from shapely.geometry import Polygon,Point,MultiPolygon

plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] =  20

def load_boundary(fp='./fig2b_data/coronavirus-data-master/Geography-resources/MODZCTA_2010_WGS1984.geo.json'):
    geo_boundary_Df = gpd.read_file(fp, driver='geojson')
    geo_boundary_Df['MODZCTA'] = geo_boundary_Df['MODZCTA'].astype(int)
    return geo_boundary_Df


def load_pop(data_dir):
    pop = np.load(data_dir).reshape(-1, 1).astype('float32')
    std = pop[pop>100].std()
    mean = pop[pop>100].mean()
    upper_bond = mean + 3*std
    pop = np.where(pop>upper_bond, upper_bond, pop)
    return pop

def gen_grid_array_NY(long_size=44, lati_size=44):
    long1 = -74.046699
    long2 = -73.703826
    lati1 = 40.909588
    lati2 = 40.541946


    long_gap = np.abs(long2 - long1) / long_size
    lati_gap = np.abs(lati2 - lati1) / lati_size

    grid_array = np.zeros(lati_size * long_size)

    return grid_array, lati_gap, long_gap

def gen_grid_array_NY(long_size=44, lati_size=44):
    long1 = -74.046699
    long2 = -73.703826
    lati1 = 40.909588
    lati2 = 40.541946

    long_gap = np.abs(long2 - long1) / long_size
    lati_gap = np.abs(lati2 - lati1) / lati_size

    grid_array = []
    for i in range(44):
        for j in range(44):
            pt = Point([long1+long_gap/2 + j * long_gap, lati1 - lati_gap / 2 - lati_gap * i])
            grid_array.append(pt)

    return grid_array



def read_fitted_nyc_inf(fp):
    inf_rate = np.load(fp)
    return inf_rate[:,-1]


def match_data(fit_loc, real_loc, old= False):
    fitted_inf_count = read_fitted_nyc_inf(fp=fit_loc)
    nyc_pop = load_pop('./population/nyc_pop.npy').reshape(-1)
    long_min = -74.046699
    long_max = -73.703826
    lati_max = 40.909588
    lati_min = 40.541946
    grid_array = gen_grid_array_NY(long_size=44, lati_size=44)
    real_inf_count = pd.read_csv(real_loc, sep=",")
    if not old:
        real_inf_count = real_inf_count.iloc[:,[0,4,5,6,8,9,10,11]]
    else:
        real_inf_count = real_inf_count.iloc[:,[0,3,4]]

    geo_boundary = load_boundary()
    x = geo_boundary.merge(real_inf_count,left_on = ['MODZCTA'],right_on = ['MODIFIED_ZCTA'],how='inner')

    real_array = []
    fit_array = []

    for _, row in x.iterrows():
        real_array.append(float(row['COVID_CASE_COUNT']))
        plg = row['geometry']
        fit_count = 0
        for i in range(len(grid_array)):
            if plg.contains(grid_array[i]):
                fit_count += fitted_inf_count[i]
        fit_array.append(fit_count)

    return np.array(fit_array), np.array(real_array)




i = 20
fit_array, real_array = match_data(fit_loc='./results/fig2b_NYC/out_curves/result_cumu_' + str(i).zfill(2) +'.npy', 
                                   real_loc='./fig2b_data/coronavirus-data-master/totals/data-by-modzcta.csv',old=True)

real_rank_idx = np.argsort(real_array)
real_rank_idx = np.argsort(real_rank_idx)
fit_rank_idx = np.argsort(fit_array)
fit_rank_idx = np.argsort(fit_rank_idx)
spearman_R,_ = spearmanr(fit_array,real_array)

data = {'Rank of Real-world Infection':real_rank_idx,'Rank of Fitted Infection':fit_rank_idx}
data = pd.DataFrame(data)


fig, ax = plt.subplots(figsize=(10, 6))

sns.kdeplot(x=real_rank_idx, y=fit_rank_idx, levels=10, color="k", linewidths=1,clip=(0,180),cbar=False,fill=True,cmap="rocket",bw=0.8)
sns.regplot(data=data,x='Rank of Real-world Infection',y='Rank of Fitted Infection',scatter_kws={"s": 30,'color':'#00635E'}, line_kws={'color':'b','linewidth':3},ci=None)
ax.set_facecolor('#1f0c2b')


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Rank of Real-world Infection',size=20)
plt.ylabel('Rank of Predicted Infection',size=20)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

plt.text(xlim[0]- (xlim[1]-xlim[0]) / 10*0    , (ylim[1]-ylim[0]) / 26 *1 + ylim[1],'Spearman\'s R: '+"%(r2).3f"%{'r2':spearman_R },fontdict = {'size': 15, 'color': 'black'})


plt.tight_layout()
plt.show()