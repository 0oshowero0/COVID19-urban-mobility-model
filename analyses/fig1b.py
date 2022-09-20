import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import floor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
from adjustText import adjust_text
from scipy.stats import pearsonr,spearmanr,linregress
from scipy.stats import t
import statsmodels.api as sm
from seaborn_regression_util import *
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] =  20


def init_city_list():
    NewYorkCity = {'first': 51, 'second': 70-6+7,'cases_name':'nyc.npy','14day':188545}
    Cook = { 'first': 48, 'second': 70-6+7,'cases_name':'cook.npy','14day':58457}
    Nassau = {'first': 48, 'second':70-6+7,'cases_name':'nassau.npy','14day':38743}
    Suffolk = {'first': 53, 'second': 70-6+7,'cases_name':'suffolk.npy','14day':37544}
    Los_Angeles = { 'first': 55, 'second': 70-6+7,'cases_name':'los_angeles.npy','14day':35392}
    Bergen = {'first': 50, 'second': 70-6+7,'cases_name':'bergen.npy','14day':17080}
    Hudson = {'first': 60, 'second': 70-6+7,'cases_name':'hudson.npy','14day':17602}
    Philadelphia = {'first': 54, 'second': 70-6+7,'cases_name':'philadelphia.npy','14day':19093}
    Middlesex = {'first': 56, 'second': 70-6+7,'cases_name':'middlesex.npy','14day':18890}
    Union = {'first': 57, 'second': 70-6+7,'cases_name':'union.npy','14day':14385}
    Miami_Dade = { 'first': 54, 'second': 70-6+7,'cases_name':'miami_dade.npy','14day':14742}
    New_Haven = { 'first': 60, 'second': 70-6+7,'cases_name':'new_haven.npy','14day':9712}
    Oakland = {'first': 57, 'second': 70-6+7,'cases_name':'oakland.npy','14day':9330}
    New_Orleans = {'first': 54, 'second': 70-6+7,'cases_name':'new_orleans.npy','14day':6768}
    Ocean = {'first': 60, 'second': 70-6+7,'cases_name':'ocean.npy','14day':7742}
    Harris = {'first': 60, 'second': 70-6+14,'cases_name':'harris.npy','14day':8817}
    King = {'first': 48, 'second': 70-6+7,'cases_name':'king.npy','14day':7290}
    Davidson = {'first': 53, 'second': 70-6+7,'cases_name':'davidson.npy','14day':3745}
    Will = { 'first': 58, 'second': 70-6+14,'cases_name':'will.npy','14day':4090}
    Hillsborough = {'first': 57, 'second': 70-6+14,'cases_name':'hillsborough.npy','14day':1530}

    city_list = {'Bergen':Bergen,'Middlesex':Middlesex,'Hudson':Hudson,'Philadelphia':Philadelphia,'NY':NewYorkCity,\
                 'Nassau':Nassau,'Cook':Cook, 'Union':Union,'Davidson':Davidson,'Will':Will,'MiamiDade':Miami_Dade,'Hillsborough':Hillsborough, \
                 'Suffolk':Suffolk,'Los_Angeles':Los_Angeles,'New_Haven':New_Haven,'Oakland':Oakland,'New_Orleans':New_Orleans,'Ocean':Ocean, \
                 'Harris':Harris,'King':King}

    return city_list

def load_cases_and_fitted_all(topk=1):
    city_list = init_city_list()
    cases = []
    fitted = []
    name_list = []
    for key,values in city_list.items():
        name_list.append(key)
        cases.append(values['14day'])
        fit_input_dir = Path('./results/fig1b_14days_pred')
        npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves/'+'*curve*.npy')])
        fitted_data = [np.load(i).reshape(-1) for i in npy_datas][0]
        fitted.append(fitted_data[-1])

    return cases, fitted, name_list



def draw():
    cases_data, fitted_data, city_name_list = load_cases_and_fitted_all(topk=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    cases = []
    fitted = []
    name = []
    for i in range(len(city_name_list)):
        cases.append(cases_data[i])
        fitted.append(fitted_data[i])
        if city_name_list[i] == 'NY':
            name.append('New York City')
        elif city_name_list[i] == 'New_Orleans':
            name.append('New Orleans')
        elif city_name_list[i] == 'New_Haven':
            name.append('New Haven')
        elif city_name_list[i] == 'Los_Angeles':
            name.append('Los Angeles')
        else:
            name.append(city_name_list[i])

    cases = np.array(cases)
    fitted = np.array(fitted)



    cases_log = np.log10(cases)
    fitted_log = np.log10(fitted)
    sns.regplot(cases_log,fitted_log,scatter_kws={"s": 30,'color':'k'},line_kws={'color':'#4c72b0'},ci=95)

    plt.xlim(3,5.5)
    plt.ylim(3,5.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    nrmse = np.sqrt(mean_squared_error(fitted,cases)) / cases.mean()
    plt.text((xlim[1]-xlim[0]) / 10 + xlim[0]  , (ylim[1]-ylim[0]) / 8 *6.6 + ylim[0],'NRMSE of Ours: '+"%(nrmse).3f"%{'nrmse':nrmse},fontdict = {'size': 13, 'color': 'blue'})
    nrmse1 = 7.222
    plt.text((xlim[1]-xlim[0]) / 10 + xlim[0]  , (ylim[1]-ylim[0]) / 8 *6 + ylim[0],'NRMSE of SEIR: '+"%(nrmse).3f"%{'nrmse':nrmse1},fontdict = {'size': 13, 'color': 'green'})
    plt.grid(which='major',ls='--', alpha=0.8)


    ori_y_ticks = ax.get_yticks()
    new_y_ticks = ["$10^{%(i).1f}$"%{'i':i} for i in ori_y_ticks]
    plt.yticks(ori_y_ticks,new_y_ticks)
    ori_x_ticks = ax.get_xticks()
    new_x_ticks = ["$10^{%(i).1f}$"%{'i':i} for i in ori_x_ticks]
    plt.xticks(ori_x_ticks,new_x_ticks)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Number of Reported Cases',size=20)
    plt.ylabel('Number of Predicted Cases',size=20)


    def label_point(x, y, val, ax):
        tmp = []
        for i in range(len(val)):
            if i %2 == 1:
                tmp.append(ax.text(x[i]-0.02, y[i], str(val[i]),ha='center', va='center',fontdict = {'size': 12, 'color': 'black'}))
            else:
                tmp.append(ax.text(x[i]-0.02, y[i], str(val[i]),ha='center', va='center',fontdict = {'size': 12, 'color': 'black'}))
        return tmp

    texts = label_point(cases_log, fitted_log, name, plt.gca())
    adjust_text(texts,only_move={'text': 'y'},arrowprops=dict(arrowstyle='-', 
                            color='gray',
                            lw=1))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw()




