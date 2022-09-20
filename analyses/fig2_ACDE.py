from matplotlib.pyplot import draw, grid
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
import json
import pandas as pd
from math import floor, inf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
import seaborn as sns
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import LinearRegression
from adjustText import adjust_text
from seaborn_regression_util import *
from scipy.optimize import curve_fit


def init_city_list():
    NewYorkCity = {'first': 51, 'second': 70-6+7,'cases_name':'nyc.npy'}
    Cook = { 'first': 48, 'second': 70-6+7,'cases_name':'cook.npy'}
    Nassau = {'first': 48, 'second':70-6+7,'cases_name':'nassau.npy'}
    Suffolk = {'first': 53, 'second': 70-6+7,'cases_name':'suffolk.npy'}
    Los_Angeles = { 'first': 55, 'second': 70-6+7,'cases_name':'los_angeles.npy'}
    Bergen = {'first': 50, 'second': 70-6+7,'cases_name':'bergen.npy'}
    Hudson = {'first': 60, 'second': 70-6+7,'cases_name':'hudson.npy'}
    Philadelphia = {'first': 54, 'second': 70-6+7,'cases_name':'philadelphia.npy'}
    Middlesex = {'first': 56, 'second': 70-6+7,'cases_name':'middlesex.npy'}
    Union = {'first': 57, 'second': 70-6+7,'cases_name':'union.npy'}
    Miami_Dade = { 'first': 54, 'second': 70-6+7,'cases_name':'miami_dade.npy'}
    New_Haven = { 'first': 60, 'second': 70-6+7,'cases_name':'new_haven.npy'}
    Oakland = {'first': 57, 'second': 70-6+7,'cases_name':'oakland.npy'}
    New_Orleans = {'first': 54, 'second': 70-6+7,'cases_name':'new_orleans.npy'}
    Ocean = {'first': 60, 'second': 70-6+7,'cases_name':'ocean.npy'}
    Harris = {'first': 60, 'second': 70-6+14,'cases_name':'harris.npy'}
    King = {'first': 48, 'second': 70-6+7,'cases_name':'king.npy'}
    Davidson = {'first': 53, 'second': 70-6+7,'cases_name':'davidson.npy'}
    Will = { 'first': 58, 'second': 70-6+14,'cases_name':'will.npy'}
    Hillsborough = {'first': 57, 'second': 70-6+14,'cases_name':'hillsborough.npy'}

    city_list = {'Bergen':Bergen,'Middlesex':Middlesex,'Hudson':Hudson,'Philadelphia':Philadelphia,'NY':NewYorkCity,\
                 'Nassau':Nassau,'Cook':Cook, 'Union':Union,'Davidson':Davidson,'Will':Will,'MiamiDade':Miami_Dade,'Hillsborough':Hillsborough, \
                 'Suffolk':Suffolk,'Los_Angeles':Los_Angeles,'New_Haven':New_Haven,'Oakland':Oakland,'New_Orleans':New_Orleans,'Ocean':Ocean, \
                 'Harris':Harris,'King':King}

    return city_list

def init_city_list_selected():
    NewYorkCity = {'first': 51, 'second': 70-6+7,'cases_name':'nyc.npy'}
    Cook = { 'first': 48, 'second': 70-6+7,'cases_name':'cook.npy'}
    Nassau = {'first': 48, 'second':70-6+7,'cases_name':'nassau.npy'}
    Suffolk = {'first': 53, 'second': 70-6+7,'cases_name':'suffolk.npy'}
    Los_Angeles = { 'first': 55, 'second': 70-6+7,'cases_name':'los_angeles.npy'}
    Bergen = {'first': 50, 'second': 70-6+7,'cases_name':'bergen.npy'}
    Hudson = {'first': 60, 'second': 70-6+7,'cases_name':'hudson.npy'}
    Philadelphia = {'first': 54, 'second': 70-6+7,'cases_name':'philadelphia.npy'}
    Middlesex = {'first': 56, 'second': 70-6+7,'cases_name':'middlesex.npy'}
    Union = {'first': 57, 'second': 70-6+7,'cases_name':'union.npy'}
    Miami_Dade = { 'first': 54, 'second': 70-6+7,'cases_name':'miami_dade.npy'}
    New_Haven = { 'first': 60, 'second': 70-6+7,'cases_name':'new_haven.npy'}
    Oakland = {'first': 57, 'second': 70-6+7,'cases_name':'oakland.npy'}
    New_Orleans = {'first': 54, 'second': 70-6+7,'cases_name':'new_orleans.npy'}
    Ocean = {'first': 60, 'second': 70-6+7,'cases_name':'ocean.npy'}
    Harris = {'first': 60, 'second': 70-6+14,'cases_name':'harris.npy'}
    King = {'first': 48, 'second': 70-6+7,'cases_name':'king.npy'}
    Davidson = {'first': 53, 'second': 70-6+7,'cases_name':'davidson.npy'}
    Will = { 'first': 58, 'second': 70-6+14,'cases_name':'will.npy'}
    Hillsborough = {'first': 57, 'second': 70-6+14,'cases_name':'hillsborough.npy'}

    city_list = {'NY':NewYorkCity,'Cook':Cook,'Will':Will,'Suffolk':Suffolk,'Los_Angeles':Los_Angeles,\
        'Bergen':Bergen}
    return city_list

def load_cases_and_fitted(select=False, mobi_low = True, inf_low = True,data_type='dispersion'):
    if select:
        city_list = init_city_list_selected()
    else:
        city_list = init_city_list()
    
    cases = []
    fitted = []
    name_list = []
    for key,values in city_list.items():
        name_list.append(key)
        case = np.load('./cases/'+values['cases_name'])[values['first']:]
        cases.append(case)
        fit_input_dir = Path('/Users/hanzhenyu/Downloads/others/worldpop/rebuild/未命名文件夹/新40fit/'+data_type)

        if mobi_low:
            if inf_low:
                npy_data = fit_input_dir.glob('./*'+str(key)+'*/*mobiLow_infLow/'+'*SEIRhist*.npy')
            else:
                npy_data = fit_input_dir.glob('./*'+str(key)+'*/*mobiLow_infHigh/'+'*SEIRhist*.npy')
        else:
            if inf_low:
                npy_data = fit_input_dir.glob('./*'+str(key)+'*/*mobiHigh_infLow/'+'*SEIRhist*.npy')
            else:
                npy_data = fit_input_dir.glob('./*'+str(key)+'*/*mobiHigh_infHigh/'+'*SEIRhist*.npy')

        seir_data = [np.load(i) for i in npy_data][0]
        fitted.append(seir_data)

    return cases, fitted, name_list


def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g



def gini_of_area_def(x,points=10000):
    x_interp = np.linspace(0,1,points)
    out_array = np.interp(x_interp,np.linspace(x.min(),x.max(),len(x)),x)

    identical = x_interp
    absolute_error = np.abs(identical - out_array).sum() / identical.sum()
    return absolute_error


def load_cases_and_fitted_all_new_infection_Rt_rank(location='./results/fig2_dispersion/dispersion/', experiment_type = 'mobiLow_infLow',select=True):
    if select:
        city_list = init_city_list_selected()
    else:
        city_list = init_city_list()
    cases = []
    fitted = []
    name_list = []
    inf_rank = []
    new_inf_rank = []
    suscep_rank = []
    for key,values in city_list.items():
        name_list.append(key)
        case = np.load('./cases/'+values['cases_name'])[values['first']:]
        cases.append(case)
        fit_input_dir = Path(location)
        npy_data = fit_input_dir.glob('./*'+str(key)+'*/out_curves_' + experiment_type + '/*SEIRhist*.npy')
        seir_data = [np.load(i) for i in npy_data][0]
        fitted.append(seir_data)

        inf_rank_tmp = []
        new_inf_rank_tmp = []
        suscep_rank_tmp = []
        for i in range(seir_data.shape[1]-1):
            new_infection_num = seir_data[4, i+1, :]

            infection = seir_data[1, i, :] + seir_data[2, i, :]
            new_infection = seir_data[4, i, :]
            suscep = seir_data[0, i, :]

            select = infection > 0

            Rt = new_infection_num[select] / infection[select]

            Rt_rank_index  = np.argsort(Rt)
            inf_rank_tmp.append(infection[select][Rt_rank_index])
            new_inf_rank_tmp.append(new_infection[select][Rt_rank_index])
            suscep_rank_tmp.append(suscep[select][Rt_rank_index])

        inf_rank.append(np.array(inf_rank_tmp))
        new_inf_rank.append(np.array(new_inf_rank_tmp))
        suscep_rank.append(np.array(suscep_rank_tmp))
    return cases, fitted, name_list, inf_rank, new_inf_rank, suscep_rank


def draw_cdf_x_Rt_rank(location='./results/fig2_dispersion/dispersion/', experiment_type = 'mobiLow_infLow',select=False,flip_y=False,figsize=(8,6), draw_s=False,title=None,color_set = 1, fit=False, label=False, bigger_font = False):
    cases_data, fitted_data, city_name_list, inf_rank, new_inf_rank, suscept_rank = load_cases_and_fitted_all_new_infection_Rt_rank(location, experiment_type, select=select)
    for i in range(len(city_name_list)):
        if city_name_list[i] == 'NY':
            city_name_list[i] = 'New York City'
        elif city_name_list[i] == 'Los_Angeles':
            city_name_list[i] = 'Los Angeles'

    infection_rank = []
    susceptible_rank = []
    new_infection_rank = []
    for city in inf_rank:
        infection_rank_tmp = city[-1]
        length = np.linspace(0,1,num=infection_rank_tmp.shape[0],endpoint=True)
        ccdf = infection_rank_tmp.cumsum() / infection_rank_tmp.sum()
        infection_rank.append([length,ccdf])
    for city in new_inf_rank:
        new_infection_rank_tmp = city[-1]
        length = np.linspace(0,1,num=new_infection_rank_tmp.shape[0],endpoint=True)
        ccdf = new_infection_rank_tmp.cumsum() / new_infection_rank_tmp.sum()
        new_infection_rank.append([length,ccdf])
    for city in suscept_rank:
        suscept_rank_tmp = city[-1]
        length = np.linspace(0,1,num=suscept_rank_tmp.shape[0],endpoint=True)
        ccdf = suscept_rank_tmp.cumsum() / suscept_rank_tmp.sum()
        susceptible_rank.append([length,ccdf])

    max_length = 0
    for i in range(len(infection_rank)):
        tmp =  len(infection_rank[i][0])
        if tmp > max_length:
            max_length = tmp
    infection_rank_interp = []
    new_infection_rank_interp = []
    susceptible_rank_interp  = []
    for i in range(len(infection_rank)):
        resample_x = np.linspace(0,1,num=max_length,endpoint=True)
        out_array = np.interp(resample_x,infection_rank[i][0],infection_rank[i][1])
        infection_rank_interp.append([resample_x,out_array])

        resample_x = np.linspace(0,1,num=max_length,endpoint=True)
        out_array = np.interp(resample_x,new_infection_rank[i][0],new_infection_rank[i][1])
        new_infection_rank_interp.append([resample_x,out_array])

        resample_x = np.linspace(0,1,num=max_length,endpoint=True)
        out_array = np.interp(resample_x,susceptible_rank[i][0],susceptible_rank[i][1])
        susceptible_rank_interp.append([resample_x,out_array])




    infection_rank_interp = np.array(infection_rank_interp)
    new_infection_rank_interp = np.array(new_infection_rank_interp)
    susceptible_rank_interp = np.array(susceptible_rank_interp)

    new_inf_gini = gini_of_area_def(new_infection_rank_interp[:,1,:].mean(axis=0))
    inf_gini = gini_of_area_def(infection_rank_interp[:,1,:].mean(axis=0))
    susceptible_gini = gini_of_area_def(susceptible_rank_interp[:,1,:].mean(axis=0))
    print(new_inf_gini,inf_gini,susceptible_gini)
    fig4, ax4 = plt.subplots(figsize=figsize,)
    if color_set == 0:
        line_color = ['#4983EB','#3B9E29','#EB9D60']
        shadow_color = ['#abc6e4','#4AD731','#EB9D60']
        alpha_color = [0.2,0.2,0.2]
    elif color_set == 1:
        line_color = ['#34b0c6','#f69d29']
        
        shadow_color = ['#2ed7f3','#fc9e2b']
        alpha_color = [0.2,0.2,0.8]

    ax4.plot(np.linspace(0,1,num=max_length,endpoint=True), np.linspace(0,1,num=max_length,endpoint=True),linewidth=2,color = 'k', linestyle=':')

    ax4.plot(new_infection_rank_interp[0,0,:], new_infection_rank_interp[:,1,:].mean(axis=0),linewidth=2,color = line_color[0],label= 'New Infections, Gini Index:%(gini).3f'%{'gini':new_inf_gini})
    ax4.fill_between(new_infection_rank_interp[0,0,:], 
                    np.percentile(new_infection_rank_interp[:,1,:],25,axis=0),
                    np.percentile(new_infection_rank_interp[:,1,:],75,axis=0), alpha=alpha_color[0], color=shadow_color[0])
    x = [new_infection_rank_interp[0,0,:][int(i*(len(new_infection_rank_interp[0,0,:])/10))] for i in range(10)]
    y = [new_infection_rank_interp[:,1,:].mean(axis=0)[int(i*(len(new_infection_rank_interp[:,1,:].mean(axis=0))/10))] for i in range(10)]
    ax4.scatter(x,y,marker='d',color = line_color[0])


    ax4.plot(infection_rank_interp[0,0,:], infection_rank_interp[:,1,:].mean(axis=0), linewidth=2,color = line_color[1],label= 'Infected Population, Gini Index:%(gini).3f'%{'gini':inf_gini})
    ax4.fill_between(infection_rank_interp[0,0,:], 
                    np.percentile(infection_rank_interp[:,1,:],25,axis=0),
                    np.percentile(infection_rank_interp[:,1,:],75,axis=0), alpha=alpha_color[1], color=shadow_color[1])
    x = [infection_rank_interp[0,0,:][int(i*(len(infection_rank_interp[0,0,:])/10))] for i in range(10)]
    y = [infection_rank_interp[:,1,:].mean(axis=0)[int(i*(len(infection_rank_interp[:,1,:].mean(axis=0))/10))] for i in range(10)]
    ax4.scatter(x,y,marker='d',color = line_color[1])


    if draw_s:
        ax4.plot(susceptible_rank_interp[0,0,:], susceptible_rank_interp[:,1,:].mean(axis=0), linewidth=2, color=line_color[2],label= 'Susceptible, Gini Index:%(gini).3f'%{'gini':susceptible_gini})
        ax4.fill_between(susceptible_rank_interp[0,0,:],
                    np.percentile(susceptible_rank_interp[:,1,:],25,axis=0),
                    np.percentile(susceptible_rank_interp[:,1,:],75,axis=0), alpha=alpha_color[2], color=shadow_color[2])
    
    def func_powerlaw(x, m, c, c0):
        return c0 + x**m * c
    if fit:
        sol1 = curve_fit(func_powerlaw, new_infection_rank_interp[0,0,:], new_infection_rank_interp[:,1,:].mean(axis=0), maxfev=2000 )
        r2 = mean_absolute_error(new_infection_rank_interp[:,1,:].mean(axis=0),func_powerlaw(new_infection_rank_interp[0,0,:],*sol1[0]))

        ax4.plot(new_infection_rank_interp[0,0,:], func_powerlaw(new_infection_rank_interp[0,0,:],*sol1[0]), linewidth=2, color=line_color[0],label= 'Power-Law Fit for Secondary Infection, $R^2$:%(r2).3f'%{'r2':r2}, linestyle='--')
        sol2= curve_fit(func_powerlaw, infection_rank_interp[0,0,:], infection_rank_interp[:,1,:].mean(axis=0), maxfev=2000 )
        r2 = mean_absolute_error(infection_rank_interp[:,1,:].mean(axis=0),func_powerlaw(infection_rank_interp[0,0,:],*sol2[0]))

        ax4.plot(infection_rank_interp[0,0,:], func_powerlaw(infection_rank_interp[0,0,:],*sol2[0]), linewidth=2, color=line_color[1],label= 'Power-Law Fit for Infected, $R^2$:%(r2).3f'%{'r2':r2}, linestyle='--')

    if label:
        plt.xlim(0,1)
        plt.ylim(0,1)
        a = infection_rank_interp[:,1,:].mean(axis=0)
        b = new_infection_rank_interp[:,1,:].mean(axis=0)
        plt.plot([0.8,0.8],[0,a[int(len(a)*0.8)]],'k--',linewidth=1.5)
        plt.plot([0.8,0.8],[0,b[int(len(b)*0.8)]],'k--',linewidth=1.5)
        plt.plot([0,0.8],[a[int(len(a)*0.8)],a[int(len(a)*0.8)]],'k--',linewidth=1.5)
        plt.plot([0,0.8],[b[int(len(b)*0.8)],b[int(len(b)*0.8)]],'k--',linewidth=1.5)
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.text((xlim[1] - xlim[0])*0.7 - 0.05 + xlim[0], (ylim[1]-ylim[0]) * a[int(len(a)*0.8)] - 0.06 + ylim[0], '(0.8,%(x).3f)'%{'x':a[int(len(a)*0.8)]},fontdict = {'size': 16, 'color': 'black'})
        plt.text((xlim[1] - xlim[0])*0.7 - 0.05 + xlim[0], (ylim[1]-ylim[0]) * b[int(len(b)*0.8)] - 0.06 + ylim[0], '(0.8,%(x).3f)'%{'x':b[int(len(b)*0.8)]},fontdict = {'size': 16, 'color': 'black'})

    plt.xlim(0,1)
    plt.ylim(0,1)
    if flip_y:
        plt.invert_yaxis()
    if not bigger_font:
        plt.ylabel("Cumulative Distribution Function",fontsize=20)
        plt.xlabel("Percentile of Infectee-Infector Ratio's Ranking",fontsize=20)

        if title:
            plt.title(title, size=18)
    else:
        plt.ylabel("Cumulative Distribution Function",fontsize=20)
        plt.xlabel("Percentile of Infectee-Infector Ratio's Ranking",fontsize=20)
        if title:
            plt.title(title, size=24)


    major_ticks = np.arange(0, 1, 0.1)
    minor_ticks = np.arange(0, 1, 0.05)
    ax4.set_xticks(major_ticks)
    ax4.set_xticks(minor_ticks, minor=True)
    ax4.set_yticks(major_ticks)
    ax4.set_yticks(minor_ticks, minor=True)
    ax4.grid(which='minor', ls='-', alpha=0.5)
    ax4.grid(which='major',ls='--', alpha=0.8)

    if not bigger_font:
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=18)
    else:
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    # fig2a
    draw_cdf_x_Rt_rank(location='./results/fig2_dispersion/dispersion/', experiment_type = 'mobiLow_infLow',select=False,figsize=(10,6),draw_s=False, title=None,color_set = 1,fit=False,label=True)

    # fig2c
    draw_cdf_x_Rt_rank(location='./results/fig2_dispersion/dispersion_allNo/', experiment_type = 'mobiLow_infLow',select=False,figsize=(8,6),draw_s=False, title='Random Movement',color_set = 1,fit=False, bigger_font = True)

    # fig2d
    draw_cdf_x_Rt_rank(location='./results/fig2_dispersion/dispersion_noPop/', experiment_type = 'mobiLow_infLow',select=False,figsize=(8,6),draw_s=False, title='Only Considering Rule of Travel Cost',color_set = 1,fit=False, bigger_font = True)

    # fig2e
    draw_cdf_x_Rt_rank(location='./results/fig2_dispersion/dispersion_noDist/', experiment_type = 'mobiLow_infLow',select=False,figsize=(8,6),draw_s=False, title='Only Considering Rule of Social Interaction',color_set = 1,fit=False, bigger_font = True)


