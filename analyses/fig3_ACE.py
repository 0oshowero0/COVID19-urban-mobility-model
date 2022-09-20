import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from argparse import ArgumentParser
from pathlib import Path
import json
import pandas as pd
from math import floor
from sklearn.metrics import r2_score
import seaborn as sns
import datetime


plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13

def init_city_list(country):
    NewYorkCity = {'first': 51, 'second': 70-6+7,'cases_name':'nyc.npy','pop_name':'nyc_pop.npy'}
    Cook = { 'first': 48, 'second': 70-6+7,'cases_name':'cook.npy','pop_name':'cook_pop.npy'}
    Nassau = {'first': 48, 'second':70-6+7,'cases_name':'nassau.npy','pop_name':'nassau_pop.npy'}
    Suffolk = {'first': 53, 'second': 70-6+7,'cases_name':'suffolk.npy','pop_name':'suffolk_pop.npy'}
    Los_Angeles = { 'first': 55, 'second': 70-6+7,'cases_name':'los_angeles.npy','pop_name':'los_angeles_pop.npy'}
    Bergen = {'first': 50, 'second': 70-6+7,'cases_name':'bergen.npy','pop_name':'bergen_pop.npy'}
    Hudson = {'first': 60, 'second': 70-6+7,'cases_name':'hudson.npy','pop_name':'hudson_pop.npy'}
    Philadelphia = {'first': 54, 'second': 70-6+7,'cases_name':'philadelphia.npy','pop_name':'philadelphia_pop.npy'}
    Middlesex = {'first': 56, 'second': 70-6+7,'cases_name':'middlesex.npy','pop_name':'middlesex_pop.npy'}
    Union = {'first': 57, 'second': 70-6+7,'cases_name':'union.npy','pop_name':'union_pop.npy'}
    Miami_Dade = { 'first': 54, 'second': 70-6+7,'cases_name':'miami_dade.npy','pop_name':'miami_dade_pop.npy'}
    New_Haven = { 'first': 60, 'second': 70-6+7,'cases_name':'new_haven.npy','pop_name':'new_haven_pop.npy'}
    Oakland = {'first': 57, 'second': 70-6+7,'cases_name':'oakland.npy','pop_name':'oakland_pop.npy'}
    New_Orleans = {'first': 54, 'second': 70-6+7,'cases_name':'new_orleans.npy','pop_name':'new_orleans_pop.npy'}
    Ocean = {'first': 60, 'second': 70-6+7,'cases_name':'ocean.npy','pop_name':'ocean_pop.npy'}
    Harris = {'first': 60, 'second': 70-6+14,'cases_name':'harris.npy','pop_name':'harris_pop.npy'}
    King = {'first': 48, 'second': 70-6+7,'cases_name':'king.npy','pop_name':'king_pop.npy'}
    Davidson = {'first': 53, 'second': 70-6+7,'cases_name':'davidson.npy','pop_name':'davidson_pop.npy'}
    Will = { 'first': 58, 'second': 70-6+14,'cases_name':'will.npy','pop_name':'will_pop.npy'}
    Hillsborough = {'first': 57, 'second': 70-6+14,'cases_name':'hillsborough.npy','pop_name':'hillsborough_pop.npy'}


    Bengaluru = {'start':'2021-03-12', 'change':'2021-05-01', 'final':'2021-07-31', 'city_name':'Bengaluru Urban','sample_rate':4, 'pop_name':'bengaluru_pop.npy'}
    Delhi = {'start':'2021-03-22', 'change':'2021-05-04', 'final':'2021-07-31', 'city_name':'Delhi','sample_rate':4, 'pop_name':'delhi_pop.npy'}
    Mumbai = {'start':'2021-02-11', 'change':'2021-04-17', 'final':'2021-07-31', 'city_name':'Mumbai','sample_rate':4, 'pop_name':'mumbai_pop.npy'}
    Pune = {'start':'2021-02-14', 'change':'2021-05-14', 'final':'2021-07-31', 'city_name':'Pune','sample_rate':4, 'pop_name':'pune_pop.npy'}
    Thane = {'start':'2021-02-24', 'change':'2021-05-01', 'final':'2021-07-31', 'city_name':'Thane','sample_rate':4, 'pop_name':'thane_pop.npy'}
    India = {'Bengaluru':Bengaluru,'Delhi':Delhi, 'Mumbai':Mumbai, 'Pune':Pune, 'Thane':Thane}

    BeloHorizonte = {'start':'2021-01-01', 'change':'2021-04-27', 'final':'2021-07-31', 'city_name':'Belo Horizonte','sample_rate':4, 'pop_name':'beloHorizonte_pop.npy'}
    Brasilia = {'start':'2021-01-01', 'change':'2021-02-23', 'final':'2021-07-31', 'city_name':'Bras_­lia','sample_rate':4, 'pop_name':'brasilia_pop.npy'}
    Fortaleza = {'start':'2021-01-01', 'change':'2021-04-01', 'final':'2021-07-31', 'city_name':'Fortaleza','sample_rate':4, 'pop_name':'fortaleza_pop.npy'}
    RioDeJaneiro = {'start':'2021-01-01', 'change':'2021-04-26', 'final':'2021-07-31', 'city_name':'Rio de Janeiro','sample_rate':4, 'pop_name':'rioDeJaneiro_pop.npy'}
    SaoPaulo = {'start':'2021-01-01', 'change':'2021-03-10', 'final':'2021-07-31', 'city_name':'S_£o Paulo','sample_rate':4, 'pop_name':'saoPaulo_pop.npy'}
    Brazil = {'BeloHorizonte':BeloHorizonte, 'Brasilia':Brasilia, 'Fortaleza':Fortaleza, 'RioDeJaneiro':RioDeJaneiro, 'SaoPaulo':SaoPaulo}
    

    city_list = {'Bergen':Bergen,'MiamiDade':Miami_Dade,'Nassau':Nassau,'Oakland':Oakland,'Suffolk':Suffolk,'Union':Union, \
                 'Cook': Cook,'Davidson':Davidson, 'Los_Angeles':Los_Angeles, 'Middlesex':Middlesex, \
                 'Harris':Harris, 'New_Orleans':New_Orleans, 'Will':Will, \
                 'Hillsborough': Hillsborough,'Hudson':Hudson,'King':King,'New_Haven':New_Haven,'NY':NewYorkCity,'Ocean':Ocean,'Philadelphia':Philadelphia,
                 }

    if country == "US":
        return city_list
    elif country == 'India':
        return India
    elif country == 'Brazil':
        return Brazil
    elif country == 'All':
        return city_list, India, Brazil


def gen_date_axis(city_list, country='US'):
    if country=='US':
        city_name = []
        date_axis = []
        begin_date = datetime.datetime.strptime('2020/1/22', "%Y/%m/%d")
        for key,values in city_list.items():
            city_name.append(key)
            date_axis.append([mdates.date2num(begin_date + datetime.timedelta(days=i)) for i in range(values['first'],99+1)])
        date_axis_dict = dict()
        for i in range(len(city_name)):
            date_axis_dict[city_name[i]] = date_axis[i]
    elif country == 'India' or country == 'Brazil':

        city_name = []
        date_axis = []
        for key,values in city_list.items():
            city_name.append(key)
            begin_date = datetime.datetime.strptime(values['start'], "%Y-%m-%d")
            end_date = datetime.datetime.strptime(values['final'], "%Y-%m-%d")
            tmp = [mdates.date2num(begin_date + datetime.timedelta(days=i)) for i in range((end_date - begin_date).days + 1)]
            date_axis.append(tmp)

        date_axis_dict = dict()
        for i in range(len(city_name)):
            date_axis_dict[city_name[i]] = date_axis[i]

    return date_axis_dict


def load_cases_Ind(data_dir, name):
    data = pd.read_csv(data_dir)
    cases = data[data['District']==name]
    return cases

def load_cases_Bra(data_dir, name):
    data = pd.read_csv(data_dir)
    cases = data[data['Município']==name]
    return cases

def load_cases_and_fitted_all(type_of_policy,country):
    if type_of_policy == 'none':
        type_of_policy_us = 'nothing'
        US_city_list, India, Brazil = init_city_list('All')
        cases_us = []
        fitted_us = []
        name_list_us = []
        for key,values in US_city_list.items():
            name_list_us.append(key)
            case = np.load('./cases/'+values['cases_name'])[values['first']:]
            cases_us.append(case)
            fit_input_dir = Path('./results/fig3_control_policy')
            npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves_'+type_of_policy_us+'/*curve_*.npy')])       
            fitted_data = np.load(npy_datas[0]).reshape(-1)
            fitted_us.append(fitted_data)
    else:
        US_city_list, India, Brazil = init_city_list('All')
        cases_us = []
        fitted_us = []
        name_list_us = []
        for key,values in US_city_list.items():
            name_list_us.append(key)
            case = np.load('./cases/'+values['cases_name'])[values['first']:]
            cases_us.append(case)
            fit_input_dir = Path('./results/fig3_control_policy')
            npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves_'+type_of_policy+'/*curve_*.npy')])       
            fitted_data = np.load(npy_datas[0]).reshape(-1)
            fitted_us.append(fitted_data)


    cases_india = []
    fitted_india = []
    name_list_india = []
    for key,values in India.items():
        name_list_india.append(key)
        cases_data = load_cases_Ind('./cases/India_epidemic_district_timeline.csv', values['city_name'])

        start_index = np.where(cases_data.columns == values['start'])[0]
        change_index = np.where(cases_data.columns == values['change'])[0]
        final_index = np.where(cases_data.columns == values['final'])[0]

        cases_data_processed = cases_data.iloc[:,int(start_index):int(final_index+1)].to_numpy().reshape(-1)

        cases_india.append(cases_data_processed)
        fit_input_dir = Path('./results/fig3_control_policy')
        npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves_'+type_of_policy+'/*curve_*.npy')])       
        fitted_data = np.load(npy_datas[0]).reshape(-1)

        origin_x = np.linspace(0, fitted_data.shape[0]-1, num=fitted_data.shape[0], endpoint=True)
        num_new_points = cases_data_processed.shape[0]
        resample_x = np.linspace(0, fitted_data.shape[0]-1, num=num_new_points, endpoint=True)
        fitted_resample = np.interp(x=resample_x, xp=origin_x, fp=fitted_data)

        fitted_india.append(fitted_resample)

    cases_bra = []
    fitted_bra = []
    name_list_bra = []
    for key,values in Brazil.items():
        name_list_bra.append(key)
        cases_data = load_cases_Bra('./cases/Brazil_epidemic_district_timeline.csv', values['city_name'])

        start_index = np.where(cases_data.columns == values['start'])[0]
        change_index = np.where(cases_data.columns == values['change'])[0]
        final_index = np.where(cases_data.columns == values['final'])[0]

        cases_data_processed = cases_data.iloc[:,int(start_index):int(final_index+1)].to_numpy().reshape(-1)

        cases_bra.append(cases_data_processed)
        fit_input_dir = Path('./results/fig3_control_policy')
        npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves_'+type_of_policy+'/*curve_*.npy')])       
        fitted_data = np.load(npy_datas[0]).reshape(-1)

        origin_x = np.linspace(0, fitted_data.shape[0]-1, num=fitted_data.shape[0], endpoint=True)
        num_new_points = cases_data_processed.shape[0]
        resample_x = np.linspace(0, fitted_data.shape[0]-1, num=num_new_points, endpoint=True)
        fitted_resample = np.interp(x=resample_x, xp=origin_x, fp=fitted_data)

        fitted_bra.append(fitted_resample)

    if country == 'US':
        cases_concate = cases_us
        fitted_concate = fitted_us
        name_list_concate = name_list_us
    elif country=='India':
        cases_concate = cases_india
        fitted_concate = fitted_india
        name_list_concate = name_list_india
    elif country == 'Brazil':
        cases_concate = cases_bra
        fitted_concate = fitted_bra
        name_list_concate = name_list_bra

    return cases_concate, fitted_concate, name_list_concate


def select_common_date(country,city_list):
    date_axis = gen_date_axis(city_list, country)
    
    init = True
    for key, value in date_axis.items():
        if init:
            min_date = value[0]
            max_date = value[-1]
            init = False
        else:
            if value[0] > min_date:
                min_date = value[0]
            if value[-1] < max_date:
                max_date = value[-1]

    cases_select_index_dict = {}  
    for key, value in date_axis.items():
        a = np.where(np.array(value) == min_date)
        b = np.where(np.array(value) == max_date)
        cases_select_index_dict[key] = (int(a[0]),int(b[0] + 1))  

    date_axis_new = date_axis[key][cases_select_index_dict[key][0]:cases_select_index_dict[key][1]]

    return date_axis_new, cases_select_index_dict
        

def load_pop(city_list):
    pop = []
    for key,values in city_list.items():
        pop.append(np.load('./population/' + values['pop_name']).sum())

    return pop


def draw_all_range(country):
    city_list = init_city_list(country)
    
    population = load_pop(city_list)
    cases_top, fitted_top,  city_name_list_top = load_cases_and_fitted_all('top',country)
    cases_random, fitted_random,  city_name_list_random = load_cases_and_fitted_all('random',country)
    cases_ii_most, fitted_ii_most,  city_name_list_ii_most = load_cases_and_fitted_all('worst_ii_ratio',country)
    cases_none, fitted_none,  city_name_list_none = load_cases_and_fitted_all('none',country)

    #######################################################################################################
    # No intervention
    date_axis, cases_select_index_dict = select_common_date(country, city_list)

    fitted_none_process_select = []
    for i in range(len(city_name_list_none)):
        cases = cases_none[i][cases_select_index_dict[city_name_list_none[i]][0]:cases_select_index_dict[city_name_list_none[i]][1]] / population[i]
        fitted_none_process_select.append(cases)

    fitted_none_process_select = np.array(fitted_none_process_select).reshape(len(city_name_list_none),-1)


    #######################################################################################################
    # Top populated
    date_axis, cases_select_index_dict = select_common_date(country, city_list)

    fitted_top_process_select = []
    for i in range(len(city_name_list_top)):
        cases = fitted_top[i][cases_select_index_dict[city_name_list_top[i]][0]:cases_select_index_dict[city_name_list_top[i]][1]] / population[i]
        fitted_top_process_select.append(cases)

    fitted_top_process_select = np.array(fitted_top_process_select).reshape(len(city_name_list_top),-1)


    #######################################################################################################
    # Random
    date_axis, cases_select_index_dict = select_common_date(country, city_list)

    fitted_random_process_select = []
    for i in range(len(city_name_list_random)):
        cases = fitted_random[i][cases_select_index_dict[city_name_list_random[i]][0]:cases_select_index_dict[city_name_list_random[i]][1]] / population[i]
        fitted_random_process_select.append(cases)
    fitted_random_process_select = np.array(fitted_random_process_select).reshape(len(city_name_list_random),-1)

    #######################################################################################################
    # Our Model
    date_axis, cases_select_index_dict = select_common_date(country, city_list)

    fitted_ii_most_process_select = []
    for i in range(len(city_name_list_ii_most)):
        cases = fitted_ii_most[i][cases_select_index_dict[city_name_list_ii_most[i]][0]:cases_select_index_dict[city_name_list_ii_most[i]][1]] / population[i]
        fitted_ii_most_process_select.append(cases)
    fitted_ii_most_process_select = np.array(fitted_ii_most_process_select).reshape(len(city_name_list_ii_most),-1)


    color_lists = sns.color_palette("pastel")
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,6))

    ax.plot(np.arange(0,len(date_axis),1),np.median(fitted_none_process_select,axis=0),linewidth=2,color = 'k',label='Complete Reopen')
    ax.fill_between(np.arange(0,len(date_axis),1), 
                    np.percentile(fitted_none_process_select,25,axis=0),
                    np.percentile(fitted_none_process_select,75,axis=0), alpha=.2, color=color_lists[4])


    ax.plot(np.arange(0,len(date_axis),1),np.median(fitted_random_process_select,axis=0),linewidth=2,color = '#5eab38',label= 'Randomly Selected Neighborhoods')
    ax.fill_between(np.arange(0,len(date_axis),1), 
                    np.percentile(fitted_random_process_select,25,axis=0),
                    np.percentile(fitted_random_process_select,75,axis=0), alpha=.2, color=color_lists[2])

    ax.plot(np.arange(0,len(date_axis),1),np.median(fitted_top_process_select,axis=0),linewidth=2,color = color_lists[1],label= 'Top Populated Neighborhoods')
    ax.fill_between(np.arange(0,len(date_axis),1), 
                    np.percentile(fitted_top_process_select,25,axis=0),
                    np.percentile(fitted_top_process_select,75,axis=0), alpha=.2, color=color_lists[1])




    ax.plot(np.arange(0,len(date_axis),1),np.median(fitted_ii_most_process_select,axis=0),linewidth=2,color = '#e71134',label= 'Neighborhoods Selected by Our Model')
    ax.fill_between(np.arange(0,len(date_axis),1),
                    np.percentile(fitted_ii_most_process_select,25,axis=0),
                    np.percentile(fitted_ii_most_process_select,75,axis=0), alpha=.2, color=color_lists[3])
    ax.set_ylabel("Percentage of Cumulated Infected Population",size=18)
    ax.set_xlabel("Day",size=20)

    print('No Intervention')
    print(np.median(fitted_none_process_select,axis=0)[-1])
    print(np.percentile(fitted_none_process_select,25,axis=0)[-1])
    print(np.percentile(fitted_none_process_select,75,axis=0)[-1])
    print('Random')
    print(np.median(fitted_random_process_select,axis=0)[-1])
    print(np.percentile(fitted_random_process_select,25,axis=0)[-1])
    print(np.percentile(fitted_random_process_select,75,axis=0)[-1])
    print('Top')
    print(np.median(fitted_top_process_select,axis=0)[-1])
    print(np.percentile(fitted_top_process_select,25,axis=0)[-1])
    print(np.percentile(fitted_top_process_select,75,axis=0)[-1])
    print('Most Infecious')
    print(np.median(fitted_ii_most_process_select,axis=0)[-1])
    print(np.percentile(fitted_ii_most_process_select,25,axis=0)[-1])
    print(np.percentile(fitted_ii_most_process_select,75,axis=0)[-1])

    if country == 'US':
        ori_y_ticks = ax.get_yticks()
        new_y_ticks = ["%(i).1f%%"%{'i':i*100} for i in ori_y_ticks]

        plt.yticks(ori_y_ticks,new_y_ticks)
        plt.xticks(fontsize=18)
    elif country == 'India':
        ori_y_ticks = ax.get_yticks()
        new_y_ticks = ["%(i).1f%%"%{'i':i*100} for i in ori_y_ticks]

        plt.yticks(ori_y_ticks,new_y_ticks)
        plt.xticks(fontsize=17)
    elif country == "Brazil":
        ori_y_ticks = ax.get_yticks()
        new_y_ticks = ["%(i).1f%%"%{'i':i*100} for i in ori_y_ticks]

        plt.yticks(ori_y_ticks,new_y_ticks)
        plt.xticks(fontsize=17)

    
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16,loc='upper left')
    plt.tight_layout()
    plt.grid(which='major',ls='--', alpha=0.8)
    plt.show()



if __name__ == "__main__":

    draw_all_range('US')
    draw_all_range('India')
    draw_all_range('Brazil')





