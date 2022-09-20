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
import seaborn as sns; #sns.set(color_codes=True)
import datetime


plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13

def init_city_list(country):
    NewYorkCity = {'first': 51, 'second': 70-6+7,'cases_name':'nyc.npy','pop_name':'NY_noStaten_pop_44.npy'}
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
    

    # city_list = {'Bergen':Bergen,'Middlesex':Middlesex,'Hudson':Hudson,'Philadelphia':Philadelphia,'NY':NewYorkCity,\
    #              'Nassau':Nassau,'Cook':Cook, 'Union':Union,'Davidson':Davidson,'Will':Will,'MiamiDade':Miami_Dade,'Hillsborough':Hillsborough, \
    #              'Suffolk':Suffolk,'Los_Angeles':Los_Angeles,'New_Haven':New_Haven,'Oakland':Oakland,'New_Orleans':New_Orleans,'Ocean':Ocean, \
    #              'Harris':Harris,'King':King}
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



def case_study_one_panel():
    def load_pop(data_dir):
        pop = np.load(data_dir).reshape(-1, 1).astype('float32')
        std = pop[pop > 100].std()
        mean = pop[pop > 100].mean()
        upper_bond = mean + 3 * std
        pop = np.where(pop > upper_bond, upper_bond, pop)
        len_of_pop = np.sqrt(len(pop)).astype(int)
        return pop.reshape(len_of_pop,len_of_pop)

    US_city_list, India, Brazil = init_city_list('All')

    pop_dis = []
    city_name_list = []
    top = []
    our = []
    for key,values in US_city_list.items():
        if key == 'Los_Angeles':

            city_name_list.append(key)
            pop = load_pop('./population/'+values['pop_name'])
            pop_dis.append(pop)
            fit_input_dir = Path('./results/fig3_control_policy_case_study')
            npy_datas_top = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves_top'+'/*index_*.npy')])
            fitted_data_top = np.load(npy_datas_top[0]).reshape(-1)
            top.append(fitted_data_top)
            npy_datas_worst_ii_ratio = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves_worst_ii_ratio'+'/*index_*.npy')])

            fitted_data_worst_ii_ratio = np.load(npy_datas_worst_ii_ratio[0],allow_pickle=True)
            our.append(fitted_data_worst_ii_ratio)

    for key,values in India.items():
        if key == 'Delhi':
            city_name_list.append(key)
            pop = load_pop('./population/'+values['pop_name'])
            pop_dis.append(pop)
            fit_input_dir = Path('./results/fig3_control_policy_case_study')
            npy_datas_top = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves_top'+'/*index_*.npy')])
            fitted_data_top = np.load(npy_datas_top[0]).reshape(-1)
            top.append(fitted_data_top)
            npy_datas_worst_ii_ratio = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves_worst_ii_ratio'+'/*index_*.npy')])

            fitted_data_worst_ii_ratio = np.load(npy_datas_worst_ii_ratio[0],allow_pickle=True)
            our.append(fitted_data_worst_ii_ratio)

    for key,values in Brazil.items():
        if key == 'SaoPaulo':
            city_name_list.append(key)
            pop = load_pop('./population/'+values['pop_name'])
            pop_dis.append(pop)
            fit_input_dir = Path('./results/fig3_control_policy_case_study')
            npy_datas_top = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves_top'+'/*index_*.npy')])
            fitted_data_top = np.load(npy_datas_top[0]).reshape(-1)
            top.append(fitted_data_top)
            npy_datas_worst_ii_ratio = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves_worst_ii_ratio'+'/*index_*.npy')])

            fitted_data_worst_ii_ratio = np.load(npy_datas_worst_ii_ratio[0],allow_pickle=True)
            our.append(fitted_data_worst_ii_ratio)


    for i in range(len(city_name_list)):
        if city_name_list[i] == 'NY':
            city_name_list[i]='New York City'
        elif city_name_list[i] == 'New_Orleans':
            city_name_list[i] = 'New Orleans'
        elif city_name_list[i] == 'New_Haven':
            city_name_list[i] = 'New Haven'
        elif city_name_list[i] == 'Los_Angeles':
            city_name_list[i] = 'Los Angeles'
        elif city_name_list[i] == 'RioDeJaneiro':
            city_name_list[i] = 'Rio De Janeiro'
        elif city_name_list[i] == 'SaoPaulo':
            city_name_list[i] = 'São Paulo'

    color_lists = sns.color_palette("pastel")
    
    for i in range(len(city_name_list)):
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(9, 6))
        axes.imshow(pop_dis[i])
        index_row = []
        index_col = []

        # show difference percentage
        count = 0
        for x in our[i][-1]:
            if x not in top[i]:
                count += 1
        percentage_of_diff = count / len(our[i][-1])
        print(percentage_of_diff)

        first_label = True
        top_rc_list = []
        for index in top[i]:
            len_of_pop = pop_dis[i].shape[0]
            r = index // len_of_pop  
            c = index % len_of_pop
            tmp = [r,c]
            top_rc_list.append(tmp)
            xlim = axes.get_xlim()
            ylim = axes.get_ylim()
            edge_len = (xlim[1] - xlim[0] ) / len_of_pop
            if first_label:
                rect = patches.Rectangle((c * edge_len - 0.5*edge_len, r * edge_len- 0.5*edge_len), edge_len, edge_len, linewidth=3, edgecolor='b', linestyle='-', facecolor='none',label='Top Populated')
                first_label = False
            else:
                rect = patches.Rectangle((c * edge_len - 0.5*edge_len, r * edge_len- 0.5*edge_len), edge_len, edge_len, linewidth=3, edgecolor='b', linestyle='-', facecolor='none')

            axes.add_patch(rect)



        if key == 'Los_Angeles':
            first = True
            first_label = True
            first_label_common = True
            index_of_time = our[i][-1]  
            for index in index_of_time:
                len_of_pop = pop_dis[i].shape[0]
                r = index // len_of_pop  
                c = index % len_of_pop
                tmp = [r,c]
                if tmp not in top_rc_list:
                    xlim = axes.get_xlim()
                    ylim = axes.get_ylim()
                    edge_len = (xlim[1] - xlim[0] )  / len_of_pop
                    if first:
                        if first_label:
                            rect = patches.Rectangle((c * edge_len - 0.5*edge_len, r * edge_len- 0.5*edge_len), edge_len, edge_len, linewidth=3, edgecolor='#e71134',facecolor='none',label='Our Model')
                            first_label = False
                            axes.add_patch(rect)
                        else:
                            rect = patches.Rectangle((c * edge_len - 0.5*edge_len, r * edge_len- 0.5*edge_len), edge_len, edge_len, linewidth=3, edgecolor='#e71134',facecolor='none',)
                            axes.add_patch(rect)
                else:
                    xlim = axes.get_xlim()
                    ylim = axes.get_ylim()
                    edge_len = (xlim[1] - xlim[0] )  / len_of_pop
                    if first:
                        if first_label_common:
                            rect = patches.Rectangle((c * edge_len - 0.5*edge_len, r * edge_len- 0.5*edge_len), edge_len, edge_len, linewidth=3, edgecolor='#9c19d9',facecolor='none',label='Overlap')
                            first_label_common = False
                            axes.add_patch(rect)
                        else:
                            rect = patches.Rectangle((c * edge_len - 0.5*edge_len, r * edge_len- 0.5*edge_len), edge_len, edge_len, linewidth=3, edgecolor='#9c19d9',facecolor='none',)
                            axes.add_patch(rect)

        else:
            first = True
            first_label = True
            first_label_common = True
            index_of_time = our[i][-1]  
            for index in index_of_time:
                len_of_pop = pop_dis[i].shape[0]
                r = index // len_of_pop  
                c = index % len_of_pop
                tmp = [r,c]
                if tmp not in top_rc_list:
                    xlim = axes.get_xlim()
                    ylim = axes.get_ylim()
                    edge_len = (xlim[1] - xlim[0] )  / len_of_pop
                    if first:
                        if first_label:
                            rect = patches.Rectangle((c * edge_len - 0.5*edge_len, r * edge_len- 0.5*edge_len), edge_len, edge_len, linewidth=3, edgecolor='#e71134',facecolor='none',label='Our Model')
                            first_label = False
                            axes.add_patch(rect)
                        else:
                            rect = patches.Rectangle((c * edge_len - 0.5*edge_len, r * edge_len- 0.5*edge_len), edge_len, edge_len, linewidth=3, edgecolor='#e71134',facecolor='none',)
                            axes.add_patch(rect)
                else:
                    xlim = axes.get_xlim()
                    ylim = axes.get_ylim()
                    edge_len = (xlim[1] - xlim[0] )  / len_of_pop
                    if first:
                        if first_label_common:
                            rect = patches.Rectangle((c * edge_len - 0.5*edge_len, r * edge_len- 0.5*edge_len), edge_len, edge_len, linewidth=3, edgecolor='#9c19d9',facecolor='none',label='Overlap')
                            first_label_common = False
                            axes.add_patch(rect)
                        else:
                            rect = patches.Rectangle((c * edge_len - 0.5*edge_len, r * edge_len- 0.5*edge_len), edge_len, edge_len, linewidth=3, edgecolor='#9c19d9',facecolor='none',)
                            axes.add_patch(rect)

        axes.set_title(city_name_list[i] + ' (%(x).2f%% Overlap)'%{'x':(1-percentage_of_diff)*100},fontsize=24)
        axes.set_xticks([])
        axes.set_yticks([])


        axes.set_xticks([])
        axes.set_yticks([])
        axes.legend(fontsize=16,loc='upper right')


        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    case_study_one_panel()




