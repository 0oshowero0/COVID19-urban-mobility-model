import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
from math import floor
from sklearn.metrics import r2_score
import datetime
import matplotlib.dates as mdates

plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13

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

def gen_date_axis(city_list):
    city_name = []
    date_axis = []
    begin_date = datetime.datetime.strptime('2020/1/22', "%Y/%m/%d")

    for key,values in city_list.items():
        city_name.append(key)
        date_axis.append([i- values['first'] for i in range(values['first'],99+1)])
    date_axis_dict = dict()
    for i in range(len(city_name)):
        date_axis_dict[city_name[i]] = date_axis[i]

    return date_axis_dict

def load_cases_and_fitted_all(type,step,):
    if type == 'date':
        step_dict = {}
        for i in range(100):
            step_dict['+' + str(i).zfill(2)] = (i + 1) * 2 
            step_dict['-' + str(i).zfill(2)] = - (i + 1) * 2 
    elif type == 'structure':
        step_dict = {}
    else:
        step_dict = {}
        for i in range(100):
            step_dict['+' + str(i).zfill(2)] = (i+1) / step
            step_dict['-' + str(i).zfill(2)] = - (i + 1) / step


    if type == 'mobility':
        input = Path('./results/fig4/change_parameters_mobility/')
    else:
        input = Path('./results/fig4/change_parameters/')
    city_list = init_city_list()
    cases = []
    fitted = []
    diff = []
    city_name_list = []
    percentage_list = []
    for key,values in city_list.items():
        city_name_list.append(key)
        case = np.load('./cases/'+values['cases_name'])[values['first']:]
        cases.append(case)
        fit_input_dir = input
        npy_datas_m = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves_'+type+'/*curve_-*.npy')])[::-1]
        npy_datas_p = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/out_curves_'+type+'/*curve_+*.npy')])            
        percentage_m= np.array([step_dict[i.stem.split('_')[-1]] for i in npy_datas_m])
        percentage_p= np.array([step_dict[i.stem.split('_')[-1]] for i in npy_datas_p])
        
        fitted_data_m = np.array([np.load(i).reshape(-1) for i in npy_datas_m])
        fitted_data_p = np.array([np.load(i).reshape(-1) for i in npy_datas_p])

        percentage = np.concatenate((np.concatenate((percentage_m,np.array([0])),axis=0),percentage_p),axis=0)   # 添加0点
        fitted_data = np.concatenate((np.concatenate((fitted_data_m,case.reshape(1,-1))),fitted_data_p))         # 添加0点
        fitted.append(fitted_data)

        percentage_list.append(percentage)

    for i in range(len(city_name_list)):
        if city_name_list[i] == 'NY':
            city_name_list[i]='New York City'
        elif city_name_list[i] == 'New_Orleans':
            city_name_list[i] = 'New Orleans'
        elif city_name_list[i] == 'New_Haven':
            city_name_list[i] = 'New Haven'
        elif city_name_list[i] == 'Los_Angeles':
            city_name_list[i] = 'Los Angeles'


    return cases, fitted, percentage_list, city_name_list




def draw_all_range(grain):
    
    city_list = init_city_list()
    date_axis = gen_date_axis(city_list)
    cases_pi, fitted_pi,  percentage_list_pi, city_name_list_pi = load_cases_and_fitted_all('pi', grain)
    cases_detect, fitted_detect, percentage_list_detect, city_name_list_detect = load_cases_and_fitted_all('detect',grain)
    cases_date, fitted_date, percentage_list_date, city_name_list_date = load_cases_and_fitted_all('date',2)

    cases_mobility, fitted_mobility, percentage_list_mobility, city_name_list_mobility = load_cases_and_fitted_all('mobility',100)
    
    #######################################################################################################
    high_pi_1 = []
    high_pi_2 = []
    low_pi_1 = []
    low_pi_2 = []
    max_length = 0
    max_index = -1
    for i in range(len(cases_pi)):
        cases = cases_pi[i]

        tmp =  len(cases)
        if tmp > max_length:
            max_length = tmp
            max_index = i
        lowest_idx = np.argsort(percentage_list_pi[i])[0]
        low2_idx = np.percentile(np.argsort(percentage_list_pi[i]),25,axis=0).astype(int)
        highest_idx = np.argsort(percentage_list_pi[i])[-1]
        high2_idx = np.percentile(np.argsort(percentage_list_pi[i]),75,axis=0).astype(int)

        low_pi_1.append(fitted_pi[i][lowest_idx]/cases[-1])
        high_pi_1.append(fitted_pi[i][highest_idx]/cases[-1])

        low_pi_2.append(fitted_pi[i][low2_idx]/cases[-1])
        high_pi_2.append(fitted_pi[i][high2_idx]/cases[-1])

    date_axis = date_axis[city_name_list_pi[max_index]]
    high_pi1_resample = []
    low_pi1_resample = []
    high_pi2_resample = []
    low_pi2_resample = []
    cases_resample = []
    for i in range(len(cases_pi)):
        origin_x = np.linspace(0, len(cases_pi[i]), num=len(cases_pi[i]),endpoint=False)
        resample_x = np.linspace(0, max_length, num=max_length,endpoint=False)
        out_array_low = np.interp(resample_x,origin_x,low_pi_1[i])
        low_pi1_resample.append([out_array_low])
        out_array_low = np.interp(resample_x,origin_x,low_pi_2[i])
        low_pi2_resample.append([out_array_low])
        out_array_high = np.interp(resample_x,origin_x,high_pi_1[i])
        high_pi1_resample.append([out_array_high])
        out_array_high = np.interp(resample_x,origin_x,high_pi_2[i])
        high_pi2_resample.append([out_array_high])
        out_array_cases = np.interp(resample_x,origin_x,cases_pi[i]/cases_pi[i][-1])
        cases_resample.append([out_array_cases])

    cases_resample = np.array(cases_resample).reshape(len(cases_pi),-1)
    high_pi1_resample = np.array(high_pi1_resample).reshape(len(cases_pi),-1)
    low_pi1_resample = np.array(low_pi1_resample).reshape(len(cases_pi),-1)
    high_pi2_resample = np.array(high_pi2_resample).reshape(len(cases_pi),-1)
    low_pi2_resample = np.array(low_pi2_resample).reshape(len(cases_pi),-1)

    #######################################################################################################

    high_detect_1 = []
    high_detect_2 = []
    low_detect_1 = []
    low_detect_2 = []
    max_length = 0
    max_index = -1
    for i in range(len(cases_detect)):
        cases = cases_detect[i]
        tmp =  len(cases)
        if tmp > max_length:
            max_length = tmp
            max_index = i
        lowest_idx = np.argsort(percentage_list_detect[i])[0]
        low2_idx = np.percentile(np.argsort(percentage_list_detect[i]),25,axis=0).astype(int)
        highest_idx = np.argsort(percentage_list_detect[i])[-1]
        high2_idx = np.percentile(np.argsort(percentage_list_detect[i]),75,axis=0).astype(int)

        low_detect_1.append(fitted_detect[i][lowest_idx]/cases[-1])
        high_detect_1.append(fitted_detect[i][highest_idx]/cases[-1])

        low_detect_2.append(fitted_detect[i][low2_idx]/cases[-1])
        high_detect_2.append(fitted_detect[i][high2_idx]/cases[-1])
    date_axis = gen_date_axis(city_list)
    date_axis = date_axis[city_name_list_detect[max_index]]

    high_detect1_resample = []
    low_detect1_resample = []
    high_detect2_resample = []
    low_detect2_resample = []
    cases_resample = []
    for i in range(len(cases_detect)):
        origin_x = np.linspace(0, len(cases_detect[i]), num=len(cases_detect[i]),endpoint=False)
        resample_x = np.linspace(0, max_length, num=max_length,endpoint=False)
        out_array_low = np.interp(resample_x,origin_x,low_detect_1[i])
        low_detect1_resample.append([out_array_low])
        out_array_low = np.interp(resample_x,origin_x,low_detect_2[i])
        low_detect2_resample.append([out_array_low])
        out_array_high = np.interp(resample_x,origin_x,high_detect_1[i])
        high_detect1_resample.append([out_array_high])
        out_array_high = np.interp(resample_x,origin_x,high_detect_2[i])
        high_detect2_resample.append([out_array_high])
        out_array_cases = np.interp(resample_x,origin_x,cases_detect[i]/cases_detect[i][-1])
        cases_resample.append([out_array_cases])

    cases_resample = np.array(cases_resample).reshape(len(cases_detect),-1)
    high_detect1_resample = np.array(high_detect1_resample).reshape(len(cases_detect),-1)
    low_detect1_resample = np.array(low_detect1_resample).reshape(len(cases_detect),-1)
    high_detect2_resample = np.array(high_detect2_resample).reshape(len(cases_detect),-1)
    low_detect2_resample = np.array(low_detect2_resample).reshape(len(cases_detect),-1)



    #######################################################################################################

    high_mobility_1 = []
    high_mobility_2 = []
    low_mobility_1 = []
    low_mobility_2 = []
    max_length = 0
    max_index = -1
    for i in range(len(cases_mobility)):
        cases = cases_mobility[i]

        tmp =  len(cases)
        if tmp > max_length:
            max_length = tmp
            max_index = i
        lowest_idx = np.argsort(percentage_list_mobility[i])[0]
        low2_idx = np.percentile(np.argsort(percentage_list_mobility[i]),25,axis=0).astype(int)
        highest_idx = np.argsort(percentage_list_mobility[i])[-1]
        high2_idx = np.percentile(np.argsort(percentage_list_mobility[i]),75,axis=0).astype(int)

        low_mobility_1.append(fitted_mobility[i][lowest_idx]/cases[-1])
        high_mobility_1.append(fitted_mobility[i][highest_idx]/cases[-1])

        low_mobility_2.append(fitted_mobility[i][low2_idx]/cases[-1])
        high_mobility_2.append(fitted_mobility[i][high2_idx]/cases[-1])
    date_axis = gen_date_axis(city_list)
    date_axis = date_axis[city_name_list_mobility[max_index]]

    high_mobility1_resample = []
    low_mobility1_resample = []
    high_mobility2_resample = []
    low_mobility2_resample = []
    cases_resample = []
    for i in range(len(cases_mobility)):
        origin_x = np.linspace(0, len(cases_mobility[i]), num=len(cases_mobility[i]),endpoint=False)
        resample_x = np.linspace(0, max_length, num=max_length,endpoint=False)
        out_array_low = np.interp(resample_x,origin_x,low_mobility_1[i])
        low_mobility1_resample.append([out_array_low])
        out_array_low = np.interp(resample_x,origin_x,low_mobility_2[i])
        low_mobility2_resample.append([out_array_low])
        out_array_high = np.interp(resample_x,origin_x,high_mobility_1[i])
        high_mobility1_resample.append([out_array_high])
        out_array_high = np.interp(resample_x,origin_x,high_mobility_2[i])
        high_mobility2_resample.append([out_array_high])
        out_array_cases = np.interp(resample_x,origin_x,cases_mobility[i]/cases_mobility[i][-1])
        cases_resample.append([out_array_cases])

    cases_resample = np.array(cases_resample).reshape(len(cases_mobility),-1)
    high_mobility1_resample = np.array(high_mobility1_resample).reshape(len(cases_mobility),-1)
    low_mobility1_resample = np.array(low_mobility1_resample).reshape(len(cases_mobility),-1)
    high_mobility2_resample = np.array(high_mobility2_resample).reshape(len(cases_mobility),-1)
    low_mobility2_resample = np.array(low_mobility2_resample).reshape(len(cases_mobility),-1)

    #######################################################################################################
    high_date_1 = []
    high_date_2 = []
    low_date_1 = []
    low_date_2 = []
    max_length = 0
    max_index = -1
    for i in range(len(cases_date)):
        cases = cases_date[i]
        tmp =  len(cases)
        if tmp > max_length:
            max_length = tmp
            max_index = i
        lowest_idx = np.argsort(percentage_list_date[i])[0]
        low2_idx = np.percentile(np.argsort(percentage_list_date[i]),25,axis=0).astype(int)
        highest_idx = np.argsort(percentage_list_date[i])[-1]
        high2_idx = np.percentile(np.argsort(percentage_list_date[i]),75,axis=0).astype(int)

        low_date_1.append(fitted_date[i][lowest_idx]/cases[-1])
        high_date_1.append(fitted_date[i][highest_idx]/cases[-1])

        low_date_2.append(fitted_date[i][low2_idx]/cases[-1])
        high_date_2.append(fitted_date[i][high2_idx]/cases[-1])
    date_axis = gen_date_axis(city_list)
    date_axis = date_axis[city_name_list_date[max_index]]

    high_date1_resample = []
    low_date1_resample = []
    high_date2_resample = []
    low_date2_resample = []
    cases_resample = []
    for i in range(len(cases_date)):
        origin_x = np.linspace(0, len(cases_date[i]), num=len(cases_date[i]),endpoint=False)
        resample_x = np.linspace(0, max_length, num=max_length,endpoint=False)
        out_array_low = np.interp(resample_x,origin_x,low_date_1[i])
        low_date1_resample.append([out_array_low])
        out_array_low = np.interp(resample_x,origin_x,low_date_2[i])
        low_date2_resample.append([out_array_low])
        out_array_high = np.interp(resample_x,origin_x,high_date_1[i])
        high_date1_resample.append([out_array_high])
        out_array_high = np.interp(resample_x,origin_x,high_date_2[i])
        high_date2_resample.append([out_array_high])
        out_array_cases = np.interp(resample_x,origin_x,cases_date[i]/cases_date[i][-1])
        cases_resample.append([out_array_cases])

    cases_resample = np.array(cases_resample).reshape(len(cases_date),-1)
    high_date1_resample = np.array(high_date1_resample).reshape(len(cases_date),-1)
    low_date1_resample = np.array(low_date1_resample).reshape(len(cases_date),-1)
    high_date2_resample = np.array(high_date2_resample).reshape(len(cases_date),-1)
    low_date2_resample = np.array(low_date2_resample).reshape(len(cases_date),-1)


    fig, axes  = plt.subplots(ncols=2, nrows=2, figsize=(16,9))


    axes[0,1].plot(date_axis, np.median(cases_resample,axis=0),linewidth=2,color = '#4983EB',label= 'Real-world Scenario')
    axes[0,1].fill_between(date_axis, 
                    np.percentile(cases_resample,25,axis=0),
                    np.percentile(cases_resample,75,axis=0), alpha=.2, color='#abc6e4')


    axes[0,1].plot(date_axis,np.median(high_pi1_resample,axis=0),linewidth=2,color = 'r',label= '10% Infection Rate Increase')
    axes[0,1].fill_between(date_axis, 
                    np.percentile(high_pi1_resample,25,axis=0),
                    np.percentile(high_pi1_resample,75,axis=0), alpha=.2, color='r')


    axes[0,1].plot(date_axis,np.median(high_pi2_resample,axis=0),linewidth=2,color = 'r',linestyle='--',label= '5% Infection Rate Increase')
    axes[0,1].fill_between(date_axis, 
                    np.percentile(high_pi2_resample,25,axis=0),
                    np.percentile(high_pi2_resample,75,axis=0), alpha=.1, color='r')

    axes[0,1].plot(date_axis,np.percentile(high_pi2_resample,25,axis=0),linewidth=1,color = '#fed1d1',linestyle='--')
    axes[0,1].plot(date_axis,np.percentile(high_pi2_resample,75,axis=0),linewidth=1,color = '#fed1d1',linestyle='--')



    axes[0,1].plot(date_axis,np.median(low_pi1_resample,axis=0),linewidth=2,color = 'g',label= '10% Infection Rate Decrease')
    axes[0,1].fill_between(date_axis, 
                    np.percentile(low_pi1_resample,25,axis=0),
                    np.percentile(low_pi1_resample,75,axis=0), alpha=.2, color='g')

    axes[0,1].plot(date_axis,np.median(low_pi2_resample,axis=0),linewidth=2,color = 'g',linestyle='--',label= '5% Infection Rate Decrease')
    axes[0,1].fill_between(date_axis, 
                    np.percentile(low_pi2_resample,25,axis=0),
                    np.percentile(low_pi2_resample,75,axis=0), alpha=.1, color='g')

    axes[0,1].plot(date_axis,np.percentile(low_pi2_resample,25,axis=0),linewidth=1,color = '#dcf3dc',linestyle='--')
    axes[0,1].plot(date_axis,np.percentile(low_pi2_resample,75,axis=0),linewidth=1,color = '#dcf3dc',linestyle='--')
    axes[0,1].set_ylabel("Normalized Infection",size=20)
    axes[0,1].legend(fontsize=14,loc='upper left')

    axes[0,1].tick_params(axis='both', labelsize=16)
    axes[0,1].grid(which='major',ls='--', alpha=0.8)


    axes[1,0].plot(date_axis, np.median(cases_resample,axis=0),linewidth=2,color = '#4983EB',label= 'Real-world Scenario')
    axes[1,0].fill_between(date_axis, 
                    np.percentile(cases_resample,25,axis=0),
                    np.percentile(cases_resample,75,axis=0), alpha=.2, color='#abc6e4')
    axes[1,0].plot(date_axis,np.median(high_detect1_resample,axis=0),linewidth=2,color = 'g',label= '10% Quarantine Rate Increase')
    axes[1,0].fill_between(date_axis, 
                    np.percentile(high_detect1_resample,25,axis=0),
                    np.percentile(high_detect1_resample,75,axis=0), alpha=.2, color='g')
    axes[1,0].plot(date_axis,np.median(high_detect2_resample,axis=0),linewidth=2,color = 'g',linestyle='--',label= '5% Quarantine Rate Increase')
    axes[1,0].fill_between(date_axis, 
                    np.percentile(high_detect2_resample,25,axis=0),
                    np.percentile(high_detect2_resample,75,axis=0), alpha=.1, color='g')
    axes[1,0].plot(date_axis,np.percentile(high_detect2_resample,25,axis=0),linewidth=1,color = '#dcf3dc',linestyle='--')
    axes[1,0].plot(date_axis,np.percentile(high_detect2_resample,75,axis=0),linewidth=1,color = '#dcf3dc',linestyle='--')

    axes[1,0].plot(date_axis,np.median(low_detect1_resample,axis=0),linewidth=2,color = 'r',label= '10% Quarantine Rate Decrease')
    axes[1,0].fill_between(date_axis, 
                    np.percentile(low_detect1_resample,25,axis=0),
                    np.percentile(low_detect1_resample,75,axis=0), alpha=.2, color='r')
    axes[1,0].plot(date_axis,np.median(low_detect2_resample,axis=0),linewidth=2,color = 'r',linestyle='--',label= '5% Quarantine Rate Decrease')
    axes[1,0].fill_between(date_axis, 
                    np.percentile(low_detect2_resample,25,axis=0),
                    np.percentile(low_detect2_resample,75,axis=0), alpha=.1, color='r')
    axes[1,0].plot(date_axis,np.percentile(low_detect2_resample,25,axis=0),linewidth=1,color = '#fed1d1',linestyle='--')
    axes[1,0].plot(date_axis,np.percentile(low_detect2_resample,75,axis=0),linewidth=1,color = '#fed1d1',linestyle='--')
    axes[1,0].set_ylabel("Normalized Infection",size=20)
    axes[1,0].legend(fontsize=14,loc='upper left')
    axes[1,0].tick_params(axis='both', labelsize=16)
    axes[1,0].grid(which='major',ls='--', alpha=0.8)

    axes[0,0].plot(date_axis, np.median(cases_resample,axis=0),linewidth=2,color = '#4983EB',label= 'Real-world Scenario')
    axes[0,0].fill_between(date_axis, 
                    np.percentile(cases_resample,25,axis=0),
                    np.percentile(cases_resample,75,axis=0), alpha=.2, color='#abc6e4')
    axes[0,0].plot(date_axis,np.median(high_mobility1_resample,axis=0),linewidth=2,color = 'r',label= '10% Mobility Activity Increase')
    axes[0,0].fill_between(date_axis, 
                    np.percentile(high_mobility1_resample,25,axis=0),
                    np.percentile(high_mobility1_resample,75,axis=0), alpha=.2, color='r')
    axes[0,0].plot(date_axis,np.median(high_mobility2_resample,axis=0),linewidth=2,color = 'r',linestyle='--',label= '5% Mobility Activity Increase')
    axes[0,0].fill_between(date_axis, 
                    np.percentile(high_mobility2_resample,25,axis=0),
                    np.percentile(high_mobility2_resample,75,axis=0), alpha=.1, color='r')
    axes[0,0].plot(date_axis,np.percentile(high_mobility2_resample,25,axis=0),linewidth=1,color = '#fed1d1',linestyle='--')
    axes[0,0].plot(date_axis,np.percentile(high_mobility2_resample,75,axis=0),linewidth=1,color = '#fed1d1',linestyle='--')
    axes[0,0].plot(date_axis,np.median(low_mobility1_resample,axis=0),linewidth=2,color = 'g',label= '10% Mobility Activity Decrease')
    axes[0,0].fill_between(date_axis, 
                    np.percentile(low_mobility1_resample,25,axis=0),
                    np.percentile(low_mobility1_resample,75,axis=0), alpha=.2, color='g')
    axes[0,0].plot(date_axis,np.median(low_mobility2_resample,axis=0),linewidth=2,color = 'g',linestyle='--',label= '5% Mobility Activity Decrease')
    axes[0,0].fill_between(date_axis, 
                    np.percentile(low_mobility2_resample,25,axis=0),
                    np.percentile(low_mobility2_resample,75,axis=0), alpha=.1, color='g')
    axes[0,0].plot(date_axis,np.percentile(low_mobility2_resample,25,axis=0),linewidth=1,color = '#dcf3dc',linestyle='--')
    axes[0,0].plot(date_axis,np.percentile(low_mobility2_resample,75,axis=0),linewidth=1,color = '#dcf3dc',linestyle='--')
    axes[0,0].set_ylim(-0.1,5)
    axes[0,0].set_ylabel("Normalized Infection",size=20)
    axes[0,0].tick_params(axis='both', labelsize=16)
    axes[0,0].legend(fontsize=14,loc='upper left')
    axes[0,0].grid(which='major',ls='--', alpha=0.8)


    axes[1,1].plot(date_axis, np.median(cases_resample,axis=0),linewidth=2,color = '#4983EB',label= 'Real-world Scenario')
    axes[1,1].fill_between(date_axis, 
                    np.percentile(cases_resample,25,axis=0),
                    np.percentile(cases_resample,75,axis=0), alpha=.2, color='#abc6e4')
    axes[1,1].plot(date_axis,np.median(high_date1_resample,axis=0),linewidth=2,color = 'r',label= '10 Days Late Intervention')
    axes[1,1].fill_between(date_axis, 
                    np.percentile(high_date1_resample,25,axis=0),
                    np.percentile(high_date1_resample,75,axis=0), alpha=.2, color='r')
    axes[1,1].plot(date_axis,np.median(high_date2_resample,axis=0),linewidth=2,color = 'r',linestyle='--',label= '5 Days Late Intervention')
    axes[1,1].fill_between(date_axis, 
                    np.percentile(high_date2_resample,25,axis=0),
                    np.percentile(high_date2_resample,75,axis=0), alpha=.1, color='r')
    axes[1,1].plot(date_axis,np.percentile(high_date2_resample,25,axis=0),linewidth=1,color = '#fed1d1',linestyle='--')
    axes[1,1].plot(date_axis,np.percentile(high_date2_resample,75,axis=0),linewidth=1,color = '#fed1d1',linestyle='--')
    axes[1,1].plot(date_axis,np.median(low_date1_resample,axis=0),linewidth=2,color = 'g',label= '10 Days Early Intervention')
    axes[1,1].fill_between(date_axis, 
                    np.percentile(low_date1_resample,25,axis=0),
                    np.percentile(low_date1_resample,75,axis=0), alpha=.2, color='g')
    axes[1,1].plot(date_axis,np.median(low_date2_resample,axis=0),linewidth=2,color = 'g',linestyle='--',label= '5 Days Early Intervention')
    axes[1,1].fill_between(date_axis, 
                    np.percentile(low_date2_resample,25,axis=0),
                    np.percentile(low_date2_resample,75,axis=0), alpha=.1, color='g')
    axes[1,1].plot(date_axis,np.percentile(low_date2_resample,25,axis=0),linewidth=1,color = '#dcf3dc',linestyle='--')
    axes[1,1].plot(date_axis,np.percentile(low_date2_resample,75,axis=0),linewidth=1,color = '#dcf3dc',linestyle='--')
    axes[1,1].axvline(x=date_axis[city_list[city_name_list_date[max_index]]['second'] - city_list[city_name_list_date[max_index]]['first'] - 1], color='#4983EB',  linewidth=2)
    axes[1,1].axvline(x=date_axis[city_list[city_name_list_date[max_index]]['second'] - city_list[city_name_list_date[max_index]]['first'] - 1 - 10], color='g', linewidth=2)
    axes[1,1].axvline(x=date_axis[city_list[city_name_list_date[max_index]]['second'] - city_list[city_name_list_date[max_index]]['first'] - 1 + 10], color='r', linewidth=2)

    axes[1,1].axvline(x=date_axis[city_list[city_name_list_date[max_index]]['second'] - city_list[city_name_list_date[max_index]]['first'] - 1 - 5], color='g', linestyle = '--', linewidth=2)
    axes[1,1].axvline(x=date_axis[city_list[city_name_list_date[max_index]]['second'] - city_list[city_name_list_date[max_index]]['first'] - 1 + 5], color='r', linestyle = '--', linewidth=2)

    axes[1,1].set_ylabel("Normalized Infection",size=20)
    axes[1,1].legend(fontsize=14,loc='upper left')
    axes[1,1].tick_params(axis='both', labelsize=16)
    axes[1,1].grid(which='major',ls='--', alpha=0.8)
    axes[1,1].set_ylim(-0.3,6)
    axes[0,0].set_xlabel("Day",size=20)
    axes[0,1].set_xlabel("Day",size=20)
    axes[1,0].set_xlabel("Day",size=20)
    axes[1,1].set_xlabel("Day",size=20)
    plt.show()





if __name__ == "__main__":
    draw_all_range(200)




