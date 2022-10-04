import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
import json
import pandas as pd
from math import floor
from sklearn.metrics import r2_score
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
from scipy.stats import pearsonr,spearmanr
import datetime
import matplotlib.dates as mdates

def ma_all(a,n=5):
    ma = []
    length = a.shape[0]
    for i in range(length):
        if i < n/2:
            ma.append(a[:i+int(n/2)+1].mean())
        elif (length - i) < n/2:
            ma.append(a[i-int((n-1)/2):].mean())
        else:
            ma.append(a[i-int((n-1)/2):i+int((n-1)/2+1)].mean())
    return np.array(ma)

def setup_args(parser=None):
    """ Set up arguments

    return:
        python dictionary
    """
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument('--ignore_range', action='store_true', default=True, help='do not plot range')
    parser.add_argument('--draw_cumu', action='store_true', default=True, help='draw cumulative results')
    parser.add_argument('--std', type=int, default=3, help='n times of standard')

    return parser


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


    Bengaluru = {'start':'2021-03-12', 'change':'2021-05-01', 'final':'2021-07-31', 'city_name':'Bengaluru Urban','sample_rate':4}
    Delhi = {'start':'2021-03-22', 'change':'2021-05-04', 'final':'2021-07-31', 'city_name':'Delhi','sample_rate':4}
    Mumbai = {'start':'2021-02-11', 'change':'2021-04-17', 'final':'2021-07-31', 'city_name':'Mumbai','sample_rate':4}
    Pune = {'start':'2021-02-14', 'change':'2021-05-14', 'final':'2021-07-31', 'city_name':'Pune','sample_rate':4}
    Thane = {'start':'2021-02-24', 'change':'2021-05-01', 'final':'2021-07-31', 'city_name':'Thane','sample_rate':4}
    India = {'Bengaluru':Bengaluru,'Delhi':Delhi, 'Mumbai':Mumbai, 'Pune':Pune, 'Thane':Thane}

    BeloHorizonte = {'start':'2021-01-01', 'change':'2021-04-27', 'final':'2021-07-31', 'city_name':'Belo Horizonte','sample_rate':4}
    Brasilia = {'start':'2021-01-01', 'change':'2021-02-23', 'final':'2021-07-31', 'city_name':'Bras_­lia','sample_rate':4}
    Fortaleza = {'start':'2021-01-01', 'change':'2021-04-01', 'final':'2021-07-31', 'city_name':'Fortaleza','sample_rate':4}
    RioDeJaneiro = {'start':'2021-01-01', 'change':'2021-04-26', 'final':'2021-07-31', 'city_name':'Rio de Janeiro','sample_rate':4}
    SaoPaulo = {'start':'2021-01-01', 'change':'2021-03-10', 'final':'2021-07-31', 'city_name':'S_£o Paulo','sample_rate':4}
    Brazil = {'BeloHorizonte':BeloHorizonte, 'Brasilia':Brasilia, 'Fortaleza':Fortaleza, 'RioDeJaneiro':RioDeJaneiro, 'SaoPaulo':SaoPaulo}
    

    city_list = {'Bergen':Bergen,'MiamiDade':Miami_Dade,'Nassau':Nassau,'Oakland':Oakland,'Suffolk':Suffolk,'Union':Union, \
                 'Cook': Cook,'Davidson':Davidson, 'Los_Angeles':Los_Angeles, 'Middlesex':Middlesex, \
                 'Harris':Harris, 'New_Orleans':New_Orleans, 'Will':Will, \
                 'Hillsborough': Hillsborough,'Hudson':Hudson,'King':King,'New_Haven':New_Haven,'NY':NewYorkCity,'Ocean':Ocean,'Philadelphia':Philadelphia,
                 }

    return city_list, India, Brazil

def load_cases_Ind(data_dir, name):
    data = pd.read_csv(data_dir)
    cases = data[data['District']==name]
    return cases

def load_cases_Bra(data_dir, name):
    data = pd.read_csv(data_dir)
    cases = data[data['Município']==name]
    return cases



def load_cases_and_fitted_all_ours(topk=10):
    US_city_list, India, Brazil = init_city_list()
    cases_us = []
    fitted_us = []
    name_list_us = []
    for key,values in US_city_list.items():
        name_list_us.append(key)
        case = np.load('./cases/'+values['cases_name'])[values['first']:]
        cases_us.append(case)
        fit_input_dir = Path('./results/fig1a_fit/our_model')
        npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/'+'*curve*.npy')])
        fitted_data = np.array([np.load(i).reshape(-1) for i in npy_datas])
        loss = np.square(fitted_data - case).sum(axis=1)
        fitted_data_topk = fitted_data[np.argsort(loss)[:topk], :]
        fitted_us.append(fitted_data_topk)

    cases_india = []
    fitted_india = []
    name_list_india = []
    for key,values in India.items():
        name_list_india.append(key)
        cases_data = load_cases_Ind('./cases/India_epidemic_district_timeline.csv', values['city_name'])

        start_index = np.where(cases_data.columns == values['start'])[0]
        change_index = np.where(cases_data.columns == values['change'])[0]
        final_index = np.where(cases_data.columns == values['final'])[0]

        origin_x = np.linspace(0, cases_data.shape[1]-1, num=cases_data.shape[1]-1, endpoint=False)
        num_new_points = int((cases_data.shape[1]-1)/values['sample_rate'])
        resample_x = np.linspace(0, cases_data.shape[1]-1, num=num_new_points, endpoint=False)
        cases_resample = np.interp(x=resample_x, xp=origin_x, fp=cases_data.iloc[:,1:].to_numpy().reshape(-1))
        new_start_index = int(start_index / values['sample_rate'])
        new_change_index = int(change_index / values['sample_rate'])
        new_final_index = int(final_index / values['sample_rate'])
        cases_data_processed = cases_resample[new_start_index:new_final_index]

        cases_india.append(cases_data_processed)
        fit_input_dir = Path('./results/fig1a_fit/our_model')
        npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/'+'*curve*.npy')])
        fitted_data = np.array([np.load(i).reshape(-1) for i in npy_datas])
        loss = np.square(fitted_data - cases_data_processed).sum(axis=1)
        fitted_data_topk = fitted_data[np.argsort(loss)[:topk], :]
        fitted_india.append(fitted_data_topk)

    cases_bra = []
    fitted_bra = []
    name_list_bra = []
    for key,values in Brazil.items():
        name_list_bra.append(key)
        cases_data = load_cases_Bra('./cases/Brazil_epidemic_district_timeline.csv', values['city_name'])

        start_index = np.where(cases_data.columns == values['start'])[0]
        change_index = np.where(cases_data.columns == values['change'])[0]
        final_index = np.where(cases_data.columns == values['final'])[0]

        origin_x = np.linspace(0, cases_data.shape[1]-1, num=cases_data.shape[1]-1, endpoint=False)
        num_new_points = int((cases_data.shape[1]-1)/values['sample_rate'])
        resample_x = np.linspace(0, cases_data.shape[1]-1, num=num_new_points, endpoint=False)
        cases_resample = np.interp(x=resample_x, xp=origin_x, fp=cases_data.iloc[:,1:].to_numpy().reshape(-1))
        new_start_index = int(start_index / values['sample_rate'])
        new_change_index = int(change_index / values['sample_rate'])
        new_final_index = int(final_index / values['sample_rate'])
        cases_data_processed = cases_resample[new_start_index:new_final_index]

        cases_bra.append(cases_data_processed)
        fit_input_dir = Path('./results/fig1a_fit/our_model')
        npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/'+'*curve*.npy')])
        fitted_data = np.array([np.load(i).reshape(-1) for i in npy_datas])
        loss = np.square(fitted_data - cases_data_processed).sum(axis=1)
        fitted_data_topk = fitted_data[np.argsort(loss)[:topk], :]
        fitted_bra.append(fitted_data_topk)


    return cases_us, fitted_us, name_list_us, cases_india, fitted_india, name_list_india, cases_bra, fitted_bra, name_list_bra



def load_cases_and_fitted_all_seir(topk=10):
    US_city_list, India, Brazil = init_city_list()
    cases_us = []
    fitted_us = []
    name_list_us = []
    for key,values in US_city_list.items():
        name_list_us.append(key)
        case = np.load('./cases/'+values['cases_name'])[values['first']:]
        cases_us.append(case)
        fit_input_dir = Path('./results/fig1a_fit/traditional_seir')
        npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/'+'*curve*.npy')])
        fitted_data = np.array([np.load(i).reshape(-1) for i in npy_datas])
        loss = np.square(fitted_data - case).sum(axis=1)
        fitted_data_topk = fitted_data[np.argsort(loss)[:topk], :]
        fitted_us.append(fitted_data_topk)


    cases_india = []
    fitted_india = []
    name_list_india = []
    for key,values in India.items():
        name_list_india.append(key)
        cases_data = load_cases_Ind('./cases/India_epidemic_district_timeline.csv', values['city_name'])

        start_index = np.where(cases_data.columns == values['start'])[0]
        change_index = np.where(cases_data.columns == values['change'])[0]
        final_index = np.where(cases_data.columns == values['final'])[0]

        origin_x = np.linspace(0, cases_data.shape[1]-1, num=cases_data.shape[1]-1, endpoint=False)
        num_new_points = int((cases_data.shape[1]-1)/values['sample_rate'])
        resample_x = np.linspace(0, cases_data.shape[1]-1, num=num_new_points, endpoint=False)
        cases_resample = np.interp(x=resample_x, xp=origin_x, fp=cases_data.iloc[:,1:].to_numpy().reshape(-1))
        new_start_index = int(start_index / values['sample_rate'])
        new_change_index = int(change_index / values['sample_rate'])
        new_final_index = int(final_index / values['sample_rate'])
        cases_data_processed = cases_resample[new_start_index:new_final_index]

        cases_india.append(cases_data_processed)
        fit_input_dir = Path('./results/fig1a_fit/traditional_seir')
        npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/'+'*curve*.npy')])
        fitted_data = np.array([np.load(i).reshape(-1) for i in npy_datas])
        loss = np.square(fitted_data - cases_data_processed).sum(axis=1)
        fitted_data_topk = fitted_data[np.argsort(loss)[:topk], :]
        fitted_india.append(fitted_data_topk)

    cases_bra = []
    fitted_bra = []
    name_list_bra = []
    for key,values in Brazil.items():
        name_list_bra.append(key)
        cases_data = load_cases_Bra('./cases/Brazil_epidemic_district_timeline.csv', values['city_name'])

        start_index = np.where(cases_data.columns == values['start'])[0]
        change_index = np.where(cases_data.columns == values['change'])[0]
        final_index = np.where(cases_data.columns == values['final'])[0]

        origin_x = np.linspace(0, cases_data.shape[1]-1, num=cases_data.shape[1]-1, endpoint=False)
        num_new_points = int((cases_data.shape[1]-1)/values['sample_rate'])
        resample_x = np.linspace(0, cases_data.shape[1]-1, num=num_new_points, endpoint=False)
        cases_resample = np.interp(x=resample_x, xp=origin_x, fp=cases_data.iloc[:,1:].to_numpy().reshape(-1))
        new_start_index = int(start_index / values['sample_rate'])
        new_change_index = int(change_index / values['sample_rate'])
        new_final_index = int(final_index / values['sample_rate'])
        cases_data_processed = cases_resample[new_start_index:new_final_index]

        cases_bra.append(cases_data_processed)
        fit_input_dir = Path('./results/fig1a_fit/traditional_seir')
        npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/'+'*curve*.npy')])
        fitted_data = np.array([np.load(i).reshape(-1) for i in npy_datas])
        loss = np.square(fitted_data - cases_data_processed).sum(axis=1)
        fitted_data_topk = fitted_data[np.argsort(loss)[:topk], :]
        fitted_bra.append(fitted_data_topk)




    return cases_us, fitted_us, name_list_us, cases_india, fitted_india, name_list_india, cases_bra, fitted_bra, name_list_bra



def gen_date_axis_India_Bra(city_list):
    city_name = []
    date_axis = []
    for key,values in city_list.items():
        city_name.append(key)
        begin_date = datetime.datetime.strptime(values['start'], "%Y-%m-%d")
        end_date = datetime.datetime.strptime(values['final'], "%Y-%m-%d")
        data = [i for i in range((end_date - begin_date).days + 1)]

        origin_x = np.linspace(0, len(data) , num=len(data), endpoint=False)
        num_new_points = int((len(data)) / values['sample_rate'])
        resample_x = np.linspace(0, len(data), num=num_new_points, endpoint=False)
        data_resample = np.interp(x=resample_x, xp=origin_x, fp=data)

        date_axis.append(data_resample)
    date_axis_dict = dict()
    for i in range(len(city_name)):
        date_axis_dict[city_name[i]] = date_axis[i]

    return date_axis_dict


def gen_date_axis(city_list):
    city_name = []
    date_axis = []
    begin_date = datetime.datetime.strptime('2020/1/22', "%Y/%m/%d")

    for key,values in city_list.items():
        city_name.append(key)
        date_axis.append([i-values['first'] for i in range(values['first'],99+1)])
    date_axis_dict = dict()
    for i in range(len(city_name)):
        date_axis_dict[city_name[i]] = date_axis[i]

    return date_axis_dict

def draw(opt):
    city_list, India, Brazil = init_city_list()
    date_axis_us = gen_date_axis(city_list)
    date_axis_india = gen_date_axis_India_Bra(India)
    date_axis_brazil = gen_date_axis_India_Bra(Brazil)

    cases_us, fitted_us, city_name_list_us, cases_india, fitted_india, city_name_list_india, cases_brazil, fitted_brazil, city_name_list_brazil = load_cases_and_fitted_all_ours()
    _, fitted_data_us_seir, _ , _, fitted_data_india_seir, _, _, fitted_data_brazil_seir, _ = load_cases_and_fitted_all_seir()

    fig, axes = plt.subplots(ncols=5, nrows=6, figsize=(15,14))
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
    plt.tight_layout()

    total_R2 = []
    # For US cities
    total_R2_seir = []
    if opt['draw_cumu'] == True:
        for i in range(len(city_name_list_us)):

            tmp_x = city_list[city_name_list_us[i]]['second'] - city_list[city_name_list_us[i]]['first'] - 1
            tmp = date_axis_us[city_name_list_us[i]][tmp_x]

            # Our results
            axes[int(floor(i/5)),i%5].plot(date_axis_us[city_name_list_us[i]], fitted_us[i].mean(axis=0), label='Estimated Cases', color='blue')
            axes[int(floor(i/5)),i%5].fill_between(date_axis_us[city_name_list_us[i]],
                            fitted_us[i].mean(axis=0) + 3*fitted_us[i].std(axis=0),
                            fitted_us[i].mean(axis=0) - 3*fitted_us[i].std(axis=0), alpha=.75, color='#abc6e4')


            axes[int(floor(i/5)),i%5].scatter(tmp, fitted_us[i].mean(axis=0)[tmp_x], c='k',marker='d', s = 40, label='Changing Point')

            # Classic SEIR
            axes[int(floor(i/5)),i%5].plot(date_axis_us[city_name_list_us[i]], fitted_data_us_seir[i].mean(axis=0), label='Traditional SEIR', color='g')
            if not opt['ignore_range']:
                axes[int(floor(i/5)),i%5].fill_between(date_axis_us[city_name_list_us[i]],
                            fitted_data_us_seir[i].mean(axis=0) + 3*fitted_data_us_seir[i].std(axis=0),
                            fitted_data_us_seir[i].mean(axis=0) - 3*fitted_data_us_seir[i].std(axis=0), alpha=.3, color='g')
            axes[int(floor(i/5)),i%5].scatter(tmp, fitted_data_us_seir[i].mean(axis=0)[tmp_x], c='k',marker='d', s = 40, label='Changing Point')

            # Real Cases
            axes[int(floor(i/5)),i%5].scatter(date_axis_us[city_name_list_us[i]], cases_us[i], c='r',marker='.', s=20, label='Reported Cases')


            r2_our = r2_score(cases_us[i],fitted_us[i].mean(axis=0))
            r2_seir = r2_score(cases_us[i],fitted_data_us_seir[i].mean(axis=0))
            r2_our = 1/(2-r2_our)
            r2_seir = 1/(2-r2_seir)
            xlim = axes[int(floor(i/5)),i%5].get_xlim()
            ylim = axes[int(floor(i/5)),i%5].get_ylim()
            axes[int(floor(i/5)),i%5].text((xlim[1]-xlim[0]) / 8 + xlim[0], (ylim[1]-ylim[0]) / 8 *7 + ylim[0],'$R^2$ score:'+"%(r2).3f"%{'r2':r2_our},fontdict = {'size': 12, 'color': 'blue'})
            axes[int(floor(i/5)),i%5].text((xlim[1]-xlim[0]) / 8 + xlim[0], (ylim[1]-ylim[0]) / 8 *6 + ylim[0],'$R^2$ score:'+"%(r2).3f"%{'r2':r2_seir},fontdict = {'size': 12, 'color': 'green'})
            axes[int(floor(i/5)),i%5].grid(which='major',ls='--', alpha=0.8)
            
            if city_name_list_us[i] == 'NY':
                axes[int(floor(i/5)),i%5].set_title('New York City')
            elif city_name_list_us[i] == 'New_Orleans':
                axes[int(floor(i/5)),i%5].set_title('New Orleans')
            elif city_name_list_us[i] == 'New_Haven':
                axes[int(floor(i/5)),i%5].set_title('New Haven')
            elif city_name_list_us[i] == 'Los_Angeles':
                axes[int(floor(i/5)),i%5].set_title('Los Angeles')
            else:
                axes[int(floor(i/5)),i%5].set_title(city_name_list_us[i])

            total_R2.append(r2_our)
            total_R2_seir.append(r2_seir)
        
        # India City
        for i in range(len(city_name_list_india)):
            # Our results
            axes[4,i%5].plot(date_axis_india[city_name_list_india[i]], fitted_india[i].mean(axis=0), label='Estimated Cases', color='blue')
            axes[4,i%5].fill_between(date_axis_india[city_name_list_india[i]],
                            fitted_india[i].mean(axis=0) + 3*fitted_india[i].std(axis=0),
                            fitted_india[i].mean(axis=0) - 3*fitted_india[i].std(axis=0), alpha=.75, color='#abc6e4')

            tmp_x = int((datetime.datetime.strptime(India[city_name_list_india[i]]['change'], "%Y-%m-%d") -  datetime.datetime.strptime(India[city_name_list_india[i]]['start'], "%Y-%m-%d")).days / India[city_name_list_india[i]]['sample_rate']) + 1
            tmp = date_axis_india[city_name_list_india[i]][tmp_x]
            axes[4,i%5].scatter(tmp, fitted_india[i].mean(axis=0)[tmp_x], c='k',marker='d', s = 40, label='Changing Point')

            # Classic SEIR
            axes[4,i%5].plot(date_axis_india[city_name_list_india[i]], fitted_data_india_seir[i].mean(axis=0), label='Traditional SEIR', color='g')
            if not opt['ignore_range']:
                axes[4,i%5].fill_between(date_axis_india[city_name_list_india[i]],
                                fitted_data_india_seir[i].mean(axis=0) + 3*fitted_data_india_seir[i].std(axis=0),
                                fitted_data_india_seir[i].mean(axis=0) - 3*fitted_data_india_seir[i].std(axis=0), alpha=.3, color='g')
            axes[4,i%5].scatter(tmp, fitted_data_india_seir[i].mean(axis=0)[tmp_x], c='k',marker='d', s = 40, label='Changing Point')

            # Real Cases
            axes[4,i%5].scatter(date_axis_india[city_name_list_india[i]], cases_india[i], c='r',marker='.', s=20, label='Reported Cases')


            r2_our = r2_score(cases_india[i],fitted_india[i].mean(axis=0))
            r2_seir = r2_score(cases_india[i],fitted_data_india_seir[i].mean(axis=0))
            r2_our = 1/(2-r2_our)
            r2_seir = 1/(2-r2_seir)
            xlim = axes[4,i%5].get_xlim()
            ylim = axes[4,i%5].get_ylim()
            axes[4,i%5].text((xlim[1]-xlim[0]) / 8 * 3.5 + xlim[0], (ylim[1]-ylim[0]) / 8 *2 + ylim[0],'$R^2$ score:'+"%(r2).3f"%{'r2':r2_our},fontdict = {'size': 12, 'color': 'blue'})
            axes[4,i%5].text((xlim[1]-xlim[0]) / 8 * 3.5 + xlim[0], (ylim[1]-ylim[0]) / 8 *1 + ylim[0],'$R^2$ score:'+"%(r2).3f"%{'r2':r2_seir},fontdict = {'size': 12, 'color': 'green'})
            axes[4,i%5].grid(which='major',ls='--', alpha=0.8)
            axes[4,i%5].ticklabel_format(axis='y',style='sci',scilimits=(0,0))

            axes[4,i%5].set_title(city_name_list_india[i])
            total_R2.append(r2_our)
            total_R2_seir.append(r2_seir)

        # Brazil Cities
        for i in range(len(city_name_list_brazil)):

            # Our Results
            axes[5,i%5].plot(date_axis_brazil[city_name_list_brazil[i]], fitted_brazil[i].mean(axis=0), label='Estimated Cases', color='blue')
            axes[5,i%5].fill_between(date_axis_brazil[city_name_list_brazil[i]],
                            fitted_brazil[i].mean(axis=0) + 3*fitted_brazil[i].std(axis=0),
                            fitted_brazil[i].mean(axis=0) - 3*fitted_brazil[i].std(axis=0), alpha=.75, color='#abc6e5')

            tmp_x = int((datetime.datetime.strptime(Brazil[city_name_list_brazil[i]]['change'], "%Y-%m-%d") -  datetime.datetime.strptime(Brazil[city_name_list_brazil[i]]['start'], "%Y-%m-%d")).days / Brazil[city_name_list_brazil[i]]['sample_rate']) + 1
            tmp = date_axis_brazil[city_name_list_brazil[i]][tmp_x]
            axes[5,i%5].scatter(tmp, fitted_brazil[i].mean(axis=0)[tmp_x], c='k',marker='d', s = 50, label='Changing Point')

            # Classic SEIR
            axes[5,i%5].plot(date_axis_brazil[city_name_list_brazil[i]], fitted_data_brazil_seir[i].mean(axis=0), label='Traditional SEIR', color='g')
            if not opt['ignore_range']:
                axes[5,i%5].fill_between(date_axis_brazil[city_name_list_brazil[i]],
                                fitted_data_brazil_seir[i].mean(axis=0) + 3*fitted_data_brazil_seir[i].std(axis=0),
                                fitted_data_brazil_seir[i].mean(axis=0) - 3*fitted_data_brazil_seir[i].std(axis=0), alpha=.3, color='g')
            axes[5,i%5].scatter(tmp, fitted_data_brazil_seir[i].mean(axis=0)[tmp_x], c='k',marker='d', s = 50, label='Changing Point')

            # Real Cases
            axes[5,i%5].scatter(date_axis_brazil[city_name_list_brazil[i]], cases_brazil[i], c='r',marker='.', s=20, label='Reported Cases')

            r2_our = r2_score(cases_brazil[i],fitted_brazil[i].mean(axis=0))
            r2_seir = r2_score(cases_brazil[i],fitted_data_brazil_seir[i].mean(axis=0))
            r2_our = 1/(2-r2_our)
            r2_seir = 1/(2-r2_seir)
            xlim = axes[5,i%5].get_xlim()
            ylim = axes[5,i%5].get_ylim()
            axes[5,i%5].text((xlim[1]-xlim[0]) / 8 + xlim[0], (ylim[1]-ylim[0]) / 8 *7 + ylim[0],'$R^2$ score:'+"%(r2).3f"%{'r2':r2_our},fontdict = {'size': 12, 'color': 'blue'})
            axes[5,i%5].text((xlim[1]-xlim[0]) / 8 + xlim[0], (ylim[1]-ylim[0]) / 8 *6 + ylim[0],'$R^2$ score:'+"%(r2).3f"%{'r2':r2_seir},fontdict = {'size': 12, 'color': 'green'})
            axes[5,i%5].grid(which='major',ls='--', alpha=0.8)
            axes[5,i%5].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            axes[5,i%5].set_title(city_name_list_brazil[i])
            total_R2.append(r2_our)
            total_R2_seir.append(r2_seir)
    else:
        # Draw New Cases
        for i in range(len(city_name_list_us)):

            tmp_x = city_list[city_name_list_us[i]]['second'] - city_list[city_name_list_us[i]]['first'] - 1
            tmp = date_axis_us[city_name_list_us[i]][tmp_x]

            # Our results
            c = np.diff(fitted_us[i],axis=1).mean(axis=0)
            axes[int(floor(i/5)),i%5].plot(date_axis_us[city_name_list_us[i]][:-1], np.where(c > 0, c, 0), label='Estimated Cases', color='blue')
            axes[int(floor(i/5)),i%5].fill_between(date_axis_us[city_name_list_us[i]][:-1],
                            np.where(c > 0, c, 0) + opt['std']*np.diff(fitted_us[i],axis=1).std(axis=0),
                            np.where(c > 0, c, 0) - opt['std']*np.diff(fitted_us[i],axis=1).std(axis=0), alpha=.75, color='#abc6e4')


            axes[int(floor(i/5)),i%5].scatter(tmp, c[tmp_x], c='k',marker='d', s = 40, label='Changing Point')

            # Classic SEIR
            c = np.diff(fitted_data_us_seir[i],axis=1).mean(axis=0)
            axes[int(floor(i/5)),i%5].plot(date_axis_us[city_name_list_us[i]][:-1], np.where(c > 0, c, 0), label='Traditional SEIR', color='g')
            if not opt['ignore_range']:
                axes[int(floor(i/5)),i%5].fill_between(date_axis_us[city_name_list_us[i]][:-1],
                            np.where(c > 0, c, 0) + opt['std']*np.diff(fitted_data_us_seir[i],axis=1).std(axis=0),
                            np.where(c > 0, c, 0) - opt['std']*np.diff(fitted_data_us_seir[i],axis=1).std(axis=0), alpha=.3, color='g')
            axes[int(floor(i/5)),i%5].scatter(tmp, c[tmp_x], c='k',marker='d', s = 40, label='Changing Point')

            # Real Cases
            c = np.diff(cases_us[i])
            c = np.where(c > 0, c, 0)
            c = ma_all(c,3)
            axes[int(floor(i/5)),i%5].scatter(date_axis_us[city_name_list_us[i]][:-1], c, c='r',marker='.', s=20, label='Reported Cases')


            r2_our = r2_score(c,np.diff(fitted_us[i],axis=1).mean(axis=0))
            r2_seir = r2_score(c,np.diff(fitted_data_us_seir[i],axis=1).mean(axis=0))
            r2_our = 1/(2-r2_our)
            r2_seir = 1/(2-r2_seir)
            xlim = axes[int(floor(i/5)),i%5].get_xlim()
            ylim = axes[int(floor(i/5)),i%5].get_ylim()
            axes[int(floor(i/5)),i%5].text((xlim[1]-xlim[0]) / 8 + xlim[0], (ylim[1]-ylim[0]) / 8 *7 + ylim[0],'$R^2$ score:'+"%(r2).3f"%{'r2':r2_our},fontdict = {'size': 12, 'color': 'blue'})
            axes[int(floor(i/5)),i%5].text((xlim[1]-xlim[0]) / 8 + xlim[0], (ylim[1]-ylim[0]) / 8 *6 + ylim[0],'$R^2$ score:'+"%(r2).3f"%{'r2':r2_seir},fontdict = {'size': 12, 'color': 'green'})
            axes[int(floor(i/5)),i%5].grid(which='major',ls='--', alpha=0.8)

            if city_name_list_us[i] == 'NY':
                axes[int(floor(i/5)),i%5].set_title('New York City')
            elif city_name_list_us[i] == 'New_Orleans':
                axes[int(floor(i/5)),i%5].set_title('New Orleans')
            elif city_name_list_us[i] == 'New_Haven':
                axes[int(floor(i/5)),i%5].set_title('New Haven')
            elif city_name_list_us[i] == 'Los_Angeles':
                axes[int(floor(i/5)),i%5].set_title('Los Angeles')
            else:
                axes[int(floor(i/5)),i%5].set_title(city_name_list_us[i])
            total_R2.append(r2_our)
            total_R2_seir.append(r2_seir)

        # India Cities
        for i in range(len(city_name_list_india)):

            # Our results
            axes[4,i%5].plot(date_axis_india[city_name_list_india[i]][:-1], np.diff(fitted_india[i],axis=1).mean(axis=0), label='Estimated Cases', color='blue')
            axes[4,i%5].fill_between(date_axis_india[city_name_list_india[i]][:-1],
                            np.diff(fitted_india[i],axis=1).mean(axis=0) + opt['std']*np.diff(fitted_india[i],axis=1).std(axis=0),
                            np.diff(fitted_india[i],axis=1).mean(axis=0) - opt['std']*np.diff(fitted_india[i],axis=1).std(axis=0), alpha=.75, color='#abc6e4')

            tmp_x = int((datetime.datetime.strptime(India[city_name_list_india[i]]['change'], "%Y-%m-%d") -  datetime.datetime.strptime(India[city_name_list_india[i]]['start'], "%Y-%m-%d")).days / India[city_name_list_india[i]]['sample_rate']) + 1
            tmp = date_axis_india[city_name_list_india[i]][tmp_x]
            axes[4,i%5].scatter(tmp, np.diff(fitted_india[i],axis=1).mean(axis=0)[tmp_x], c='k',marker='d', s = 40, label='Changing Point')

            # Classic SEIR
            axes[4,i%5].plot(date_axis_india[city_name_list_india[i]][:-1], np.diff(fitted_data_india_seir[i],axis=1).mean(axis=0), label='Traditional SEIR', color='g')
            if not opt['ignore_range']:
                axes[4,i%5].fill_between(date_axis_india[city_name_list_india[i]][:-1],
                                np.diff(fitted_data_india_seir[i],axis=1).mean(axis=0) + opt['std']*np.diff(fitted_data_india_seir[i],axis=1).std(axis=0),
                                np.diff(fitted_data_india_seir[i],axis=1).mean(axis=0) - opt['std']*np.diff(fitted_data_india_seir[i],axis=1).std(axis=0), alpha=.3, color='g')
            axes[4,i%5].scatter(tmp, np.diff(fitted_data_india_seir[i],axis=1).mean(axis=0)[tmp_x], c='k',marker='d', s = 40, label='Changing Point')

            # Real Cases
            c = np.diff(cases_india[i])
            c = np.where(c > 0, c, 0)
            c = ma_all(c,3)
            axes[4,i%5].scatter(date_axis_india[city_name_list_india[i]][:-1], c, c='r',marker='.', s=20, label='Reported Cases')


            r2_our = r2_score(c,np.diff(fitted_india[i],axis=1).mean(axis=0))
            r2_seir = r2_score(c,np.diff(fitted_data_india_seir[i],axis=1).mean(axis=0))
            r2_our = 1/(2-r2_our)
            r2_seir = 1/(2-r2_seir)
            xlim = axes[4,i%5].get_xlim()
            ylim = axes[4,i%5].get_ylim()
            axes[4,i%5].text((xlim[1]-xlim[0]) / 8 + xlim[0], (ylim[1]-ylim[0]) / 8 *7 + ylim[0],'$R^2$ score:'+"%(r2).3f"%{'r2':r2_our},fontdict = {'size': 12, 'color': 'blue'})
            axes[4,i%5].text((xlim[1]-xlim[0]) / 8 + xlim[0], (ylim[1]-ylim[0]) / 8 *6 + ylim[0],'$R^2$ score:'+"%(r2).3f"%{'r2':r2_seir},fontdict = {'size': 12, 'color': 'green'})

            axes[4,i%5].grid(which='major',ls='--', alpha=0.8)
            axes[4,i%5].ticklabel_format(axis='y',style='sci',scilimits=(0,0))

            axes[4,i%5].set_title(city_name_list_india[i])
            total_R2.append(r2_our)
            total_R2_seir.append(r2_seir)

        # Brazil Cities
        for i in range(len(city_name_list_brazil)):

            # Our results
            axes[5,i%5].plot(date_axis_brazil[city_name_list_brazil[i]][:-1], np.diff(fitted_brazil[i],axis=1).mean(axis=0), label='Estimated Cases', color='blue')
            axes[5,i%5].fill_between(date_axis_brazil[city_name_list_brazil[i]][:-1],
                            np.diff(fitted_brazil[i],axis=1).mean(axis=0) + opt['std']*np.diff(fitted_brazil[i],axis=1).std(axis=0),
                            np.diff(fitted_brazil[i],axis=1).mean(axis=0) - opt['std']*np.diff(fitted_brazil[i],axis=1).std(axis=0), alpha=.75, color='#abc6e5')

            tmp_x = int((datetime.datetime.strptime(Brazil[city_name_list_brazil[i]]['change'], "%Y-%m-%d") -  datetime.datetime.strptime(Brazil[city_name_list_brazil[i]]['start'], "%Y-%m-%d")).days / Brazil[city_name_list_brazil[i]]['sample_rate']) + 1
            tmp = date_axis_brazil[city_name_list_brazil[i]][tmp_x]
            axes[5,i%5].scatter(tmp, np.diff(fitted_brazil[i],axis=1).mean(axis=0)[tmp_x], c='k',marker='d', s = 50, label='Changing Point')

            # Classic SEIR
            axes[5,i%5].plot(date_axis_brazil[city_name_list_brazil[i]][:-1], np.diff(fitted_data_brazil_seir[i],axis=1).mean(axis=0), label='Traditional SEIR', color='g')
            if not opt['ignore_range']:
                axes[5,i%5].fill_between(date_axis_brazil[city_name_list_brazil[i]],
                                np.diff(fitted_data_brazil_seir[i],axis=1).mean(axis=0) + opt['std']*np.diff(fitted_data_brazil_seir[i],axis=1).std(axis=0),
                                np.diff(fitted_data_brazil_seir[i],axis=1).mean(axis=0) - opt['std']*np.diff(fitted_data_brazil_seir[i],axis=1).std(axis=0), alpha=.3, color='g')
            axes[5,i%5].scatter(tmp, np.diff(fitted_data_brazil_seir[i],axis=1).mean(axis=0)[tmp_x], c='k',marker='d', s = 50, label='Changing Point')

            # Real cases
            c = np.diff(cases_brazil[i])
            c = np.where(c > 0, c, 0)
            c = ma_all(c,3)
            axes[5,i%5].scatter(date_axis_brazil[city_name_list_brazil[i]][:-1], c, c='r',marker='.', s=20, label='Reported Cases')
            #axes[5,i%5].axvline(x=date_axis[city_name_list[i]][city_list[city_name_list[i]]['second'] - city_list[city_name_list[i]]['first'] - 1], color='#F27A11', linestyle = '--', linewidth=1.5)


            r2_our = r2_score(c,np.diff(fitted_brazil[i],axis=1).mean(axis=0))
            r2_seir = r2_score(c,np.diff(fitted_data_brazil_seir[i],axis=1).mean(axis=0))
            r2_our = 1/(2-r2_our)
            r2_seir = 1/(2-r2_seir)
            xlim = axes[5,i%5].get_xlim()
            ylim = axes[5,i%5].get_ylim()
            axes[5,i%5].text((xlim[1]-xlim[0]) / 8 + xlim[0], (ylim[1]-ylim[0]) / 8 *7 + ylim[0],'$R^2$ score:'+"%(r2).3f"%{'r2':r2_our},fontdict = {'size': 12, 'color': 'blue'})
            axes[5,i%5].text((xlim[1]-xlim[0]) / 8 + xlim[0], (ylim[1]-ylim[0]) / 8 *6 + ylim[0],'$R^2$ score:'+"%(r2).3f"%{'r2':r2_seir},fontdict = {'size': 12, 'color': 'green'})

            axes[5,i%5].grid(which='major',ls='--', alpha=0.8)
            axes[5,i%5].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            axes[5,i%5].set_title(city_name_list_brazil[i])
            total_R2.append(r2_our)
            total_R2_seir.append(r2_seir)


    plt.show()
    
    #plt.savefig('./fig1a.pdf',dpi=300)


if __name__ == "__main__":
    parser = setup_args()
    opt = vars(parser.parse_args())
    draw(opt)




