import numpy as np
from pathlib import Path
import json
import pandas as pd





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



def load_params(topk=10):
    US_city_list, India, Brazil = init_city_list()

    country_list = []
    city_list = []
    topk_infRate_first = []
    topk_infRate_second = []
    topk_quarantineRate_first = []
    topk_quarantineRate_second = []
    topk_mob_first = []
    topk_mob_second = []

    for key,values in US_city_list.items():
        case = np.load('./cases/'+values['cases_name'])[values['first']:]
        fit_input_dir = Path('./results/fig1a_fit/our_model')
        npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/'+'*curve*.npy')])
        fitted_data = np.array([np.load(i).reshape(-1) for i in npy_datas])
        loss = np.square(fitted_data - case).sum(axis=1)

        params_locs = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/'+'opt_params*.json')])
        params = []
        for loc in params_locs:
            with open(loc,'r') as f:
                p = json.load(f)
                params.append(p)


        params_topk = [params[i] for i in list(np.argsort(loss)[:topk])]
        for i in range(len(params_topk)):
            country_list.append('US')
            city_list.append(key)
            topk_infRate_first.append(params_topk[i]['opt1']['pi'])
            topk_infRate_second.append(params_topk[i]['opt2']['pi'])
            topk_quarantineRate_first.append(params_topk[i]['opt1']['early_detect'])
            topk_quarantineRate_second.append(params_topk[i]['opt2']['early_detect'])
            topk_mob_first.append(params_topk[i]['opt1']['mobility'])
            topk_mob_second.append(params_topk[i]['opt2']['mobility'])

    for key,values in India.items():
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

        fit_input_dir = Path('./results/fig1a_fit/our_model')
        npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/'+'*curve*.npy')])
        fitted_data = np.array([np.load(i).reshape(-1) for i in npy_datas])
        loss = np.square(fitted_data - cases_data_processed).sum(axis=1)

        params_locs = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/'+'opt_params*.json')])
        params = []
        for loc in params_locs:
            with open(loc,'r') as f:
                p = json.load(f)
                params.append(p)

        params_topk = [params[i] for i in list(np.argsort(loss)[:topk])]
        for i in range(len(params_topk)):
            country_list.append('India')
            city_list.append(key)
            topk_infRate_first.append(params_topk[i][0]['Pi'])
            topk_infRate_second.append(params_topk[i][1]['Pi'])
            topk_quarantineRate_first.append(params_topk[i][0]['early_detect'])
            topk_quarantineRate_second.append(params_topk[i][1]['early_detect'])
            topk_mob_first.append(params_topk[i][0]['mobility'])
            topk_mob_second.append(params_topk[i][1]['mobility'])



    for key,values in Brazil.items():
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

        fit_input_dir = Path('./results/fig1a_fit/our_model')
        npy_datas = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/'+'*curve*.npy')])
        fitted_data = np.array([np.load(i).reshape(-1) for i in npy_datas])
        loss = np.square(fitted_data - cases_data_processed).sum(axis=1)

        params_locs = sorted([i for i in fit_input_dir.glob('./*'+str(key)+'*/'+'opt_params*.json')])
        params = []
        for loc in params_locs:
            with open(loc,'r') as f:
                p = json.load(f)
                params.append(p)

        params_topk = [params[i] for i in list(np.argsort(loss)[:topk])]
        for i in range(len(params_topk)):
            country_list.append('Brazil')
            city_list.append(key)
            topk_infRate_first.append(params_topk[i][0]['Pi'])
            topk_infRate_second.append(params_topk[i][1]['Pi'])
            topk_quarantineRate_first.append(params_topk[i][0]['early_detect'])
            topk_quarantineRate_second.append(params_topk[i][1]['early_detect'])
            topk_mob_first.append(params_topk[i][0]['mobility'])
            topk_mob_second.append(params_topk[i][1]['mobility'])


    params_dataframe = pd.DataFrame({'Country':country_list,'City':city_list, 'first_stage_infRate':topk_infRate_first, \
        'second_stage_infRate':topk_infRate_second,'first_stage_quarantineRate':topk_quarantineRate_first, \
        'second_stage_quarantineRate':topk_quarantineRate_second, 'first_stage_mob':topk_mob_first,'second_stage_mob':topk_mob_second})
    
    return params_dataframe




def analyze_params():
    params_top10 = load_params(topk=10)
    top10 = params_top10.groupby(['Country','City']).agg([np.mean, np.std]).reset_index()
    top10.to_csv('./top10_params.csv',index=False)





if __name__ == "__main__":
    analyze_params()




