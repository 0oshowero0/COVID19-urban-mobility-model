import numpy as np
from multiprocessing import Pool
from datetime import datetime
from argparse import ArgumentParser
from COVID_Model import City
from pathlib import Path
import json
from bayes_opt import BayesianOptimization
import pandas as pd
import setproctitle
setproctitle.setproctitle('Fortaleza_SEIR@hanzhenyu')


MULTI_PROCESSING = 20

def load_pop(data_dir):
    pop = np.load(data_dir).reshape(-1, 1).astype('float32')
    std = pop[pop>100].std()
    mean = pop[pop>100].mean()
    upper_bond = mean + 3*std
    pop = np.where(pop>upper_bond, upper_bond, pop)
    return pop

def load_cases(data_dir, name):
    data = pd.read_csv(data_dir)
    #cases = data[data['Município']==name].iloc[:,1:].to_numpy().reshape(-1, 1).astype('float32')
    cases = data[data['Município']==name]
    return cases


def setup_args(parser=None):
    """ Set up arguments

    return:
        python dictionary
    """
    if parser is None:
        parser = ArgumentParser()
    # Default Params
    parser.add_argument('--city_name', default='Fortaleza', help='City Name')
    parser.add_argument('--save_dir', default='./simulated_results_Fortaleza_625/', help='Result Loc')
    parser.add_argument('--population_data', default='./population/fortaleza_pop.npy', help='Loc of pop data')
    parser.add_argument('--cases_data', default='./cases/Brazil_epidemic_district_timeline.csv', help='Loc of cases data')
    parser.add_argument('--units', default=625, help='Unit Num', type=int)
    parser.add_argument('--unit_distance', default=1, help='Unit distance between blocks(km)', type=int)
    parser.add_argument('--start_date', default='2021-01-01', help='Start Time')
    parser.add_argument('--change_point', default='2021-04-01', help='Interval of cases')
    parser.add_argument('--final_date', default='2021-07-31', help='End Time')
    parser.add_argument('--sample_rate', default=4, help='Sample Rate of cases curve')

    parser.add_argument('--Pi', default=5*2.7913484249081293e-05, help='transmission rate of I (to fit)', type=float)
    parser.add_argument('--Pe', default=5*2.7913484249081293e-05, help='transmission rate of E (the same with Pi)', type=float)
    parser.add_argument('--PE', default=0.3, help='probability of a health people to be E when get infected', type=float)
    parser.add_argument('--e_to_i', default=1 / 5.2, help='probability of the E turn to I', type=float)
    parser.add_argument('--i_to_r', default=1 / 14, help='recover rate of I', type=float)

    parser.add_argument('--mobility', default=0.4, help='Mobility Param (to fit)', type=float)
    parser.add_argument('--early_detect', default=0, help='early detect rate (to fit; to accelerate I to R)', type=float)

    parser.add_argument('--self_quarantine', default=0, help='Self Quarantine of S (deprecated, not actually use)', type=float)
    parser.add_argument('--ki_disc', default=1, help='mobility discount of I when moving (deprecated, not actually use)', type=float)
    parser.add_argument('--ke_disc', default=1, help='mobility discount of E when moving (deprecated, not actually use)', type=float)
    parser.add_argument('--Pi_disc', default=1, help='discount of transmission rate of I (deprecated, not actually use)', type=float)
    parser.add_argument('--Pe_disc', default=1, help='discount of transmission rate of E (deprecated, not actually use)', type=float)

    return parser




def multi_process_fit(process_i,fit_epoch):
    parser = setup_args()
    opt = vars(parser.parse_args())
    output_dir = Path(opt['save_dir'])
    output_dir.mkdir(exist_ok=True,parents=True)
    ################################################################################
    # Load Data
    pop_data = load_pop(opt['population_data'])
    cases_data = load_cases(opt['cases_data'],opt['city_name'])

    start_index = np.where(cases_data.columns == opt['start_date'])[0]
    change_index = np.where(cases_data.columns == opt['change_point'])[0]
    final_index = np.where(cases_data.columns == opt['final_date'])[0]

    # Sampling epidemic curve
    origin_x = np.linspace(0, cases_data.shape[1]-1, num=cases_data.shape[1]-1, endpoint=False)
    num_new_points = int((cases_data.shape[1]-1)/opt['sample_rate'])
    resample_x = np.linspace(0, cases_data.shape[1]-1, num=num_new_points, endpoint=False)
    cases_resample = np.interp(x=resample_x, xp=origin_x, fp=cases_data.iloc[:,1:].to_numpy().reshape(-1))
    new_start_index = int(start_index / opt['sample_rate'])
    new_change_index = int(change_index / opt['sample_rate'])
    new_final_index = int(final_index / opt['sample_rate'])

    cases_data_processed = []
    cases_data_processed.append(cases_resample[new_start_index:new_change_index])
    cases_data_processed.append(cases_resample[new_change_index:new_final_index])

    # Set bias of cases number
    cases_bias = cases_resample[new_start_index]
    # Set active cases
    init_cases_num = np.diff(cases_data.iloc[:,(int(start_index)-3):int(start_index)]).sum()
    opt['cases_bias'] = cases_bias
    opt['init_cases_num'] = int(init_cases_num)




    optimizers = []
    # Fit first part
    city = City(opt)
    city.setPopCases(pop_data, cases_data_processed[0])
    city.init_blocks(pop_data, manual_init_case=True)

    pbounds = {'pi': (0, 0.0006), 'early_detect': (0, 1), 'mobility': (0, 0.0003)}

    optimizer = BayesianOptimization(
        f=city.fit,
        pbounds=pbounds,
    )
    optimizer.maximize(
        init_points=20,
        n_iter=fit_epoch,
    )
    optimizers.append(optimizer)

    #  Fit second part
    opt['Pi'] = optimizers[0].max['params']['pi']
    opt['Pe'] = optimizers[0].max['params']['pi']
    opt['early_detect'] = optimizers[0].max['params']['early_detect']
    opt['mobility'] = optimizers[0].max['params']['mobility']
    city = City(opt)
    city.setPopCases(pop_data, cases_data_processed[0])
    city.init_blocks(pop_data,manual_init_case=True)
    S_number, E_number, I_number, R_number, new_spread = city.begin_simulate(len(cases_data_processed[0]))

    new_pop = city.get_blk_pop()
    city.setPopCases(new_pop, cases_data_processed[1])
    city.make_check_point(float(new_spread.cumsum()[-1]))

    pbounds = {'pi': (0, 0.0006),'early_detect': (0, 1), 'mobility': (optimizer.max['params']['mobility'],0.0003)}

    optimizer = BayesianOptimization(
        f=city.fit_second,
        pbounds=pbounds
    )
    optimizer.maximize(
        init_points=20,
        n_iter=fit_epoch,
    )

    optimizers.append(optimizer)


    # Forward
    city = City(opt)
    city.setPopCases(pop_data, cases_data_processed[0])
    city.init_blocks(pop_data, manual_init_case=True)
    opts = []
    for optimizer in optimizers:
        opt = {'Pi': optimizer.max['params']['pi'], 'early_detect': optimizer.max['params']['early_detect'],
                'mobility': optimizer.max['params']['mobility']}
        opts.append(opt)

    new_spread = city.begin_simulate_multi_parted(opts, cases_data_processed,output_dir.joinpath('result_' + str(process_i).zfill(2) + '.png'))


    i = 0 
    total_opt = {}
    for opt in opts:
        total_opt['opt'+str(i)] = opt


    with open(output_dir.joinpath('opt_params_' + str(process_i).zfill(2) + '.json'), 'w') as f:
        json.dump(opts, f)

    np.save(output_dir.joinpath('result_curve_' + str(process_i).zfill(2) + '.npy'), new_spread.reshape(-1))




if __name__ == "__main__":
    fit_num = 40
    p = Pool(MULTI_PROCESSING)

    result = [p.apply_async(multi_process_fit, args=(i,200)) for i in range(fit_num)]

    for i in result:
       i.get()


    print(datetime.now())