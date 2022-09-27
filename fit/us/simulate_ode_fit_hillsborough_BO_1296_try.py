import numpy as np
from multiprocessing import Pool
from datetime import datetime
from argparse import ArgumentParser
from COVID_Model import City
from pathlib import Path
import json
from bayes_opt import BayesianOptimization
import setproctitle
setproctitle.setproctitle('Hillsborough_SEIR@hanzhenyu')


MULTI_PROCESSING = 20

def load_pop(data_dir):
    pop = np.load(data_dir).reshape(-1, 1).astype('float32')
    std = pop[pop>100].std()
    mean = pop[pop>100].mean()
    upper_bond = mean + 2*std
    pop = np.where(pop>upper_bond, upper_bond, pop)
    return pop

def load_cases(data_dir):
    cases = np.load(data_dir).reshape(-1, 1).astype('float32')
    return cases


def setup_args(parser=None):
    """ Set up arguments

    return:
        python dictionary
    """
    if parser is None:
        parser = ArgumentParser()
    # 定义基本参数
    parser.add_argument('--population_data', default='./population/hillsborough_pop.npy', help='Loc of pop data')
    parser.add_argument('--cases_data', default='./cases/hillsborough.npy', help='Loc of cases data')
    parser.add_argument('--units', default=1296, help='Unit Num', type=int)
    parser.add_argument('--unit_distance', default=2, help='Unit distance between blocks(km)', type=int)
    parser.add_argument('--init_cases', default=10, help='Init cases', type=int)

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




def multi_process_fit(i,fit_epoch):
    output_dir = Path('./simulated_results_Hillsborough_1296/')
    output_dir.mkdir(exist_ok=True,parents=True)

    parser = setup_args()
    opt = vars(parser.parse_args())
    ################################################################################
    # Load Data
    pop_data = load_pop(opt['population_data'])
    cases_data = load_cases(opt['cases_data'])
    cases_data_first = cases_data[57:70-6+14]
    cases_data_second = cases_data[70-6+14:]



    error = 99999999999999999999
    while error > 0.5:
        error = 0
        # Fit first part
        city = City(opt)
        city.setPopCases(pop_data, cases_data_first)
        city.init_blocks(pop_data)

        pbounds = {'pi': (0, 0.0006), 'early_detect': (0, 1), 'mobility': (0, 0.0003)}

        optimizer1 = BayesianOptimization(
            f=city.fit,
            pbounds=pbounds,
        )
        optimizer1.maximize(
            init_points=20,
            n_iter=fit_epoch,
        )

        # Fit second part
        opt['Pi'] = optimizer1.max['params']['pi']
        opt['Pe'] = optimizer1.max['params']['pi']
        opt['early_detect'] = optimizer1.max['params']['early_detect']
        opt['mobility'] = optimizer1.max['params']['mobility']
        city = City(opt)
        city.setPopCases(pop_data, cases_data_first)
        city.init_blocks(pop_data)
        S_number, E_number, I_number, R_number, new_spread = city.begin_simulate(len(cases_data_first))


        new_pop = city.get_blk_pop()
        city.setPopCases(new_pop, cases_data_second)
        city.make_check_point(float(new_spread.cumsum()[-1]))

    error = 99999999999999999999
    while error > 0.5:
        error = 0

        pbounds = { 'pi': (0, 0.0006),'early_detect': (0, 1), 'mobility':(0,optimizer1.max['params']['mobility'])}

        optimizer2 = BayesianOptimization(
            f=city.fit_second,
            pbounds=pbounds
        )
        optimizer2.maximize(
            init_points=20,
            n_iter=fit_epoch,
        )

        # Forward use fitted params.
        city = City(opt)
        city.setPopCases(pop_data, cases_data_first)
        city.init_blocks(pop_data)
        opt1 = {'Pi': optimizer1.max['params']['pi'], 'early_detect': optimizer1.max['params']['early_detect'],
                 'mobility': optimizer1.max['params']['mobility']}
        opt2 = {'Pi': optimizer2.max['params']['pi'], 'early_detect': optimizer2.max['params']['early_detect'],
                 'mobility': optimizer2.max['params']['mobility']}

        new_spread = city.begin_simulate_two_parted(opt1, opt2, cases_data_first, cases_data_second,
                                                    output_dir.joinpath('result_' + str(i).zfill(2) + '.png'))




        opts = {'opt1': {'pi': optimizer1.max['params']['pi'], 'early_detect': optimizer1.max['params']['early_detect'], \
                         'mobility': optimizer1.max['params']['mobility']}, \
                'opt2': {'pi': optimizer2.max['params']['pi'], 'early_detect': optimizer2.max['params']['early_detect'], \
                         'mobility': optimizer2.max['params']['mobility']}
                }


        with open(output_dir.joinpath('opt_params_' + str(i).zfill(2) + '.json'), 'w') as f:
            json.dump(opts, f)

        np.save(output_dir.joinpath('result_curve_' + str(i).zfill(2) + '.npy'), new_spread.reshape(-1))



if __name__ == "__main__":
    fit_num = 40
    p = Pool(MULTI_PROCESSING)

    result = [p.apply_async(multi_process_fit, args=(i,200)) for i in range(fit_num)]

    for i in result:
       i.get()



    print(datetime.now())