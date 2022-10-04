import numpy as np
from math import sqrt, floor, exp
import copy
import matplotlib.pyplot as plt

class City:
    def __init__(self, opt):
        self.units_num = opt['units']
        self.L = int(sqrt(self.units_num))
        assert self.L ** 2 == self.units_num

        self.unit_dist = opt['unit_distance']  # physical distance of each unit block

        # Init disease params
        self.disease_params = dict()
        self.disease_params['Pi'] = opt['Pi']   # transmission rate of I
        self.disease_params['Pe'] = opt['Pe']   # transmission rate of E
        self.disease_params['PE'] = opt['PE']   # probability of a health people to be E when get infected
        self.disease_params['e_to_i'] = opt['e_to_i']  # probability of the a E turn to I
        self.disease_params['i_to_r'] = opt['i_to_r']  # recover rate of I

        # Init policy params
        self.policy_params = dict()
        self.policy_params['mobility'] = opt['mobility']  # Mobility rate of gravity model

        self.policy_params['self_quarantine'] = opt['self_quarantine'] # self quarantine of S
        self.policy_params['ki_discount'] = opt['ki_disc'] # discount of I when moving
        self.policy_params['ke_discount'] = opt['ke_disc'] # discount of E when moving

        self.policy_params['P_i_discount'] = opt['Pi_disc'] # discount of transmission rate of I
        self.policy_params['P_e_discount'] = opt['Pe_disc'] # discount of transmission rate of E

        self.policy_params['early_detect'] = opt['early_detect']   # early detect rate (to accelerate I to R)

        if opt.__contains__('cases_bias'):
            self.cases_bias = opt['cases_bias']

        if opt.__contains__('init_cases_num'):
            self.init_cases_num = opt['init_cases_num']

        # Init the blocks for SEIR model
        self.blocks_matrix = np.zeros((self.units_num, 4))   # unit number x SEIR



        # Init the distance matrix for moving (for faster speed)
        self.dist_matrix = np.ones((self.units_num,self.units_num))
        for i in range(self.units_num):
            for j in range(self.units_num):
                if i != j:
                    x_i = int(floor(i / self.L))
                    y_i = int(floor(i % self.L))
                    x_j = int(floor(j / self.L))
                    y_j = int(floor(j % self.L))
                    self.dist_matrix[i,j] = exp((abs(x_i - x_j) + abs(y_i - y_j)) * self.unit_dist / 82)


    def setPopCases(self,pop, cases):
        pop_cp = copy.deepcopy(pop)
        cases_copy = copy.deepcopy(cases)
        cases_copy = cases_copy.reshape(-1)

        self.pop = pop_cp.astype(int)
        self.cases = cases_copy.astype(int)
        self.max_pop = int(pop_cp.max() * 1.2)

    def get_blk_pop(self):
        return self.blocks_matrix.sum(axis=1).reshape(-1)


    def fit(self, pi, early_detect,  mobility):
        self.disease_params['Pi'] = float(pi)
        self.disease_params['Pe'] = float(pi)
        self.policy_params['early_detect'] = float(early_detect)
        self.policy_params['mobility'] = float(mobility)

        self.blocks_matrix = np.zeros((self.units_num, 4))
        self.init_blocks(self.pop, manual_init_case=True)

        cases_cp = copy.deepcopy(self.cases)
        S,E,I,R,new_spread = self.begin_simulate(len(cases_cp))
        new_spread = new_spread.cumsum()
        new_spread = new_spread.reshape(-1)


        diff = -np.square(new_spread + self.cases_bias - cases_cp).sum()
        assert ~np.isnan(diff)
        print(diff)
        return diff

    def fit_second(self,pi, early_detect, mobility):
        self.disease_params['Pi'] = float(pi)
        self.disease_params['Pe'] = float(pi)
        self.policy_params['early_detect'] = float(early_detect)
        self.policy_params['mobility'] = float(mobility)

        self.blocks_matrix = np.zeros((self.units_num, 4))
        self.init_blocks(self.pop, ckpt = True)

        cases_cp = copy.deepcopy(self.cases)
        S,E,I,R,new_spread = self.begin_simulate(len(cases_cp),fit=True)
        new_spread = new_spread.cumsum()
        new_spread = new_spread.reshape(-1)
        new_spread += self.ckpt['total_infect']


        diff = -np.square(new_spread + self.cases_bias - cases_cp).sum()
        assert ~np.isnan(diff)
        print(diff)
        return diff


    def init_blocks(self, population, ckpt = False,manual_init_case=False):
        if not ckpt:
            pop_copy = copy.deepcopy(population)
            init_case_index = np.argsort(pop_copy.reshape(-1))[-self.init_cases_num:]

            self.max_pop = int(pop_copy.max() * 1.2)
            for idx in range(self.units_num):
                if idx in init_case_index:
                    self.blocks_matrix[idx, 0] = pop_copy[idx]  #S
                    self.blocks_matrix[idx, 2] = 1              #I
                else:
                    self.blocks_matrix[idx,0] = pop_copy[idx]   #S
        else:
            self.blocks_matrix = copy.deepcopy(self.ckpt['data'])


    def move(self,fit=True):
        """ Move individuals according to gravity model. It can achieve no uncertenty at all, at the cost of efficiency

        return:
            none
        """

        pop_vec = self.get_blk_pop() # unit_num x 1
        assert pop_vec.min() >= 0
        pop_vec_cp = copy.deepcopy(pop_vec).reshape(-1,1).tolist()
        move_matrix = self.policy_params['mobility'] * (np.power(pop_vec_cp,0.46).reshape(-1,1) * np.power(pop_vec_cp,0.64).T)
        move_matrix = move_matrix / self.dist_matrix
        for i in range(move_matrix.shape[0]):
            move_matrix[i,i] = 0


        pop = self.blocks_matrix.sum(axis=1)
        out_num = move_matrix.sum(axis=1)
        violate_index = out_num > pop
        if violate_index.sum() > 0:
            move_matrix[violate_index,:] = move_matrix[violate_index,:] / out_num[violate_index].reshape(-1,1) * pop[violate_index].reshape(-1,1)

        proportion = self.blocks_matrix / self.blocks_matrix.sum(axis=1,keepdims=True)
        proportion[np.isnan(proportion)] = 0
        move_with_proportion = np.expand_dims(move_matrix,2) * np.expand_dims(proportion,1)

        move_out = np.floor(move_with_proportion.sum(axis=1))
        #move_in = np.floor(move_with_proportion.sum(axis=0))

        move_out_E_index = np.argsort(move_with_proportion[:, :, 1], axis=1)[:,::-1]
        move_out_I_index = np.argsort(move_with_proportion[:, :, 2], axis=1)[:,::-1]


        for i in range(move_with_proportion.shape[0]):
            j = 0
            while move_out[i, 1] > 0:
                out_num = np.ceil(move_with_proportion[i, move_out_E_index[i,j], 1])
                move_with_proportion[i, move_out_E_index[i,j], 1] = out_num
                move_out[i, 1] -= out_num
                j += 1
            move_with_proportion[i, move_out_E_index[i,j:], 1] = 0

            j = 0
            while move_out[i, 2] > 0:
                out_num = np.ceil(move_with_proportion[i, move_out_I_index[i,j], 2])
                move_with_proportion[i, move_out_I_index[i,j], 2] = out_num
                move_out[i, 2] -= out_num
                j += 1
            move_with_proportion[i, move_out_I_index[i,j:], 2] = 0

        move_out = np.floor(move_with_proportion.sum(axis=1))
        move_in = np.floor(move_with_proportion.sum(axis=0))

        move = move_in - move_out

        self.blocks_matrix = self.blocks_matrix + move
        self.blocks_matrix[self.blocks_matrix<0] = 0
        assert np.isnan(self.blocks_matrix).sum() < 1

        if not fit:
            return np.abs(np.linalg.eigvals(move_matrix)).max()

    def spread(self):
        #Step0. Self quarantine
        S_quarantine = self.blocks_matrix[:,0] * self.policy_params['self_quarantine']
        self.blocks_matrix[:, 0] -= S_quarantine
        self.blocks_matrix[:, 3] += S_quarantine


        I_quarantine = np.where((self.blocks_matrix[:,2] * self.policy_params['early_detect']) < 1, np.zeros((self.blocks_matrix.shape[0])), self.blocks_matrix[:,2] * self.policy_params['early_detect'])
        self.blocks_matrix[:,2] -= I_quarantine
        E_quarantine = np.where((self.blocks_matrix[:, 1] * self.policy_params['early_detect']) < 1,
                                np.zeros((self.blocks_matrix.shape[0])),
                                self.blocks_matrix[:, 1] * self.policy_params['early_detect'])
        self.blocks_matrix[:, 1] -= E_quarantine
        self.blocks_matrix[:, 3] += (I_quarantine + E_quarantine)


        # Step2.Transmission
        S_infect = self.blocks_matrix[:,0] * self.blocks_matrix[:,1] * self.disease_params['Pe'] + self.blocks_matrix[:,0] * self.blocks_matrix[:,2] * self.disease_params['Pi']
        S_infect[S_infect > self.blocks_matrix[:,0]] = 0
        E_new = S_infect * self.disease_params['PE']
        I_new = S_infect - E_new

        # Step3. E_to_I
        E_to_I = self.blocks_matrix[:,1] * self.disease_params['e_to_i']

        # Step4. I_to_R
        I_to_R = self.blocks_matrix[:, 2] * self.disease_params['i_to_r']

        # Step5. Update parameters

        self.blocks_matrix[:, 0] -= S_infect
        self.blocks_matrix[:, 1] += (E_new - E_to_I)
        self.blocks_matrix[:, 2] += (I_new + E_to_I - I_to_R)
        self.blocks_matrix[:, 3] += I_to_R

        assert np.isnan(self.blocks_matrix).sum() < 1

        return (I_new + E_to_I).sum()

    def move_and_spread(self,fit = True):
        if fit:
            self.move()
        else:
            move_rho = self.move()
        newly_spread = self.spread()
        if fit:
            return newly_spread
        else:
            return newly_spread,move_rho

    def begin_simulate(self, iter_num, fit = True):
        S = np.zeros((1, iter_num))
        E = np.zeros((1, iter_num))
        I = np.zeros((1, iter_num))
        R = np.zeros((1, iter_num))
        new_spread = np.zeros((1, iter_num))
        if not fit:
            move_rho = np.zeros((1, iter_num))
            for i in range(iter_num):
                # 绘制每次simulate的分布
                # plt.clf()
                # S1 = self.blocks_matrix[:,0]
                # I1 = self.blocks_matrix[:,1]
                # #plt.plot(S1, label="S")
                # plt.plot(I1, label="I")
                # plt.tight_layout()
                # plt.legend()
                # plt.savefig(fname='./temp/'+str(i)+'.png', figsize=[8, 6])
                # plt.clf()
                #
                S[0, i] = self.blocks_matrix[:, 0].sum()
                E[0, i] = self.blocks_matrix[:, 1].sum()
                I[0, i] = self.blocks_matrix[:, 2].sum()
                R[0, i] = self.blocks_matrix[:, 3].sum()
                new_spread[0, i], move_rho[0, i] = self.move_and_spread(fit)

            return S, E, I, R, new_spread, move_rho
        else:
            for i in range(iter_num):
                # 绘制每次simulate的分布
                # plt.clf()
                # S1 = self.blocks_matrix[:,0]
                # I1 = self.blocks_matrix[:,1]
                # #plt.plot(S1, label="S")
                # plt.plot(I1, label="I")
                # plt.tight_layout()
                # plt.legend()
                # plt.savefig(fname='./temp/'+str(i)+'.png', figsize=[8, 6])
                # plt.clf()

                S[0, i] = self.blocks_matrix[:, 0].sum()
                E[0, i] = self.blocks_matrix[:, 1].sum()
                I[0, i] = self.blocks_matrix[:, 2].sum()
                R[0, i] = self.blocks_matrix[:, 3].sum()
                new_spread[0, i] = self.move_and_spread(fit)

            return S, E, I, R, new_spread


    def begin_simulate_multi_parted(self,opts_list,cases_list,save_path, fit = True):

        epochs = len(opts_list)

        for i in range(epochs):
            self.disease_params['Pi'] = float(opts_list[i]['Pi'])
            self.disease_params['Pe'] = float(opts_list[i]['Pi'])
            self.policy_params['early_detect'] = float(opts_list[i]['early_detect'])
            self.policy_params['mobility'] = float(opts_list[i]['mobility'])

            self.blocks_matrix = np.zeros((self.units_num, 4))
            self.init_blocks(self.pop, manual_init_case=(i == 0), ckpt=(i>0))

            if not fit:
                S_tmp, E_tmp, I_tmp, R_tmp, new_spread_tmp, move_rho = self.begin_simulate(len(cases_list[i]),fit)
            else:
                S_tmp, E_tmp, I_tmp, R_tmp, new_spread_tmp = self.begin_simulate(len(cases_list[i]), fit)
            
            if i == 0:
                S = S_tmp
                E = E_tmp
                I = I_tmp
                R = R_tmp
                new_spread = new_spread_tmp
            else:
                S = np.concatenate((S, S_tmp),axis=1)
                E = np.concatenate((E, E_tmp),axis=1)
                I = np.concatenate((I, I_tmp),axis=1)
                R = np.concatenate((R, R_tmp),axis=1)
                new_spread = np.concatenate((new_spread, new_spread_tmp), axis=1)

            self.make_check_point(float(new_spread.cumsum()[-1]))


        S = S.sum(axis=0)
        E = E.sum(axis=0)
        I = I.sum(axis=0)
        R = R.sum(axis=0)
        new_spread = new_spread.cumsum().reshape(-1) + self.cases_bias

        cases_total = np.array(cases_list[0])
        for i in range(epochs - 1):
            cases_total = np.concatenate((cases_total,np.array(cases_list[i+1])))

        #plt.plot(S, label="S")
        plt.plot(E, label="E")
        plt.plot(I, label="I")
        #plt.plot(R, label="R")
        plt.plot(new_spread, label='Total EI')
        plt.plot(cases_total, label='True')

        plt.ylabel("People")
        plt.xlabel("Step")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname=save_path, figsize=[8, 6])
        plt.clf()

        return new_spread

    def make_check_point(self,total_infect):
        self.ckpt = {'data':self.blocks_matrix,'total_infect':total_infect}
