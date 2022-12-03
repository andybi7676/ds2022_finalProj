import os
import math
import pickle as pkl
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from feature import *
from user import *

class Egonet():

    users = None
    @classmethod
    def register_users(cls, users):
        cls.users = users
    
    @classmethod
    def _init_value(cls):
        return 0.0
    
    def _normalize_npy(cls, np_ary):
        return (np_ary - np_ary.min()) / (np_ary.max() - np_ary.min())

    def __init__(self, egonet_fpath: str, alpha=0.3, sigma=0.25, d_c=0.0):
        assert self.users != None, f"please register users first!"
        self.alpha = alpha
        self.sigma = sigma
        self.d_c = d_c
        self.egonet_user_ids = []
        with open(egonet_fpath, 'r') as er:
            lines = er.readlines()
            for line in tqdm(lines):
                contents = line.strip('\r\n ').split(':')
                user_id = int(contents[0])
                friends_ids = [int(fr) for fr in contents[1].strip('\r\n ').split()]
                user = self.users[user_id]
                for fr_id in friends_ids:
                    user.add_neighbor_to_dict(self.users[fr_id])
                user.stabilize()
                self.egonet_user_ids.append(user_id)
        self.egonet_user_ids.sort()
        self.size = len(self.egonet_user_ids)
        self.create_iv_pairs()
        self.calculate_distance()
        self.calculate_local_density()
        self.calculate_delta_distance()
    
    def create_iv_pairs(self):
        self.valid_iv_pairs = []
        for id_i in self.egonet_user_ids:
            for id_j in self.egonet_user_ids:
                if id_i != id_j:
                    self.valid_iv_pairs.append((id_i, id_j))
        print((1, 2) in self.valid_iv_pairs)
    
    def log(self, log_str: str):
        print(log_str)

    def simN(self, user_i, user_v):
        salton_score = len(user_i & user_v) / max(1, math.sqrt(len(user_i) * len(user_v)))
        return salton_score
    
    def simP(self, user_i, user_v):
        score = user_i.feature * user_v.feature
        return score
    
    def get_distance(self, id_i: int, id_v: int):
        return self.D[(id_i, id_v)]
    
    def calculate_distance(self):
        self.D = {}
        sims = []
        for id_i, id_v in self.valid_iv_pairs:
            user_i = self.users[id_i]
            user_v = self.users[id_v]
            simP = self.simP(user_i, user_v)
            simN = self.simN(user_i, user_v)
            # self.D[(id_i, id_v)] = 1 / (self.alpha*simP + (1-self.alpha)*simN + 1e-3) # original
            sims.append(self.alpha*simP + (1-self.alpha)*simN)
        sims = np.array(sims)
        sims = self._normalize_npy(sims) # normalize sim
        diss = (1 - sims**2)**0.5        # concept: from cos(theta) to sin(theta)
        print(f"diss max = {diss.max()}, min={diss.min()}")
        for (id_i, id_v), dis in zip(self.valid_iv_pairs, diss):
            self.D[(id_i, id_v)] = dis
    
    def calculate_local_density(self):
        # store local density in 'user' also
        self.Rho = defaultdict(self._init_value)
        denominator = 2 * self.sigma**2
        def normalize(dis_i_v):
            reg_dis = abs(dis_i_v - self.d_c)
            return - reg_dis / denominator
        raw_rho = defaultdict(self._init_value)
        for id_i, id_v in self.valid_iv_pairs:
            add_val = math.exp(normalize(self.get_distance(id_i, id_v)))
            raw_rho[id_i] += add_val
        self.user_rhos_npy = np.array([raw_rho[user_id] for user_id in self.egonet_user_ids])
        self.user_rhos_npy = self._normalize_npy(self.user_rhos_npy)
        for id_i, rho in zip(self.egonet_user_ids, self.user_rhos_npy):
            self.Rho[id_i] = rho
            self.users[id_i].local_density = rho # Might be redundant
    
    def calculate_delta_distance(self):
        self.delta = defaultdict(self._init_value)
        self.user_ids_npy = np.array(self.egonet_user_ids)
        max_user_id = self.user_ids_npy[np.argmax(self.user_rhos_npy)]
        for id_i, rho_i in zip(self.user_ids_npy, self.user_rhos_npy):
            if id_i == max_user_id:
                v_filter = self.user_ids_npy != id_i
                metric = "max"
            else:
                v_filter = (self.user_rhos_npy >= rho_i) & (self.user_ids_npy != id_i) # implicitly filter out id_i
                metric = "min"
            v_candidates = self.user_ids_npy[v_filter]
            self.delta[id_i] = eval(metric)([self.get_distance(id_i, id_v) for id_v in v_candidates])
        self.user_deltas_npy = np.array([self.delta[user_id] for user_id in self.egonet_user_ids])
    
    def visualize(self):
        fig, ax = plt.subplots()
        print(self.user_rhos_npy.max(), self.user_deltas_npy.max())
        rhos_std = self.user_rhos_npy.std()
        deltas_std = self.user_deltas_npy.std()
        print(rhos_std, deltas_std)
        ax.scatter(self.user_rhos_npy, self.user_deltas_npy)
        for user_id, rho, delta in zip(self.user_ids_npy, self.user_rhos_npy, self.user_deltas_npy):
            ax.annotate(user_id, (rho, delta))
        fig.savefig("./tmp.png")
    
    # def define_centers(self):
    #     return 
    
    # def initialize_clusters(self):
        

if __name__ == "__main__":
    featurelist_fp = '/home/andybi7676/Desktop/ds2022_finalProj/featureList.txt'
    feature_fp = '/home/andybi7676/Desktop/ds2022_finalProj/features.txt'
    cache_fp = '.cache/users.pkl'
    
    users = Users(feature_fp, featurelist_fp) 
    Egonet.register_users(users)
    ego_fpath = '/home/andybi7676/Desktop/ds2022_finalProj/egonets/345.egonet'
    ego = Egonet(ego_fpath) 
    ego.visualize()
     
    # print(ego_0.Rho)
    # print(ego_0.delta) 

    
