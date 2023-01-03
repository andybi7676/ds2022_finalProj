import os
import math
import glob
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
            self.owner = int(egonet_fpath.split('/')[-1].split('.')[0])
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
        # print((1, 2) in self.valid_iv_pairs)
    
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
            # sims.append(self.alpha*simP + (1-self.alpha)*simN + 1e-5) # paper
            # ------proposed------- #
            sims.append(self.alpha*simP + (1-self.alpha)*simN)
        sims = np.array(sims)
        sims = self._normalize_npy(sims) # normalize sim
        diss = (1 - sims**2)**0.5        # concept: from cos(theta) to sin(theta)
        # ---------proposed-------- #
        # sims = np.array(sims)
        # diss = 1 / (sims)
        # diss = self._normalize_npy(diss)
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
    
    def visualize(self, out_fpath="./tmp.png"):
        fig, ax = plt.subplots()
        # print(self.user_rhos_npy.max(), self.user_deltas_npy.max())
        rhos_std = self.user_rhos_npy.std()
        deltas_std = self.user_deltas_npy.std()
        # print(rhos_std, deltas_std)
        ax.scatter(self.user_rhos_npy, self.user_deltas_npy)
        for user_id, rho, delta in zip(self.user_ids_npy, self.user_rhos_npy, self.user_deltas_npy):
            ax.annotate(user_id, (rho, delta))
        fig.savefig(out_fpath)
    
    def determine_centers(self, top_k=False):
        threshold = 1.0
        rhos_deltas_combination = self.user_rhos_npy + self.user_deltas_npy
        center_filter = (rhos_deltas_combination > threshold)
        center_user_ids_npy = self.user_ids_npy[center_filter]
        # sort by local density:
        self.center_user_ids_rhos = [(user_id, self.Rho[user_id]) for user_id in center_user_ids_npy]
        self.center_user_ids_rhos.sort(key=lambda x: x[1])
        self.center_user_ids = []
        for user_id in center_user_ids_npy:
            self.center_user_ids.append(user_id)
            self.users[user_id].is_center = True

    def assign_circles(self): # need to improve in the future
        for user_id in self.user_ids_npy:
            cur_user = self.users[user_id]
            if cur_user.is_center: continue
            neighbors = cur_user.neighbors
            parent_id = None
            cur_rho = self.Rho[user_id]
            valid_neighbors_ids = []
            nearest_nb_id = None
            min_distance = 1000
            for nb_id in neighbors:
                if self.Rho[nb_id] > cur_rho:
                    valid_neighbors_ids.append(nb_id)
                    cur_distance = self.get_distance(user_id, nb_id)
                    if cur_distance < min_distance:
                        parent_id = nb_id
                        min_distance = cur_distance
            if parent_id != None:
                self.users[parent_id].add_child(cur_user) 
        for center_user_id, _ in self.center_user_ids_rhos:
            center_user = self.users[center_user_id]
            descendants_ids = center_user.get_descendants_ids()
            for desc_id in descendants_ids:
                center_user.circle[desc_id] = self.get_distance(center_user_id, desc_id)
                
        # for user_id in self.user_ids_npy:
        #     if self.users[user_id].is_center: continue
        #     min_dist = 1.0
        #     min_dist_center_id = None
        #     for center_user_id in self.center_user_ids:
        #         cur_dist = self.get_distance(user_id, center_user_id)
        #         if cur_dist <= min_dist:
        #             min_dist = cur_dist
        #             min_dist_center_id = center_user_id
        #     self.users[min_dist_center_id].circle[user_id] = min_dist
    
    def integrate_circles(self):
        if len(self.center_user_ids_rhos) < 2: return
        cur_nxt_center_pairs = [(self.center_user_ids[j], self.center_user_ids[j+1]) for j in range(len(self.center_user_ids)-1)]
        for cur_center_id, nxt_center_id in cur_nxt_center_pairs:
            cur_center_user = self.users[cur_center_id]
            cur_circle_user_ids = cur_center_user.get_circle_list()
            nxt_center_user = self.users[nxt_center_id]
            if len(nxt_center_user.get_circle_list()) == 0: continue
            for user_id in cur_circle_user_ids:
                cur_dist = self.get_distance(user_id, nxt_center_id)
                if cur_dist < nxt_center_user.get_circle_distsmax():
                    nxt_center_user.circle[user_id] = cur_dist
    
    def merge_circles(self):
        pass
    
    def get_circles(self):
        self.determine_centers()
        print(self.center_user_ids_rhos)
        self.assign_circles()
        self.integrate_circles()
        self.circles = {}
        for center_user_id in self.center_user_ids:
            center_user = self.users[center_user_id]
            self.circles[center_user_id] = center_user.get_circle_list()
        print(self.circles)
    
    def get_answers(self):
        circle_reprs = []
        for circle_owner_id, circle_user_ids in self.circles.items():
            circle_repr = " ".join([str(circle_owner_id)] + [str(user_id) for user_id in circle_user_ids])
            circle_reprs.append(circle_repr)
        self.answer = f"{self.owner}," + ";".join(circle_reprs)
        return self.answer

def main():
    featurelist_fp = '/home/andybi7676/Desktop/ds2022_finalProj/featureList.txt'
    feature_fp = '/home/andybi7676/Desktop/ds2022_finalProj/features.txt'
    # cache_fp = '.cache/users.pkl'
    exp_dir = "./exps/improve_distance_func"
    training_root_dir = '/home/andybi7676/Desktop/ds2022_finalProj/Training'
    data_root_dir = '/home/andybi7676/Desktop/ds2022_finalProj/egonets'
    os.makedirs(exp_dir, exist_ok=True)
    visualize = True

    if visualize:
        visualize_dir = osp.join(exp_dir, "visualize")
        os.makedirs(visualize_dir, exist_ok=True)
    
    training_ego_ids =[train_data_fpath.split('/')[-1].split('.')[0] for train_data_fpath in glob.glob(osp.join(training_root_dir, "*.circles"))]
    print(training_ego_ids)
    users = Users(feature_fp, featurelist_fp) 
    Egonet.register_users(users)
    with open(osp.join(exp_dir, "hyp.txt"), 'w') as out_fw, open(
        osp.join(exp_dir, "err.txt"), 'w'
    ) as err_fw:
        print("UserId,Predicted", file=out_fw, flush=True)
        for ego_id in tqdm(training_ego_ids):
            try:
                ego_fpath = osp.join(data_root_dir, f"{ego_id}.egonet")
                ego = Egonet(ego_fpath) 
                if visualize:
                    ego.visualize(osp.join(visualize_dir, f"{ego_id}.png"))
                ego.get_circles()
                print(ego.get_answers(), file=out_fw, flush=True)
            except:
                print(ego_id, file=err_fw, flush=True)


if __name__ == "__main__":
    main()
    # print(ego_0.Rho)
    # print(ego_0.delta) 

    
