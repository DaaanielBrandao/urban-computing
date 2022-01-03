import time
import math
import random
import numpy as np


class PoissonFactorModel(object):
    def __init__(self, K=30, alpha=20.0, beta=0.2, sim = False):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.U, self.L = None, None
        #
        self.sim = sim
        self.b_u, self.b_i = None,None
        #



    def save_model(self, path):
        ctime = time.time()
        print("Saving U and L...",)
        np.save(path + "U", self.U)
        np.save(path + "L", self.L)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_model(self, path):
        ctime = time.time()
        print("Loading U and L...",)
        self.U = np.load(path + "U.npy")
        self.L = np.load(path + "L.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def read_cat(self):
        file1 = open('./new_data/fsq_pois_cats.txt', 'r')
        Lines = file1.readlines()
        cat_dict = {}
        self.C1 = {}
        self.C0 = random.uniform(0, 0.5)
        for line in Lines:
            line_split = line.split()
            cat_dict[int(line_split[0])] = line_split[1]
            self.C1[line_split[1]] = random.uniform(0, 0.5)
        #print(len(self.C1))
        #print(cat_dict)
        return cat_dict
    def read_totd(self):
        file1 = open('./new_data/fsq_poi_totd.txt', 'r')
        Lines1 = file1.readlines()
        file2 = open('./new_data/fsq_user_totd.txt', 'r')
        Lines2 = file2.readlines()
        totd_poi_dict = {}
        totd_user_dict = {}
        self.T1 = {}
        self.T0 = random.uniform(0, 0.5)
        for line in Lines1:
            line_split = line.split()
            totd_poi_dict[int(line_split[0])] = line_split[1]
            self.T1[line_split[1]] = random.uniform(0, 0.5)
        for line in Lines2:
            line_split = line.split()
            totd_user_dict[int(line_split[0])] = line_split[1]
            self.T1[line_split[1]] = random.uniform(0, 0.5)
        return totd_poi_dict, totd_user_dict
    def read_season(self):
        file1 = open('./new_data/fsq_poi_season.txt', 'r')
        Lines1 = file1.readlines()
        file2 = open('./new_data/fsq_user_season.txt', 'r')
        Lines2 = file2.readlines()
        season_poi_dict = {}
        season_user_dict = {}
        self.S1 = {}
        self.S0 = random.uniform(0, 0.5)
        for line in Lines1:
            line_split = line.split()
            season_poi_dict[int(line_split[0])] = line_split[1]
            self.S1[line_split[1]] = random.uniform(0, 0.5)
        for line in Lines2:
            line_split = line.split()
            season_user_dict[int(line_split[0])] = line_split[1]
            self.S1[line_split[1]] = random.uniform(0, 0.5)
        return season_poi_dict, season_user_dict

    def Dist(self,C1,C0):
        #print(C1,C0)
        #print(math.sqrt((C1-C0)**2))
        return  math.sqrt((C1-C0)**2)
    def train(self, sparse_check_in_matrix, max_iters=50, learning_rate=1e-4):
        ctime = time.time()
        print("Training PFM...", )

        alpha = self.alpha
        beta = self.beta
        K = self.K

        F = sparse_check_in_matrix
        M, N = sparse_check_in_matrix.shape
        U = 0.5 * np.sqrt(np.random.gamma(alpha, beta, (M, K))) / K
        L = 0.5 * np.sqrt(np.random.gamma(alpha, beta, (N, K))) / K
        #
        self.U, self.L = U, L
        self.b_u = np.zeros(M)
        self.b_i = np.zeros(N)
        self.b =F[np.nonzero(F)].mean()

        self.cat = self.read_cat()
        self.totd_poi, self.totd_user = self.read_totd()
        self.season_poi, self.season_user = self.read_season()


        #print(F)

        F = F.tocoo()
        entry_index = list(zip(F.row, F.col))

        F = F.tocsr()
        F_dok = F.todok()
        
        #
        #self.b = np.mean(F_dok[np.where(F_dok != 0)])
        #self.b = np.mean(F_dok[np.nonzero(F_dok)])
        #print(self.b)
        #
        C1 = self.C1
        C0 = self.C0
        T1 = self.T1
        T0 = self.T0
        S1 = self.S1
        S0 = self.S0


        tau = 10
        last_loss = float('Inf')
        for iters in range(max_iters):
            F_Y = F_dok.copy()
            #
            learning_rate_k = learning_rate * tau / (tau + iters)
            if self.sim:
                for i,j in entry_index:
                    prediction = self.predict(i, j)
                    e = (F_Y[i,j] - prediction)
                    #categories
                    C1[self.cat[j]] += learning_rate_k*(e*(U[i].dot(L[j])* ((C1[self.cat[j]]-C0)/self.Dist(C1[self.cat[j]],C0)) - alpha* C1[self.cat[j]]))
                    C0 += learning_rate_k*(e*(U[i].dot(L[j])* ((C1[self.cat[j]]-C0)/self.Dist(C1[self.cat[j]],C0)) - alpha* C0))
                    #totd
                    T1[self.totd_poi[j]] += learning_rate_k*(e*(U[i].dot(L[j])* ((T1[self.totd_poi[j]]-T0)/self.Dist(T1[self.totd_poi[j]],T0)) - alpha* T1[self.totd_poi[j]]))
                    T0 += learning_rate_k*(e*(U[i].dot(L[j])* ((T1[self.totd_poi[j]]-T0)/self.Dist(T1[self.totd_poi[j]],T0)) - alpha* T0))
                    #season
                    S1[self.season_poi[j]] += learning_rate_k*(e*(U[i].dot(L[j])* ((S1[self.season_poi[j]]-S0)/self.Dist(S1[self.season_poi[j]],S0)) - alpha* S1[self.season_poi[j]]))
                    S0 += learning_rate_k*(e*(U[i].dot(L[j])* ((S1[self.season_poi[j]]-S0)/self.Dist(S1[self.season_poi[j]],S0)) - alpha* S0))  
            for i, j in entry_index:
                #prediction = self.predict(i, j)
                #e = (F_Y[i,j] - prediction)
                #print(i,' ',j ,': ',prediction)
                #self.b_u[i] += learning_rate_k * (e - beta * self.b_u[i])
                #self.b_i[j] += learning_rate_k * (e - beta * self.b_i[j])         
                F_Y[i, j] = 1.0 * F_dok[i, j] / U[i].dot(L[j]) - 1
            F_Y = F_Y.tocsr()

    
            U += learning_rate_k * (F_Y.dot(L) + (alpha - 1) / U - 1 / beta)
            L += learning_rate_k * ((F_Y.T).dot(U) + (alpha - 1) / L - 1 / beta)
            #
            self.U, self.L = U, L
            self.C0, self.C1 = C0, C1
            self.T0, self.T1 = T0, T1
            self.S0, self.S1 = S0, S1
            #
            loss = 0.0
            for i, j in entry_index:
                loss += (F_dok[i, j] - U[i].dot(L[j]))**2

            print('Iteration:', iters,  'loss:', loss)

            if loss > last_loss:
                print("Early termination.")
                break
            last_loss = loss

        print("Done. Elapsed time:", time.time() - ctime, "s")
        self.U, self.L = U, L

    def Sim(self,cat,totd_user,totd_poi,season_user,season_poi):
        #print(self.Dist(self.C1[cat],self.C0))
        return (1 - self.Dist(self.C1[cat],self.C0))*(1 - self.Dist(self.T1[totd_user],self.T1[totd_poi]))*(1 - self.Dist(self.S1[season_user],self.S1[season_poi]))

    def predict(self, uid, lid, sigmoid=False):
        if sigmoid:
            return 1.0 / (1 + math.exp(-self.U[uid].dot(self.L[lid])))
        elif self.sim:
           #print(lid,lid in self.cat)
            return self.U[uid].dot(self.L[lid]) * self.Sim(self.cat[lid], self.totd_user[uid],self.totd_poi[lid],self.season_user[uid],self.season_poi[lid]) #+ self.b_u[uid] + self.b_i[lid] + self.b
        return self.U[uid].dot(self.L[lid]) 