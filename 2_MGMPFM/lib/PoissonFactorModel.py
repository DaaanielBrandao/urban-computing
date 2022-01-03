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
        self.T1 = {}
        self.T0 = random.uniform(0, 0.5)
        for line in Lines:
            line_split = line.split()
            cat_dict[int(line_split[0])] = line_split[1]
            self.T1[line_split[1]] = random.uniform(0, 0.5)
        #print(len(self.T1))
        #print(cat_dict)
        return cat_dict

    def Dist(self,T1,T0):
        #print(T1,T0)
        #print(math.sqrt((T1-T0)**2))
        return  math.sqrt((T1-T0)**2)
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
        #print(self.b)
        #

        self.cat = self.read_cat()


        #print(F)

        F = F.tocoo()
        entry_index = list(zip(F.row, F.col))

        F = F.tocsr()
        F_dok = F.todok()
        
        #
        #self.b = np.mean(F_dok[np.where(F_dok != 0)])
        #
        T1 = self.T1
        T0 = self.T0


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
                    #print(prediction)
                    T1[self.cat[j]] += learning_rate_k*(e*(U[i].dot(L[j])* ((T1[self.cat[j]]-T0)/self.Dist(T1[self.cat[j]],T0)) - alpha* T1[self.cat[j]]))
                    T0 += learning_rate_k*(e*(U[i].dot(L[j])* ((T1[self.cat[j]]-T0)/self.Dist(T1[self.cat[j]],T0)) - alpha* T0)) 
            for i, j in entry_index:
                #prediction = self.predict(i, j)
                #e = (F_Y[i,j] - prediction)
        
                #print(i,' ',j ,': ',prediction)
                #self.b_u[i] += alpha * (e - beta * self.b_u[i])
                #self.b_i[j] += alpha * (e - beta * self.b_i[j])         
                F_Y[i, j] = 1.0 * F_dok[i, j] / U[i].dot(L[j]) - 1
            F_Y = F_Y.tocsr()

    
            U += learning_rate_k * (F_Y.dot(L) + (alpha - 1) / U - 1 / beta)
            L += learning_rate_k * ((F_Y.T).dot(U) + (alpha - 1) / L - 1 / beta)
            #
            self.U, self.L = U, L
            self.T0, self.T1 = T0, T1
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

    def Sim(self,cat):
        #print(self.Dist(self.T1[cat],self.T0))
        return 1 - self.Dist(self.T1[cat],self.T0)

    def predict(self, uid, lid, sigmoid=False):
        if sigmoid:
            return 1.0 / (1 + math.exp(-self.U[uid].dot(self.L[lid])))
        elif self.sim:
           #print(lid,lid in self.cat)
            return self.U[uid].dot(self.L[lid]) * self.Sim(self.cat[lid])
        return self.U[uid].dot(self.L[lid]) 