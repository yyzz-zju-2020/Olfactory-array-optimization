"""
Ref:https://github.com/luizaes/sca-algorithm and https://github.com/LucXiong/Swarm-intelligence-optimization-algorithm/blob/main/SCA.py
Mirjalili, S. (2016). SCA: A Sine Cosine Algorithm for solving optimization problems. Knowledge-Based Systems, 96, 120-133. https://doi.org/10.1016/j.knosys.2015.12.022. 
"""

#This method takes the example of selecting 6 arrays from 16 arrays
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sko.tools import func_transformer
from Best_combination import combine_point
from sko import logger
import os


class sca():
    def __init__(self, pop_size=5, n_dim=2, a=2, lb=-1e5, ub=1e5, max_iter=200, func=None,n_processes=0):
        self.pop = pop_size
        self.n_dim = n_dim
        self.a = a
        self.func = func_transformer(func,n_processes)
        self.max_iter = max_iter

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        a = [ac for ac in range(16)]
        self.X = np.zeros((self.pop, self.n_dim))
        for jkl in range(self.pop):
            self.X[jkl] = np.array(random.sample(a, 6))
        self.X = np.sort(self.X)
        self.Y = self.func(self.X).reshape(-1, 1) # y = f(x) for all particles

        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = [np.inf for i in range(self.pop)]  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.ally = []
        self.update_gbest()

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        for i in range(len(self.Y)):
            if self.pbest_y[i] > self.Y[i]:
                self.pbest_x[i] = self.X[i]
                self.pbest_y[i] = self.Y[i]


    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.index(min(self.pbest_y))
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def update(self, i):
        r1 = self.a - i * ((self.a) / self.max_iter)
        for j in range(self.pop):
            for k in range(self.n_dim):
                r2 = 2 * math.pi * random.uniform(0.0, 1.0)
                r3 = 2 * random.uniform(0.0, 1.0)
                r4 = random.uniform(0.0, 1.0)
                if r4 < 0.5:
                    try:
                        self.X[j][k] = self.X[j][k] + (r1 * math.sin(r2) * abs(r3 * self.gbest_x[k] - self.X[j][k]))
                    except:
                        self.X[j][k] = self.X[j][k] + (r1 * math.sin(r2) * abs(r3 * self.gbest_x[0][k] - self.X[j][k]))
                else:
                    try:
                        self.X[j][k] = self.X[j][k] + (r1 * math.cos(r2) * abs(r3 * self.gbest_x[k] - self.X[j][k]))
                    except:
                        self.X[j][k] = self.X[j][k] + (r1 * math.cos(r2) * abs(r3 * self.gbest_x[0][k] - self.X[j][k]))
        self.X = np.clip(self.X, self.lb, self.ub)
        for ax in range(len(self.X)):
            X_temp = self.X[ax]
            for ab in range(len(X_temp)):
                X_temp[ab] = round(X_temp[ab])
            X_temp = np.sort(X_temp) 
            while len(np.unique(X_temp)) != 6: #For duplicate integers, neighboring integers are chosen as one of the solutions
                for ik in range(len(X_temp) - 1):
                    if X_temp[ik + 1] == X_temp[ik]:
                        akl = random.choice([X_temp[ik + 1] - 1, X_temp[ik + 1] + 1])
                        if akl < 0:
                            X_temp[ik + 1] = X_temp[ik + 1] + 1
                        elif akl > 16:
                            X_temp[ik + 1] = X_temp[ik + 1] - 1
                        else:
                            X_temp[ik + 1] = akl
                    else:
                        pass
                    X_temp = np.sort(X_temp)
            # if len(X_temp) < 6:         #Compared to the last method, this solution has greater randomness
            #     count = len(X_temp)
            #     while count < 6:
            #         c = np.random.randint(0, 16)

            #         if c not in X_temp:
            #             c = np.array(c)
            #             X_temp = np.append(X_temp, c)
            #             count = count + 1
            # X_temp = np.sort(X_temp)
            self.X[ax] = X_temp
        # print(self.X)
        self.Y = self.func(self.X).reshape(-1, 1)   # Function for fitness evaluation of new solutions


    def run(self):
        logger1 = logger.Logger(os.path.join('file_save', 'pbest.txt'), title='pbest')
        logger2 = logger.Logger(os.path.join('file_save', 'ally.txt'), title='ally')
        logger1.set_names(['pbest'])
        logger2.set_names(['ally1', 'ally2'])
        for i in range(self.max_iter):
            logger2.append([round(self.Y[0][0], 6), round(self.Y[1][0], 6)])
            self.update(i)
            self.update_pbest()
            self.update_gbest()
            logger1.append([round(self.gbest_y[0], 6)])
            self.gbest_y_hist.append(self.gbest_y)
            self.ally.append(min(self.Y))

        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

class CallingCounter(object):
    def __init__ (self, func):
        self.func = func
        self.count = 0

    def __call__ (self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@CallingCounter
def demo_func(p):
    aqc = demo_func.count
    print(f'D %d'%aqc)

    x1, x2, x3, x4, x5, x6 = p
    print([x1, x2, x3, x4, x5, x6])
    x1, x2, x3, x4, x5, x6 = x1-1, x2-1, x3-1, x4-1, x5-1, x6-1
    # aqc = aqc - 1
    time.sleep(2)
    Loss, Train_acc, Val_acc, a0 = combine_point([x1, x2, x3, x4, x5, x6], aqc)
    logger0.append([x1, x2, x3, x4, x5, x6, Loss, Train_acc, Val_acc])
    return Loss


path_txt = 'file_save'
logger0 = logger.Logger(os.path.join(path_txt, 'log1.txt'),title = '123')

logger0.set_names(['Com1', 'Com2','Com3','Com4','Com5','Com6','Loss', 'Train_Acc','Val_Acc'])
from sko.tools import set_run_mode
import time

if __name__ == '__main__':
    # time.sleep(2)
    set_run_mode(demo_func, 'commom') #Whether the multithreading is synchronized or not can be selected here. common or multithreading
    n_dim = 6
    lb = [1 for i in range(n_dim)]
    ub = [16 for i in range(n_dim)]
    demo_func = demo_func
    sca = sca(n_dim=6, pop_size=2, a=20, max_iter=200, lb=lb, ub=ub, func=demo_func)
    sca.run()
    print('best_x is ', sca.gbest_x, 'best_y is', sca.gbest_y)

    plt.plot(sca.gbest_y_hist)
    plt.plot(sca.ally)
    plt.show()
