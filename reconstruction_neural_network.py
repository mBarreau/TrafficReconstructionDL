# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:15:37 2020

@author: barreau
"""

import numpy as np

#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from time import time

from scipy.stats import t as student

from pyDOE import lhs

from neural_network import NeuralNetwork

# np.random.seed(12345)
# tf.set_random_seed(12345)
        
def hms( seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print('{:.0f}h {:.0f}m {:.0f}s'.format(h, m, s))

class ReconstructionNeuralNetwork():
    
    def __init__(self, x, t, u, L, Tmax, units=5, layers=(2, 1), Vf=1, 
                 gamma=0, N_f=1000, N_g=100):
        
        self.Nxi = x.shape[1]
        
        num_hidden_layers = int(Tmax*8/100)
        num_nodes_per_layer = int(20*L/5000)
        layers = [2]
        for _ in range(num_hidden_layers):
            layers.append(num_nodes_per_layer)
        layers.append(1)
        
        x_train, t_train, u_train, X_f_train, t_g_train = self.createTrainingDataset(x, t, u, L, Tmax, N_f, N_g)
        VfNorm = Vf*(self.ub[1] - self.lb[1]) / (self.ub[0] - self.lb[0])
        gammaNorm = gamma * 2 * (self.ub[1] - self.lb[1]) / (self.ub[0] - self.lb[0])**2
        
        
        self.neural_network = NeuralNetwork(x_train, t_train, u_train, X_f_train, t_g_train, layers_density=layers, 
                                              layers_trajectories=(1, 2*self.Nxi, 2*self.Nxi, 2*self.Nxi, self.Nxi),
                                              Vf=VfNorm, gamma=gammaNorm)
        self.train()
            
    def createTrainingDataset(self, x, t, u, L, Tmax, N_f, N_g):       
        
        self.lb = np.array([np.amin(x), np.amin(t)])
        self.ub = np.array([np.amax(x), np.amax(t)])
        self.lb[0], self.lb[1] = 0, 0
        
        x = 2*(x - self.lb[0])/(self.ub[0] - self.lb[0])-1
        t = 2*(t - self.lb[1])/(self.ub[1] - self.lb[1])-1
        u = 2*u-1
        
        X_f = np.array([2, 2])*lhs(2, samples=N_f)
        X_f = X_f - np.ones(X_f.shape)
        t_g = 2*lhs(1, samples=N_g)-1
        
        X_trajectories = np.array([x.reshape(-1,), (np.tile(t, (1, self.Nxi))).reshape(-1,)], dtype=np.float32).T
        X_f = np.vstack([X_f, X_trajectories])
        
        # np.random.shuffle(X_f)
        # return (t_u_shuffled, pv_u_shuffled, u_shuffled, X_f, t_trajectories, x_trajectories, t_g)
        
        return (x, t, u, X_f, t_g)

    def train(self):
        start = time()
        self.neural_network.train()
        hms(time() - start)
        
    def predict(self, x, t):
        
        x = 2*(x - self.lb[0])/(self.ub[0] - self.lb[0])-1
        t = 2*(t - self.lb[1])/(self.ub[1] - self.lb[1])-1
        
        return self.neural_network.predict(x, t)/2+0.5
    
    def predict_trajectories(self, t):
        
        t = 2*(t - self.lb[1])/(self.ub[1] - self.lb[1])-1        
        return (self.neural_network.predict_trajectories(t)+1)*(self.ub[0] - self.lb[0])/2 + self.lb[0]
    
    
    def plot(self, axisPlot, u):
        x = axisPlot[0]
        t = axisPlot[1]

        Nx = len(x)
        Nt = len(t)
            
        XY_prediction = np.zeros((Nx * Nt, 2))
        k = 0
        for i in range(0, Nx):
            for j in range(0, Nt):
                XY_prediction[k] = np.array([x[i], t[j]])
                k = k + 1
        xstar = XY_prediction[:, 0:1]
        tstar = XY_prediction[:, 1:2]
        
        U_prediction = self.predict(xstar, tstar).reshape(Nx, Nt)
        X_prediction = self.predict_trajectories(t.reshape(t.shape[0], 1)).reshape(Nt, self.Nxi)

        figReconstruction = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(t, x)
        plt.pcolor(X, Y, U_prediction, vmin=0.0, vmax=1.0, shading='auto')
        plt.plot(t, X_prediction, color="orange")
        plt.xlabel(r'Time [s]')
        plt.ylabel(r'Position [m]')
        plt.xlim(min(t), max(t))
        plt.ylim(min(x), max(x))
        plt.colorbar()
        plt.tight_layout()
        # plt.title('Reconstruction')
        self.t_train = (np.tile(self.neural_network.t, (1,self.Nxi)).reshape((-1,1)) + 1)*(self.ub[1] - self.lb[1])/2 + self.lb[1]
        self.x_train = (self.neural_network.x.reshape((-1,1)) + 1)*(self.ub[0] - self.lb[0])/2 + self.lb[0]
        plt.scatter(self.t_train, self.x_train, s=0.5, c="red") 
        plt.show()
        figReconstruction.savefig('reconstruction.eps', bbox_inches='tight')
        
        
        figError = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(t, x)
        plt.pcolor(X, Y, np.abs(U_prediction-u), vmin=0.0, vmax=1.0, shading='auto')
        plt.plot(t, X_prediction, color="orange")
        plt.xlabel(r'Time [s]')
        plt.ylabel(r'Position [m]')
        plt.xlim(min(t), max(t))
        plt.ylim(min(x), max(x))
        plt.colorbar()
        plt.tight_layout()
        # plt.title('Absolute error')
        plt.show()
        figError.savefig('error.eps', bbox_inches='tight') 
        
        return [figReconstruction, figError]