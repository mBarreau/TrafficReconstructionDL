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
from pyDOE import lhs
from neural_network import NeuralNetwork
        
def hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print('{:.0f}h {:.0f}m {:.0f}s'.format(h, m, s))

class ReconstructionNeuralNetwork():
    
    def __init__(self, x, t, rho, v, L, Tmax, V, F, N_f=1000, N_g=100):
        '''
        Initialize a neural network for density reconstruction

        Parameters
        ----------
        x : 2D numpy array of shape (N_data, N)
            space coordinate of training points.
        t : 1D numpy array of shape (N_data, 1)
            time coordinate of training points.
        rho : 2D numpy array of shape (N_data, N)
            density values at training points.
        v : list of N numpy array of shape (?,)
            velocity values at training points.
        L : float64
            Length of the spacial domain.
        Tmax : float64
            Length of the temporal domain.
        V : lambda function
            Velocity of the agents function.
        F : lambda function
            Flux of the PDE (velocity of the characteristics).
        N_f : integer, optional
            Number of physical points for F. The default is 1000.
        N_g : integer, optional
            Number of physical points for G. The default is 100.

        Returns
        -------
        None.

        '''
        
        self.Nxi = x.shape[1] # Number of agents
        
        num_hidden_layers = int(Tmax*8/100) 
        num_nodes_per_layer = int(20*L/7000) 
        layers = [2] # There are two inputs: space and time
        for _ in range(num_hidden_layers):
            layers.append(num_nodes_per_layer)
        layers.append(1)
        
        x_train, t_train, u_train, v_train, X_f_train, t_g_train = self.createTrainingDataset(x, t, rho, v, L, Tmax, N_f, N_g) # Creation of standardized training dataset
        
        self.neural_network = NeuralNetwork(x_train, t_train, u_train, v_train, X_f_train, t_g_train, layers_density=layers, 
                                              layers_trajectories=(1, 5, 5, 5, 1),
                                              layers_speed=(1, 5, 5, 1)) # Creation of the neural network

        self.train() # Training of the neural network
            
    def createTrainingDataset(self, x, t, rho, v, L, Tmax, N_f, N_g):       
        '''
        Standardize the dataset

        Parameters
        ----------
        x : list of N arrays of float64 (?,)
            Position of agents along time.
        t : list of N arrays of float64 (?,)
            Time coordinate of agents.
        rho : list of N arrays of float64 (?,)
            Density measurement from each agent.
        v : list of N arrays of float64 (?,)
            Velocity measurement from each agent.
        L : float
            Length of the road.
        Tmax : float
            Time-window.
        N_f : int
            Number of physical points for f.
        N_g : int
            Number of physical points for g.

        Returns
        -------
        x : list of N arrays of float64 (?,)
            Standardized position of agents along time.
        t : list of N arrays of float64 (?,)
            Standardized time coordinate of agents.
        u : list of N arrays of float64 (?,)
            Standardized density measurement from each agent.
        v : list of N arrays of float64 (?,)
            Standardized velocity measurement from each agent.
        X_f : 2D array of shape (N_f, 2)
            Standardized location of physical points for f.
        t_g : list of float64
            List of standardized physical points for g.

        '''
        
        self.lb = np.array([np.amin(x), np.amin(t)])
        self.ub = np.array([np.amax(x), np.amax(t)])
        self.lb[0], self.lb[1] = 0, 0
        
        x = [2*(x_temp - self.lb[0])/(self.ub[0] - self.lb[0]) - 1 for x_temp in x]
        t = [2*(t_temp - self.lb[1])/(self.ub[1] - self.lb[1]) - 1 for t_temp in t]
        rho = [2*rho_temp-1 for rho_temp in rho]
        v = [v_temp*(self.ub[0] - self.lb[0])/(self.ub[1] - self.lb[1]) for v_temp in v]
        
        X_f = np.array([2, 2])*lhs(2, samples=N_f)
        X_f = X_f - np.ones(X_f.shape)
        t_g = 2*lhs(1, samples=N_g)-1
        
        X_trajectories = np.array([x.reshape(-1,), (np.tile(t, (1, self.Nxi))).reshape(-1,)], dtype=np.float32).T
        X_f = np.vstack([X_f, X_trajectories])
        np.random.shuffle(X_f)
        
        return (x, t, rho, v, X_f, t_g)

    def train(self):
        '''
        Train the neural network

        Returns
        -------
        None.

        '''
        start = time()
        self.neural_network.train()
        hms(time() - start)
        
    def predict(self, x, t):
        '''
        Return the estimated density at (t, x)

        Parameters
        ----------
        x : numpy array (?, )
            space coordinate.
        t : numpy array (?, )
            time coordinate.

        Returns
        -------
        numpy array
            estimated density.

        '''
        
        x = 2*(x - self.lb[0])/(self.ub[0] - self.lb[0])-1
        t = 2*(t - self.lb[1])/(self.ub[1] - self.lb[1])-1
        
        return self.neural_network.predict(x, t)/2+0.5
    
    def predict_trajectories(self, t):
        '''
        Return the estimated agents' locations at t

        Parameters
        ----------
        t : numpy array (?, )
            time coordinate.

        Returns
        -------
        numpy array
            estimated agents location.

        '''
        
        t = 2*(t - self.lb[1])/(self.ub[1] - self.lb[1])-1        
        return (self.neural_network.predict_trajectories(t)+1)*(self.ub[0] - self.lb[0])/2 + self.lb[0]
    
    
    def plot(self, axisPlot, rho):
        '''
        

        Parameters
        ----------
        axisPlot : tuple of two 1D-numpy arrays of shape (?,)
            Plot mesh.
        rho : 2D numpy array
            Values of the real density at axisPlot.

        Returns
        -------
        list of two Figures
            return the reconstruction and error figures.

        '''
        
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
        
        rho_prediction = self.predict(xstar, tstar).reshape(Nx, Nt)
        X_prediction = self.predict_trajectories(t.reshape(t.shape[0], 1)).reshape(Nt, self.Nxi)

        figReconstruction = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(t, x)
        plt.pcolor(X, Y, rho_prediction, vmin=0.0, vmax=1.0, shading='auto', rasterized=True)
        plt.plot(t, X_prediction, color="orange")
        plt.xlabel(r'Time [s]')
        plt.ylabel(r'Position [m]')
        plt.xlim(min(t), max(t))
        plt.ylim(min(x), max(x))
        plt.colorbar()
        plt.tight_layout()
        # plt.title('Reconstruction')
        plt.show()
        figReconstruction.savefig('reconstruction.eps', bbox_inches='tight')
        
        
        figError = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(t, x)
        plt.pcolor(X, Y, np.abs(rho_prediction-rho), vmin=0.0, vmax=1.0, shading='auto', rasterized=True)
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