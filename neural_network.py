# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:17:54 2020

@author: barreau
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.backend as K

import numpy as np
np.random.seed(1234)

import matplotlib.pyplot as plt

class NeuralNetwork():
    
    def __init__(self, N_neurons, Vf=1, gamma=0.05, mu1=1, mu2=0):
        self.Vf = Vf
        self.gamma = gamma
        self.mu1 = mu1
        self.mu2 = mu2
        self.N_neurons = N_neurons
        self.updateModel(self.N_neurons)
        
    def updateModel(self, N_neurons=-1):
        
        if isinstance(N_neurons, list) == False:
            N_neurons = self.N_neurons            
        
        input_tensor = Input(shape=(2,))
        hidden = input_tensor
        for i in range(len(N_neurons)):
            hidden = Dense(N_neurons[i], activation='relu')(hidden)
        output_tensor = Dense(1, activation='linear')(hidden)
        self.model = Model(input_tensor, output_tensor)
        
        self.model.compile(loss=self.custom_loss_wrapper(input_tensor), optimizer="adam")
        
    def updateMu(self, mu1=-1, mu2=-1):
        if mu1 >= 0:
            self.mu1 = mu1
        if mu2 >= 0:
            self.mu2 = mu2
        self.updateModel()
        
    def f(self, u, ux, uxx):
        return -self.Vf*(1-2*u)*ux + self.gamma*uxx
    
    def fit(self, x_train, t_train, z_train, epochs=200):
        xt_train = np.zeros((x_train.shape[0],2))
        for i in range(x_train.shape[0]):
            xt_train[i,:] = [x_train[i], t_train[i]]
        epochs_train = epochs
        history = self.model.fit(xt_train, z_train, epochs=epochs_train, verbose=1)
        return history
    
    def custom_loss_wrapper(self,input_tensor):
        def custom_loss(u_true, u_pred):
            MSEu = K.mean(K.square(u_pred - u_true), axis=-1)
            
            # x = K.slice(input_tensor, [0, 0], [-1, 1])
            # t = K.slice(input_tensor, [0, 1], [-1, 1])
            
            uPrime = K.gradients(u_pred, input_tensor)[-1]
            ux = K.slice(uPrime, [0, 0], [-1, 1]);
            ut = K.slice(uPrime, [0, 1], [-1, 1]);
            
            uxPrime = K.gradients(ux, input_tensor)[-1]
            uxx = K.slice(uxPrime, [0, 0], [-1, 1]);
            
            MSEf = K.mean(K.square(ut - self.f( u_pred, ux, uxx)), axis=-1)
            
            return self.mu1*MSEu + self.mu2*MSEf
        
        return custom_loss
    
    def predict(self, x, t):
        if isinstance(x, list):
            xt = np.zeros((len(x),2))
            for i in range(len(x)):
                xt[i,:] = [x[i], t[i]]
            return self.model.predict(xt)
        else:
            return self.predict([x], [t])
        
    def plot(self, axisPlot):
        x = axisPlot[0]
        t = axisPlot[1]
        
        Nx = len(x)
        Nt = len(t)
        
        XY_prediction = np.zeros((Nx*Nt,2))
        k = 0
        for i in range(0, Nx):
            for j in range(0, Nt):
                XY_prediction[k] = np.array([x[i], t[j]])
                k = k + 1
        Z_prediction = self.model.predict(XY_prediction)
        Z_prediction = Z_prediction.reshape(Nx, Nt)
        
        fig = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(t, x)
        plt.pcolor(X, Y, Z_prediction)
        plt.xlabel('Time [h]')
        plt.ylabel('Position [km]')
        plt.xlim(min(t), max(t))
        plt.ylim(min(x), max(x))
        plt.colorbar()
        plt.tight_layout()
        # fig.savefig('densityEstimated.eps', bbox_inches='tight')
        plt.show()
