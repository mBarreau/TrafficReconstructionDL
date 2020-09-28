# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:15:37 2020

@author: barreau
"""

import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf

import numpy as np

#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from time import time

from scipy.stats import t as student

from pyDOE import lhs

# np.random.seed(12345)
# tf.set_random_seed(12345)

class NeuralNetwork():

    def __init__(self, x, t, u, X_f, t_g, layers, trajectories_layers, 
                 units=1, Vf=1, gamma=0.0, init=[], initWeights=[], initBias=[], alpha=1):

        # np.random.seed(1234)
        # tf.set_random_seed(1234)

        self.x = x
        self.t = t
        self.u = u.reshape((-1,1))

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        self.t_g = t_g
        
        self.Nxi = trajectories_layers[-1]

        self.Vf = Vf
        self.VfVar = tf.Variable(Vf, dtype=tf.float32, trainable=False)
        self.gamma = np.sqrt(gamma)
        self.gammaVar = tf.Variable(tf.truncated_normal([1,1], mean=self.gamma, stddev=0.01, dtype=tf.float32), dtype=tf.float32, trainable=True)[0,0]

        self.units, self.bias = self.initialize_neural_network_unit(units, layers, init, alpha)
        listVarUnits = [self.bias]
        for i in range(len(self.units)):
            listVarUnits.extend(self.units[i].getVariables())
        
        self.weights_trajectories, self.biases_trajectories = self.initialize_neural_network(layers=trajectories_layers, initWeights=initWeights, initBias=initBias, alpha=alpha)
        # listVarTrajectories = self.weights_trajectories + self.biases_trajectories
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        # PDE part
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        
        self.u_pred = self.net_u(tf.reshape(self.net_x(self.t_tf), (-1,1)), 
                                  tf.reshape(tf.tile(self.t_tf, [1, self.Nxi]), (-1,1)))
        # self.u_pred = self.net_u(tf.reshape(self.x_tf, (-1,1)), 
        #                          tf.reshape(tf.tile(self.t_tf, [1, self.Nxi]), (-1,1)))
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)        
        
        # Agents part
        self.t_g_tf = tf.placeholder(tf.float32, shape=[None, self.t_g.shape[1]])
        
        self.x_pred = self.net_x(self.t_tf)
        self.g_pred = self.net_g(self.t_g_tf)

        # MSE part
        self.MSEu = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.MSEf = tf.reduce_mean(tf.square(self.f_pred))
        self.save_loss = []
        
        self.MSEtrajectories = tf.reduce_sum(tf.square(self.x_tf - self.x_pred))/self.Nxi
        self.MSEg = tf.reduce_mean(tf.square(self.g_pred))
        
        self.loss_trajectories = 1*(self.MSEtrajectories + 0.5*self.MSEg)
        # self.loss = self.MSEu + self.MSEtrajectories + 0.5*(self.MSEf + 1*self.MSEg + 0.1*tf.square(self.gammaVar))
        self.loss = self.MSEu + 0.1*self.MSEf + 0.5*self.loss_trajectories
        self.lossPrecise = self.MSEu + self.MSEf + self.loss_trajectories + tf.square(self.gammaVar)
        
        self.optimizer_trajectories = tf.contrib.opt.ScipyOptimizerInterface(self.loss_trajectories, 
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 5000,
                                                                          'maxfun': 5000,
                                                                          'maxcor': 50,
                                                                          'maxls': 20,
                                                                          'ftol': 5.0 * np.finfo(float).eps})
        
        self.optimizerAdam = tf.train.AdamOptimizer().minimize(self.loss, var_list=listVarUnits)
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=listVarUnits,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 5000,
                                                                          'maxfun': 5000,
                                                                          'maxcor': 50,
                                                                          'maxls': 20,
                                                                          'ftol': 5.0 * np.finfo(float).eps})
        
        self.optimizerPrecise = tf.contrib.opt.ScipyOptimizerInterface(self.lossPrecise,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                          'maxfun': 50000,
                                                                          'maxcor': 150,
                                                                          'maxls': 75,
                                                                          'ftol': 1.0 * np.finfo(float).eps})
        

        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    def initialize_neural_network(self, layers=(2, 2), initWeights=[], initBias=[], alpha=1):
        weights, biases = [], []
        
        num_layers = len(layers)
        if len(initWeights) == 0:
            initWeights = [np.nan]*num_layers
            initBias = [np.nan]*num_layers
            
        for l in range(num_layers-1):
            
            if np.isnan(initWeights[l]).any():
                initWeights[l] = np.zeros((layers[l], layers[l+1]), dtype=np.float32)
                initBias[l] = np.zeros((1, layers[l+1]), dtype=np.float32)
                
            W = self.xavier_initializer(size=[layers[l], layers[l+1]], init=initWeights[l], act="tanh", alpha=alpha)
            b = tf.Variable(initBias[l], dtype=tf.float32) 

            weights.append(W)
            biases.append(b)
            
        return weights, biases
    
    def initialize_neural_network_unit(self, nb_units=1, layers=(2, 2), init=[], alpha=1):
        units = []
        
        if len(init) == 0:
            bias = tf.Variable(0, dtype=tf.float32) 
        else:
            bias = tf.Variable(init[-1], dtype=tf.float32) 
        
        for u in range(nb_units):
            if len(init) == 0:
                initUnit = Unit()
                initUnit.zeroUnit(layers)
            else:
                initUnit = init[u]

            units.append(self.initialize_unit(layers, nb_units, initUnit, alpha=1))
            
        return (units, bias)
    
    def initialize_unit(self, layers, nb_units, init, alpha=1):
        unit = Unit(weights=[], biases=[])
        
        num_layers = len(layers)
        for l in range(num_layers-1):
            W = self.xavier_initializer(size=[layers[l], layers[l+1]], init=init.getWeights(l), act="tanh", alpha=alpha)
            b = tf.Variable(init.getBiases(l), dtype=tf.float32) 

            unit.addWeights(W)
            unit.addBiases(b)
            
        unit.weight = self.xavier_initializer(size=[1, 1], init=init.weight, alpha=np.sqrt(2/(nb_units+1))*alpha)
            
        return unit
    
    def save(self):
        convertedUnits = []
        for unit in self.units:
            convertedUnits.append(unit.convert(self.sess))
        convertedUnits.append(self.sess.run(self.bias))
        return (convertedUnits, self.sess.run(self.weights_trajectories), self.sess.run(self.biases_trajectories))

    def xavier_initializer(self, size, init, act="tanh", alpha=1):
        in_dim = size[0]
        out_dim = size[1]

        xavier_stddev = np.sqrt(2/in_dim)*alpha
        xavier_bound = np.sqrt(6/(in_dim + out_dim))*alpha
        
        if act == "relu":
            return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], mean=init, stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)
        else:
            return tf.Variable(init + tf.random.uniform([in_dim, out_dim], minval=init-xavier_bound, maxval=init+xavier_bound, dtype=tf.float32), dtype=tf.float32)
            #return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], mean=init, stddev=xavier_bound*np.sqrt(2/6), dtype=tf.float32), dtype=tf.float32)
           
    
    def neural_network_trajectories(self, X, weights, biases, act=tf.nn.tanh):
        num_layers = len(weights) + 1

        # H = (X - lb) / (ub - lb)
        H = X
        for l in range(num_layers - 2):
            W, b = weights[l], biases[l]
            H = act(tf.add(tf.matmul(H, W), b))
        W, b = weights[-1], biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    
    def neural_network(self, X, units, bias, act=tf.nn.tanh):
        num_units = len(units)

        Y = bias
        #H = (X - lb[1]) / (ub[1] - lb[1])
        H = X
        for u in range(num_units): 
            Y = tf.add(Y, self.neural_network_unit(H, units[u], act, nb_unit=num_units))

        #return (Y - lb[0])/(ub[0] - lb[0])
        return Y
    
    def neural_network_unit(self, X, unit, act, nb_unit=1):
        
        weights = unit.getWeights()
        biases = unit.getBiases()
        num_layers = len(weights) + 1

        H = X
        for l in range(num_layers - 2): # hidden layers
            W, b = weights[l], biases[l]
            H = act(tf.add(tf.matmul(H, W), b))
            
        W, b = weights[-1], biases[-1] # Output layer
        if nb_unit > 1:
            return unit.weight*tf.nn.tanh(tf.add(tf.matmul(H, W), b)) # This is for several units
        else:
            return tf.add(tf.matmul(H, W), b) # This is for a single unit network

    def net_u(self, x, t):
        u = self.neural_network(tf.concat([x,t],1), self.units, self.bias)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        # f = u_t + self.VfVar * (1 - 2 * u) * u_x - self.gammaVar**2 * u_xx
        f = u_t - self.VfVar * u * u_x \
            - self.gammaVar**2 * u_xx
        return f
    
    def net_x(self, t):
        x = self.neural_network_trajectories(t, self.weights_trajectories, self.biases_trajectories)
        return x
    
    def net_g(self, t):
        x_trajectories = self.net_x(t) 
        g = []
        for i in range(x_trajectories.shape[1]):
            x = tf.slice(x_trajectories, [0,i], [-1,1])
            x_t = tf.gradients(x, t)[0]
            u = self.net_u(x, t)
            # g.append(x_t - self.VfVar * (1 - u))
            g.append(x_t - self.VfVar * (1 - u)/2)
        return g

    def loss_callback(self, MSEu, MSEf, MSEtrajectories, MSEg, total_loss, gamma):
        
        if self.epoch%10 == 1:
            print('Epoch: %.0f | MSEu: %.5e | MSEf: %.5e | MSEtrajectories: %.5e | MSEg: %.5e | Gamma: %.5e | Total: %.5e' %
                  (self.epoch, MSEu, MSEf, MSEtrajectories, MSEg, gamma**2, total_loss))
            
        self.epoch += 1

        self.save_loss.append(total_loss)
        
    def loss_callback_trajectory(self, MSEtrajectories, MSEg, total_loss):
        
        if self.epoch%10 == 1:
            print('Epoch: %.0f | MSEtrajectories: %.5e | MSEg: %.5e | Total: %.5e' %
                  (self.epoch, MSEtrajectories, MSEg, total_loss))
            
        self.epoch += 1

    def train(self):
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u, 
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                   self.t_g_tf: self.t_g}
        
        self.epoch = 1
        print('*********************************')
        print('*** TRAJECTORY RECONSTRUCTION ***')
        print('*********************************')
        self.optimizer_trajectories.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.MSEtrajectories, self.MSEg, self.loss_trajectories],
                                loss_callback=self.loss_callback_trajectory)
        
        print('*********************************')
        print('******** RECONSTRUCTION *********')
        print('*********************************')
        print('Adam')
        self.epoch = 1
        for _ in range(500):
            if self.epoch%10 == 1:
                self.loss_callback(self.sess.run(self.MSEu, {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}), 
                                self.sess.run(self.MSEf, {self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}), 
                                self.sess.run(self.MSEtrajectories, {self.x_tf: self.x, self.t_tf: self.t}), 
                                self.sess.run(self.MSEg, {self.t_g_tf: self.t_g}), 
                                self.sess.run(self.loss, tf_dict), 
                                self.sess.run(self.gammaVar))
            self.sess.run(self.optimizerAdam, tf_dict)
            self.epoch = self.epoch + 1   
            
        self.loss_callback(self.sess.run(self.MSEu, {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}), 
                            self.sess.run(self.MSEf, {self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}), 
                            self.sess.run(self.MSEtrajectories, {self.x_tf: self.x, self.t_tf: self.t}), 
                            self.sess.run(self.MSEg, {self.t_g_tf: self.t_g}), 
                            self.sess.run(self.loss, tf_dict), 
                            self.sess.run(self.gammaVar))
        
        print('L-BFGS-B')
        self.epoch = 1
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.MSEu, self.MSEf, self.MSEtrajectories, self.MSEg, self.loss, self.gammaVar],
                                loss_callback=self.loss_callback)
        
        self.epoch = 1
        print('*********************************')
        print('******* RECONSTRUCTION 2 ********')
        print('*********************************')          
        self.optimizerPrecise.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.MSEu, self.MSEf, self.MSEtrajectories, self.MSEg, self.lossPrecise, self.gammaVar],
                                loss_callback=self.loss_callback)
        
        return self.save_loss
    
    def predict(self, x, t):
        x = np.float32(x)
        t = np.float32(t)
        return np.minimum(np.maximum(self.sess.run(self.net_u(x,t)), -1), 1) # self.u_pred, {self.x_u_tf: x, self.t_u_tf: t}
    
    def predict_trajectories(self, t):
        return self.sess.run(self.x_pred, {self.t_tf: t})

        
class Unit():
    
    def __init__(self, weights=[], biases=[], weight=0):
        self.weights = weights
        self.biases = biases
        self.weight = weight
        
    def zeroUnit(self, layers):
        num_layers = len(layers)
        
        for l in range(num_layers-1):
            W = np.zeros((layers[l], layers[l+1]), dtype=np.float32)
            b = np.zeros((1, layers[l+1]), dtype=np.float32)
            self.addWeights(W)
            self.addBiases(b)
            
    def convert(self, sess):
        scalarWeights = []
        for weightTensor in self.weights:
            scalarWeights.append(sess.run(weightTensor))
        
        scalarBiases = []
        for biasTensor in self.biases:
            scalarBiases.append(sess.run(biasTensor))
        
        scalarWeight = sess.run(self.weight)
        return Unit(scalarWeights, scalarBiases, scalarWeight)
        
    def addWeights(self, weights):
        self.weights.append(weights)
        
    def addBiases(self, biases):
        self.biases.append(biases)
    
    def getWeights(self, layer=-1):
        if layer == -1:
            return self.weights
        else:
            return self.weights[layer]
    
    def getBiases(self, layer=-1):
        if layer == -1:
            return self.biases
        else:
            return self.biases[layer]
        
    def getVariables(self):
        listVar = self.weights + self.biases
        listVar.append(self.weight)
        return listVar
        
def hms( seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print('{:.0f}h {:.0f}m {:.0f}s'.format(h, m, s))

class ConfidenceNeuralNetwork():
    
    def __init__(self, x, t, u, L, Tmax, units=5, layers=(2, 1), Vf=1, 
                 gamma=0.0, Nexp=1, C=0.95, N_f=1000, N_g=100, lookahead=1):
        
        self.Nexp = Nexp
        self.Nxi = x.shape[1]
        self.confidenceFactor = student.ppf((1+C)/2, self.Nexp)
        
        x_train, t_train, u_train, X_f_train, t_g_train = self.createTrainingDataset(x, t, u, L, Tmax, N_f, N_g, lookahead)
        VfNorm = Vf*(self.ub[1] - self.lb[1]) / (self.ub[0] - self.lb[0])
        gammaNorm = gamma * 2 * (self.ub[1] - self.lb[1]) / (self.ub[0] - self.lb[0])**2
        
        
        self.neural_networks = [NeuralNetwork(x_train, t_train, u_train, X_f_train, t_g_train, layers=layers, 
                                              trajectories_layers=(1, 2*self.Nxi, 2*self.Nxi, 2*self.Nxi, 2*self.Nxi, self.Nxi),
                                              units=units, Vf=VfNorm, gamma=gammaNorm)]
        self.trainLast()
        for _ in range(Nexp):
            
            x_train, t_train, u_train, X_f_train, t_g_train= self.createTrainingDataset(x, t, u, L, Tmax, N_f, N_g, lookahead)
            
            init = self.neural_networks[-1].save()
            self.neural_networks.append(NeuralNetwork(x_train, t_train, u_train, X_f_train, t_g_train, layers=layers, 
                                              trajectories_layers=(1, 2*self.Nxi, 2*self.Nxi, 2*self.Nxi, 2*self.Nxi, self.Nxi),
                                              units=units, Vf=VfNorm, gamma=gammaNorm, 
                                              init=init[0], initWeights=init[1], initBias=init[2], alpha=0.1))
            self.trainLast()
            
    def createTrainingDataset(self, x, t, u, L, Tmax, N_f, N_g, lookahead=1):       
        
        self.lb = np.array([np.amin(x), np.amin(t)])
        self.ub = np.array([np.amax(x), np.amax(t)])
        self.lb[0], self.lb[1] = 0, 0
        
        x = 2*(x - self.lb[0])/(self.ub[0] - self.lb[0])-1
        t = 2*(t - self.lb[1])/(self.ub[1] - self.lb[1])-1
        u = 2*u-1
        
        X_f = np.array([2, lookahead*2])*lhs(2, samples=N_f)
        X_f = X_f - np.ones(X_f.shape)
        t_g = lookahead*2*lhs(1, samples=N_g)-1
        
        X_trajectories = np.array([x.reshape(-1,), (np.tile(t, (1, self.Nxi))).reshape(-1,)], dtype=np.float32).T
        X_f = np.vstack([X_f, X_trajectories])
        
        # np.random.shuffle(X_f)
        # return (t_u_shuffled, pv_u_shuffled, u_shuffled, X_f, t_trajectories, x_trajectories, t_g)
        
        return (x, t, u, X_f, t_g)

    def trainLast(self):
        start = time()
        self.neural_networks[-1].train()
        hms(time() - start)
        
    def predict(self, x, t):
        
        x = 2*(x - self.lb[0])/(self.ub[0] - self.lb[0])-1
        t = 2*(t - self.lb[1])/(self.ub[1] - self.lb[1])-1
        
        mean_value = 0
        V_value = 0
        for i in range(self.Nexp+1):
            output_value = self.neural_networks[i].predict(x, t)/2+0.5
            V_value = V_value + output_value**2
            mean_value = mean_value + output_value
        mean_value = mean_value/(self.Nexp+1)
        if self.Nexp > 0:
            V_value = abs(V_value/self.Nexp - (self.Nexp+1)*mean_value**2/self.Nexp)
            prediction_interval = self.confidenceFactor*np.sqrt(V_value*(1+1/(self.Nexp+1)))
        else:
            prediction_interval = -np.ones(mean_value.shape)
            
        return (mean_value, prediction_interval)
    
    def predict_trajectories(self, t):
        
        t = 2*(t - self.lb[1])/(self.ub[1] - self.lb[1])-1
        
        mean_value = 0
        V_value = 0
        for i in range(self.Nexp+1):
            output_value = (self.neural_networks[i].predict_trajectories(t)+1)*(self.ub[0] - self.lb[0])/2 + self.lb[0]
            V_value = V_value + output_value**2
            mean_value = mean_value + output_value
        mean_value = mean_value/(self.Nexp+1)
        if self.Nexp > 0:
            V_value = abs(V_value/self.Nexp - (self.Nexp+1)*mean_value**2/self.Nexp)
            prediction_interval = self.confidenceFactor*np.sqrt(V_value*(1+1/(self.Nexp+1)))
            prediction_interval = np.maximum(prediction_interval, 0)
            prediction_interval = np.minimum(prediction_interval, 1)
        else:
            prediction_interval = -np.ones(mean_value.shape)
        
        return (mean_value, prediction_interval)
    
    
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
        U_prediction = self.predict(xstar, tstar)
        U_mean = U_prediction[0]
        U_mean = U_mean.reshape(Nx, Nt)
        U_confidence = U_prediction[1]
        U_confidence = U_confidence.reshape(Nx, Nt)
        
        X_prediction = self.predict_trajectories(t.reshape(t.shape[0], 1))
        X_mean = X_prediction[0]
        X_mean = X_mean.reshape(Nt, self.Nxi)
        X_confidence = X_prediction[1]
        X_confidence = X_confidence.reshape(Nt, self.Nxi)

        figReconstruction = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(t, x)
        plt.pcolor(X, Y, U_mean, vmin=0.0, vmax=1.0, shading='auto')
        plt.plot(t, X_mean, color="orange")
        plt.xlabel(r'Time [s]')
        plt.ylabel(r'Position [m]')
        plt.xlim(min(t), max(t))
        plt.ylim(min(x), max(x))
        plt.colorbar()
        plt.tight_layout()
        # plt.title('Mean value')
        self.t_train = (np.tile(self.neural_networks[-1].t, (1,self.Nxi)).reshape((-1,1)) + 1)*(self.ub[1] - self.lb[1])/2 + self.lb[1]
        self.x_train = (self.neural_networks[-1].x.reshape((-1,1)) + 1)*(self.ub[0] - self.lb[0])/2 + self.lb[0]
        plt.scatter(self.t_train, self.x_train, s=0.5, c="red") 
        plt.show()
        figReconstruction.savefig('reconstruction.eps', bbox_inches='tight')
        
        
        figError = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(t, x)
        plt.pcolor(X, Y, np.abs(U_mean-u), vmin=0.0, vmax=1.0, shading='auto')
        plt.plot(t, X_mean, color="orange")
        plt.xlabel(r'Time [s]')
        plt.ylabel(r'Position [m]')
        plt.xlim(min(t), max(t))
        plt.ylim(min(x), max(x))
        plt.colorbar()
        plt.tight_layout()
        # plt.title('Absolute error')
        plt.show()
        figError.savefig('error.eps', bbox_inches='tight')
        
        
        if self.Nexp > 0:
            figConfidence = plt.figure(figsize=(7.5, 5))
            X, Y = np.meshgrid(t, x)
            plt.pcolor(X, Y, U_confidence, shading='auto')
            plt.xlabel(r'Time [s]')
            plt.ylabel(r'Position [m]')
            plt.xlim(min(t), max(t))
            plt.ylim(min(x), max(x))
            plt.colorbar()
            plt.tight_layout()
            # plt.title('Confidence interval')
            figConfidence.savefig('densityEstimated.eps', bbox_inches='tight')
            plt.show()
            return [figReconstruction, figError, figConfidence]
        else:
            return [figReconstruction, figError]