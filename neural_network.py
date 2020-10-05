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

# np.random.seed(12345)
# tf.set_random_seed(12345)

class NeuralNetwork():

    def __init__(self, x, t, u, X_f, t_g, layers_density, layers_trajectories, 
                 units=1, Vf=1, gamma=0.0, init=[], initWeights=[], initBias=[]):

        # np.random.seed(1234)
        # tf.set_random_seed(1234)

        self.x = x
        self.t = t
        self.u = u.reshape((-1,1))

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        self.t_g = t_g
        
        self.Nxi = layers_trajectories[-1]

        self.Vf = Vf
        self.VfVar = tf.Variable(Vf, dtype=tf.float32, trainable=False)
        self.gamma = np.sqrt(gamma)
        self.gamma_var = tf.Variable(tf.truncated_normal([1,1], mean=self.gamma, stddev=0.01, dtype=tf.float32), dtype=tf.float32, trainable=True)[0,0]

        self.act_density = tf.nn.tanh
        self.weights_density, self.biases_density = self.initialize_neural_network(layers_density, init, act="tanh")
        list_var_density = self.weights_density + self.biases_density
        list_var_density.append(self.gamma_var)
        
        self.weights_trajectories, self.biases_trajectories = self.initialize_neural_network(layers_trajectories, initWeights=initWeights, initBias=initBias, act="tanh")
        self.weights_trajectoriesRelu, self.biases_trajectoriesRelu = self.initialize_neural_network(layers_trajectories, initWeights=initWeights, initBias=initBias, act="relu")
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
        # self.loss = self.MSEu + self.MSEtrajectories + 0.5*(self.MSEf + 1*self.MSEg + 0.1*tf.square(self.gamma_var))
        self.loss = self.MSEu + 0.1*self.MSEf + 0.5*self.loss_trajectories
        self.lossPrecise = self.MSEu + self.MSEf + self.loss_trajectories + 0.1*tf.square(self.gamma_var)
        
        self.optimizer_trajectories = tf.contrib.opt.ScipyOptimizerInterface(self.loss_trajectories, 
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 2000,
                                                                          'maxfun': 5000,
                                                                          'maxcor': 50,
                                                                          'maxls': 20,
                                                                          'ftol': 5.0 * np.finfo(float).eps})
        
        self.optimizerAdam = tf.train.AdamOptimizer().minimize(self.loss, var_list=list_var_density)
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=list_var_density,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 5000,
                                                                          'maxfun': 5000,
                                                                          'maxcor': 50,
                                                                          'maxls': 20,
                                                                          'ftol': 5.0 * np.finfo(float).eps})
        
        self.optimizerPrecise = tf.contrib.opt.ScipyOptimizerInterface(self.lossPrecise,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 10000,
                                                                          'maxfun': 50000,
                                                                          'maxcor': 150,
                                                                          'maxls': 75,
                                                                          'ftol': 1.0 * np.finfo(float).eps})
        

        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    def initialize_neural_network(self, layers, initWeights=[], initBias=[], act="tanh"):
        weights, biases = [], []
        
        num_layers = len(layers)
        if len(initWeights) == 0:
            initWeights = [np.nan]*num_layers
            initBias = [np.nan]*num_layers
            
        for l in range(num_layers-1):
            
            if np.isnan(initWeights[l]).any():
                initWeights[l] = np.zeros((layers[l], layers[l+1]), dtype=np.float32)
                initBias[l] = np.zeros((1, layers[l+1]), dtype=np.float32)
                
            W = self.xavier_initializer(size=[layers[l], layers[l+1]], init=initWeights[l], act=act)
            b = tf.Variable(initBias[l], dtype=tf.float32) 

            weights.append(W)
            biases.append(b)
            
        return weights, biases
    
    def xavier_initializer(self, size, init, act="tanh"):
        in_dim = size[0]
        out_dim = size[1]

        xavier_stddev = np.sqrt(2/in_dim)
        xavier_bound = np.sqrt(6/(in_dim + out_dim))
        
        if act == "relu":
            return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], mean=init, stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)
        else:
            #return tf.Variable(init + tf.random.uniform([in_dim, out_dim], minval=init-xavier_bound, maxval=init+xavier_bound, dtype=tf.float32), dtype=tf.float32)
            return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], mean=init, stddev=xavier_bound*np.sqrt(2/6), dtype=tf.float32), dtype=tf.float32)
           
    
    def neural_network(self, X, weights, biases, act=tf.nn.tanh):
        num_layers = len(weights) + 1

        # H = (X - lb) / (ub - lb)
        H = X
        for l in range(num_layers - 2):
            W, b = weights[l], biases[l]
            H = act(tf.add(tf.matmul(H, W), b))
            
        W, b = weights[-1], biases[-1]
        return tf.add(tf.matmul(H, W), b)

    def net_u(self, x, t):
        u = self.neural_network(tf.concat([x,t],1), self.weights_density, 
                                self.biases_density, act=self.act_density)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        # f = u_t + self.VfVar * (1 - 2 * u) * u_x - self.gamma_var**2 * u_xx
        f = u_t - self.VfVar * u * u_x - self.gamma_var**2 * u_xx
        return f
    
    def net_x(self, t):
        xTanh = self.neural_network(t, self.weights_trajectories, 
                                    self.biases_trajectories, act=tf.nn.tanh)
        xRelu = self.neural_network(t, self.weights_trajectoriesRelu, 
                                                 self.biases_trajectoriesRelu,
                                                 act=tf.nn.relu)
        return xTanh + xRelu
    
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
                                self.sess.run(self.gamma_var))
            self.sess.run(self.optimizerAdam, tf_dict)
            self.epoch = self.epoch + 1   
            
        self.loss_callback(self.sess.run(self.MSEu, {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}), 
                            self.sess.run(self.MSEf, {self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}), 
                            self.sess.run(self.MSEtrajectories, {self.x_tf: self.x, self.t_tf: self.t}), 
                            self.sess.run(self.MSEg, {self.t_g_tf: self.t_g}), 
                            self.sess.run(self.loss, tf_dict), 
                            self.sess.run(self.gamma_var))
        
        print('L-BFGS-B')
        self.epoch = 1
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.MSEu, self.MSEf, self.MSEtrajectories, self.MSEg, self.loss, self.gamma_var],
                                loss_callback=self.loss_callback)
        
        self.epoch = 1
        print('*********************************')
        print('******* RECONSTRUCTION 2 ********')
        print('*********************************')          
        self.optimizerPrecise.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.MSEu, self.MSEf, self.MSEtrajectories, self.MSEg, self.lossPrecise, self.gamma_var],
                                loss_callback=self.loss_callback)
        
        return self.save_loss
    
    def predict(self, x, t):
        x = np.float32(x)
        t = np.float32(t)
        return np.minimum(np.maximum(self.sess.run(self.net_u(x,t)), -1), 1) # self.u_pred, {self.x_u_tf: x, self.t_u_tf: t}
    
    def predict_trajectories(self, t):
        return self.sess.run(self.x_pred, {self.t_tf: t})