# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:17:25 2020

@author: barreau
"""

import numpy as np
np.random.seed(1234)
import godunov as g
import reconstruction_network_unit as rn
import matplotlib.pyplot as plt
from pyDOE import lhs

def get_probe_vehicle_data(L=-1, Tmax=-1, selectedPacket=-1, totalPacket=-1):
    
    if L > 0 and Tmax > 0:
        N = 1000
        x_true = np.random.rand(N,1)*L
        t = np.random.rand(N,1)*Tmax
        Nt = t.shape[0]
        u_true = simu_godunov.getDatas(x_true, t)
        Nxi = 1
    else:
        x_true, t, u_true = simu_godunov.getMeasurements()
        # x_p = np.random.rand(100,1)*L
        # t_p = np.zeros(100,1)
        # u_p = simu_godunov.getDatas(x_p, t_p)
        Nt = t.shape[0]
        Nxi = x_true.shape[-1]
        N = Nt * Nxi
        
    if totalPacket == -1:
        totalPacket = Nt
    if selectedPacket <= 0:
        selectedPacket = totalPacket
    elif selectedPacket < 1:
        selectedPacket = int(np.ceil(totalPacket*selectedPacket))

    t = t.reshape(Nt, 1) 
    nPackets = int(np.ceil(Nt/totalPacket))
    for k in range(Nxi):
        toBeSelected = np.empty((0,1), dtype=np.int)
        for i in range(nPackets):
            randomPackets = np.arange(i*totalPacket, min((i+1)*totalPacket, Nt), dtype=np.int)
            np.random.shuffle(randomPackets)
            if selectedPacket > randomPackets.shape[0]:
                toBeSelected = np.append(toBeSelected, randomPackets[0:-1])
            else:
                toBeSelected = np.append(toBeSelected, randomPackets[0:selectedPacket])
            
        toBeSelected = np.sort(toBeSelected)    
        try:
            x_selected = np.append(x_selected, np.reshape(x_true[toBeSelected, k], [-1,1]), axis=1)
            # t_selected = np.append(t_selected, t[toBeSelected], axis=1)
            u_selected = np.append(u_selected, np.reshape(u_true[toBeSelected, k], [-1,1]), axis=1)
        except NameError:
            x_selected = np.reshape(x_true[toBeSelected, 0], [-1,1])
            t_selected = t[toBeSelected]
            u_selected = np.reshape(u_true[toBeSelected, 0], [-1,1])

    
    Nt2 = toBeSelected.shape[0]
    N = Nt2*Nxi
    x = x_selected + np.random.normal(0, 1, N).reshape(Nt2, Nxi)
    u_meas = u_selected + np.random.normal(0, 0.1, N).reshape(Nt2, Nxi)
    u_meas = np.maximum(np.minimum(u_meas, 1), 0)
    
    x = x_selected
    u_meas = u_selected

    return x, t_selected, u_meas

# General parameters
Vf = 25
gamma = 0
rhoBar = 0.2
Tmax = 100
p = 1/15
L = 5000
Nexp = 0
rhoMax = 120
rhoSigma = 0.6

Vbar = Vf*(1-rhoBar) # Average speed
Lplus = Tmax*(Vbar+0.1*Vf)/1.1 # Additionnal length
Ltotal = L+Lplus

Ncar = rhoBar*rhoMax*Ltotal/1000 # Number of cars
Npv = int(Ncar*p) # Number of PV

# Initial position and time of probes vehicles
xiPos = L*lhs(1, samples=Npv).reshape((Npv,))
xiPos = np.flip(np.sort(xiPos))
xiT = np.array([0]*Npv)

simu_godunov = g.SimuGodunov(Vf, gamma, xiPos, xiT, L=Ltotal, Tmax=Tmax,
                             zMin=0, zMax=1, Nx=1000, rhoBar=rhoBar, rhoSigma=rhoSigma)

Nx, Nt, Nxi = simu_godunov.sim.Nx, simu_godunov.sim.Nt, simu_godunov.pv.Nxi
L, Tmax = simu_godunov.sim.L, simu_godunov.sim.Tmax
it = np.arange(Nt)

u = simu_godunov.simulation()

simu_godunov.plot()
axisPlot = simu_godunov.getAxisPlot()

xiArrayPlot = (simu_godunov.pv.xiArray[:, it][[-1, 0], :]*Nx/L).astype(np.int)
init1, init2 = xiArrayPlot[0,0], xiArrayPlot[-1,0]
N_init = init2 - init1

x_train, t_train, u_train = get_probe_vehicle_data(selectedPacket=0.33,totalPacket=-1)
# x_train, t_train, u_train = get_probe_vehicle_data(L=Ltotal, Tmax=Tmax)

num_hidden_layers = 8
num_nodes_per_layer = 30
layers = [2]
for _ in range(num_hidden_layers):
    layers.append(num_nodes_per_layer)
layers.append(1)

# confidence_neural_network = rn.ConfidenceNeuralNetwork(x_train, t_train, u_train, L, Tmax, Nexp=Nexp, units=50,
#                                                         layers=(2,5,5,1), gamma=gamma, Vf=Vf, N_f=10000, N_g=100) # Several units
confidence_neural_network = rn.ConfidenceNeuralNetwork(x_train, t_train, u_train, L, Tmax, Nexp=Nexp, units=1,
                                                        layers=layers, gamma=gamma, Vf=Vf, N_f=7500, N_g=150) # 1 unit
[_, figError] = confidence_neural_network.plot(axisPlot, u)
simu_godunov.pv.plot(axisPlot[1])
plt.show()
figError.savefig('error.eps', bbox_inches='tight')

