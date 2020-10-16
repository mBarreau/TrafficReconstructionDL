# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:17:25 2020

@author: barreau
"""

import numpy as np
np.random.seed(12345)
import godunov as g
import reconstruction_neural_network as rn
import matplotlib.pyplot as plt
from pyDOE import lhs

#####################################
####     General parameters     #####
#####################################

Vf = 25 # Maximum car speed in m.s^-1
gamma = 0 # dissipativity coefficient (0 by default, discrepencies may occur if very small)
Tmax = 100 # simulation time
p = 1/15 # Probability that a car is a PV
L = 5000 # Length of the road
rhoBar = 0.2 # Average density of cars on the road
rhoMax = 120 # Number of vehicles per kilometer
rhoSigma = 0.6 # initial condition standard deviation
noise = False # noise on the measurements and on the trajectories
data_from_pv = True # collect data from PV or randsdomly inside the domain
V = lambda rho: Vf*(1-rho) # Equilibrium velocity function
F = lambda rho: Vf*(1-2*rho) # Flux function of the PDE


def get_probe_vehicle_data(L=-1, Tmax=-1, selectedPacket=-1, totalPacket=-1, noise=False):
    '''
    Collect data from the probe vehicles

    Parameters
    ----------
    L : float64, optional
        Space-length of the domain. Used only if L and Tmax are strictly 
        positive. 
        The default is -1.
    Tmax : float64, optional
        Time-length of the domain. Used only if L and Tmax are strictly 
        positive. 
        The default is -1.
    selectedPacket : float64, optional
        Number of measurements per packet selected. If -1 then all 
        the measurements are used. 
        If a real number between [0, 1], this is the fraction of used 
        measurements.
        Otherwise, it is the number of measurements used within a packet. 
        It must be an integer less than totalPacket. 
        The default is -1.
    totalPacket : integer, optional
        Length of a packet. If -1, there is only one packet.
        The default is -1.
    noise : boolean, optional
        If True, noise is added on the measurements. The default is False.

    Returns
    -------
    x : 2D numpy array of shape (?, N)
        space coordinate of the measurements.
    t_selected : 1D numpy array f shape (?, 1)
        time coordinate of the measurements.
    rho_meas : 2D numpy array of shape (?, N)
        density measurements.

    '''
    if L > 0 and Tmax > 0:
        N = 1000
        x_true = np.random.rand(N,1)*L
        t = np.random.rand(N,1)*Tmax
        Nt = t.shape[0]
        rho_true = simu_godunov.getDatas(x_true, t)
        Nxi = 1
    else:
        x_true, t, rho_true = simu_godunov.getMeasurements()
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
                rho_selected = np.append(rho_selected, np.reshape(rho_true[toBeSelected, k], [-1,1]), axis=1)
            except NameError:
                x_selected = np.reshape(x_true[toBeSelected, 0], [-1,1])
                t_selected = t[toBeSelected]
                rho_selected = np.reshape(rho_true[toBeSelected, 0], [-1,1])
        Nt2 = toBeSelected.shape[0]
        N = Nt2*Nxi
            

    if noise:
        noise_trajectory = np.random.normal(0, 2, N)
        if L > 0 and Tmax > 0:
            noise_trajectory = np.cumsum(noise_trajectory.reshape(Nt2, Nxi), axis=0)
        x = x_selected + noise_trajectory
        rho_meas = rho_selected + np.random.normal(0.1, 0.2, N).reshape(Nt2, Nxi)
        rho_meas = np.maximum(np.minimum(rho_meas, 1), 0)
    else:
        x = x_selected
        rho_meas = rho_selected

    return x, t_selected, rho_meas

Vbar = Vf*(1-rhoBar) # Average speed
Lplus = Tmax*(Vbar+0.1*Vf)/1.1 # Additionnal length
Ltotal = L + Lplus

Ncar = rhoBar*rhoMax*Ltotal/1000 # Number of cars
Npv = int(Ncar*p) # Number of PV

# Initial position and time of probes vehicles
xiPos = L*lhs(1, samples=Npv).reshape((Npv,))
xiPos = np.flip(np.sort(xiPos))
xiT = np.array([0]*Npv)

# Godunov simulation of the PDE
simu_godunov = g.SimuGodunov(Vf, gamma, xiPos, xiT, L=Ltotal, Tmax=Tmax,
                             zMin=0, zMax=1, Nx=1000, rhoBar=rhoBar, rhoSigma=rhoSigma)
rho = simu_godunov.simulation()
simu_godunov.plot()
axisPlot = simu_godunov.getAxisPlot()

# collect data from PV
x_train, t_train, rho_train = get_probe_vehicle_data(selectedPacket=-1, totalPacket=-1, noise=noise)
# to collect data from random points in the domain [0, L] \times [0, T], use the parameters L=L, Tmax=T 

trained_neural_network = rn.ReconstructionNeuralNetwork(x_train, t_train, rho_train, 
                                                    Ltotal, Tmax, V, F, 
                                                    N_f=7500, N_g=150)

[_, figError] = trained_neural_network.plot(axisPlot, rho)
simu_godunov.pv.plot(axisPlot[1])
plt.show()
figError.savefig('error.eps', bbox_inches='tight')


