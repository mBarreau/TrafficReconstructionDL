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
data_from_pv = True # collect data from PV or randomly inside the domain
V = lambda rho: Vf*(1-rho) # Equilibrium velocity function
F = lambda rho: Vf*(1-2*rho) # Flux function of the PDE


def get_probe_vehicle_data(L=-1, Tmax=-1, selectedPacket=-1, totalPacket=-1, noise=False):
    '''
    Collect data from N probe vehicles

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
    x_selected : list of N numpy array of shape (?,1)
        space coordinate of the measurements.
    t_selected : list of N numpy array f shape (?,1)
        time coordinate of the measurements.
    rho_selected : list of N numpy array of shape (?,1)
        density measurements.

    '''
    if L > 0 and Tmax > 0:
        N = 1000
        x_true = np.random.rand(N,1)*L
        t = np.random.rand(N,1)*Tmax
        Nt = t.shape[0]
        rho_true = simu_godunov.getDatas(x_true, t)
        Nxi = 1
        
        if noise:
            x_selected = x_true+ np.random.normal(0.1, 0.2, N).reshape(-1, 1)
            rho_temp = rho_true + np.random.normal(0.1, 0.2, N).reshape(-1, 1)
            rho_selected = np.maximum(np.minimum(rho_temp, 1), 0)
        else:
            x_selected = x_true
            rho_selected = rho_true
        
    else:
        x_true, t, rho_true = simu_godunov.getMeasurements()
        Nxi = len(x_true)
        
        x_selected = []
        t_selected = []
        rho_selected = []
        for k in range(Nxi):
            
            Nt = t[k].shape[0]
            
            if totalPacket == -1:
                totalPacket = Nt
            if selectedPacket <= 0:
                selectedPacket = totalPacket
            elif selectedPacket < 1:
                selectedPacket = int(np.ceil(totalPacket*selectedPacket))
            
            nPackets = int(np.ceil(Nt/totalPacket))
            toBeSelected = np.empty((0,1), dtype=np.int)
            for i in range(nPackets):
                randomPackets = np.arange(i*totalPacket, min((i+1)*totalPacket, Nt), dtype=np.int)
                np.random.shuffle(randomPackets)
                if selectedPacket > randomPackets.shape[0]:
                    toBeSelected = np.append(toBeSelected, randomPackets[0:-1])
                else:
                    toBeSelected = np.append(toBeSelected, randomPackets[0:selectedPacket])
            toBeSelected = np.sort(toBeSelected) 
            
            if noise:
                noise_trajectory = np.random.normal(0, 2, Nt)
                noise_trajectory = np.cumsum(noise_trajectory.reshape(-1,), axis=0)
                noise_meas = np.random.normal(0.1, 0.2, toBeSelected.shape[0]).reshape(-1,)
            else:
                noise_trajectory = np.array([0]*Nt)
                noise_meas = np.array([0]*Nt)
                
            x_selected.append(np.reshape(x_true[k][toBeSelected] + noise_trajectory[toBeSelected], (-1,1)))
            rho_temp = rho_true[k][toBeSelected] + noise_meas
            rho_selected.append(np.reshape(np.maximum(np.minimum(rho_temp, 1), 0), (-1,1)))
            t_selected.append(np.reshape(t[k][toBeSelected], (-1,1)))

    return x_selected, t_selected, rho_selected

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
if data_from_pv:
    x_train, t_train, rho_train = get_probe_vehicle_data(selectedPacket=-1, totalPacket=-1, noise=noise)
else:
    x_train, t_train, rho_train = get_probe_vehicle_data(L=L, Tmax=Tmax, selectedPacket=-1, totalPacket=-1, noise=noise)

trained_neural_network = rn.ReconstructionNeuralNetwork(x_train, t_train, rho_train, 
                                                    Ltotal, Tmax, V, F, 
                                                    N_f=7500, N_g=150)

[_, figError] = trained_neural_network.plot(axisPlot, rho)
simu_godunov.pv.plot(axisPlot[1])
plt.show()
figError.savefig('error.eps', bbox_inches='tight')


