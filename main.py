# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:17:25 2020

@author: barreau
"""

import numpy as np
# np.random.seed(12345)
import godunov as g
import reconstruction_neural_network as rn
import matplotlib.pyplot as plt
from pyDOE import lhs
import csv

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
V = lambda rho: Vf*(1-rho) # Equilibrium velocity function
F = lambda rho: Vf*(1-2*rho) # Flux function of the PDE

Vbar = Vf*(1-rhoBar) # Average speed
Lplus = Tmax*(Vbar+0.1*Vf)/1.1 # Additionnal length
Ltotal = L + Lplus

Ncar = rhoBar*rhoMax*Ltotal/1000 # Number of cars
Npv = int(Ncar*p) # Number of PV

for i in range(50):
    print("******** SIMULATION %.0f ********" % (i+1))

    # Initial position and time of probes vehicles
    xiPos = L*lhs(1, samples=Npv).reshape((Npv,))
    xiPos = np.flip(np.sort(xiPos))
    xiT = np.array([0]*Npv)

    # Godunov simulation of the PDE
    simu_godunov = g.SimuGodunov(Vf, gamma, xiPos, xiT, L=Ltotal, Tmax=Tmax, zMin=0, zMax=1, Nx=1000, rhoBar=rhoBar, rhoSigma=rhoSigma)
    rho = simu_godunov.simulation()
    simu_godunov.plot()
    axisPlot = simu_godunov.getAxisPlot()

    # collect data from PV
    x_train, t_train, rho_train = simu_godunov.getMeasurements(selectedPacket=-1,
                                                               totalPacket=-1,
                                                               noise=noise)

    trained_neural_network = rn.ReconstructionNeuralNetwork(x_train, t_train, rho_train,
                                                        Ltotal, Tmax, V, F,
                                                        N_f=7500, N_g=150)
    # [_, figError] = trained_neural_network.plot(axisPlot, rho)
    L2_error = trained_neural_network.plot(axisPlot, rho)

    with open('error_BFGS.csv', 'a', newline='', encoding='utf-8') as file:
        w = csv.writer(file)
        w.writerow([L2_error])

simu_godunov.pv.plot()

# figError.savefig('error.eps', bbox_inches='tight')
# plt.show()