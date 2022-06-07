import Constants as CON
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import dedalus
import dedalus.public as de
import scipy.signal as sc
import seawater as sw


Nx = CON.Nx
Nz = CON.Nz
H = CON.H
L = CON.L
l = CON.l
h = CON.h
z0 = CON.z0
h_z = CON.H-z0
mu = CON.mu
sigma = CON.sigma
E_0 = (9.8*(sw.dens0(30,-2)-sw.dens0(28,-2))*z0**4) #initial potential energy at z=24
phi_0 = sw.dens0(28,-2)*np.sqrt(9.8*(sw.dens0(30,-2)-sw.dens0(28,-2))/sw.dens0(28,-2)*z0**(9))

xbasis = de.Chebyshev('x', 741, interval=(10, L-10))
zbasis = de.Chebyshev('z', 640, interval=(0, H))
domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)

def average_data(d):
    return np.mean(d[84:166]) #5000s-10000s

def stdev_data(d):
    return np.std(d[84:166])

x, z = domain.grids()
#a = ['a005', 'a009', 'a102', 'a200'] #heights
#c = ['c003', 'c100', 'c107', 'c204'] #speeds
a = ['a005', 'a095', 'a102']
c = ['c003', 'c100']
data = [[], [], [], []] #Height[Speeds]
avgs = [{}, {}, {}, {}] #a=0.5,0.9,1.2,2  Heights{MLD: [speeds], KED: [speeds], ...}
for i in range(len(a)):
    for j in range(len(c)):
        if (j == 1 and i == 0) or (j == 1 and i == 1) or j != 1:
            data[i].append(np.loadtxt('potentialdata_{}_1-890.txt'.format(a[i]+c[j]), unpack=True))

for i in range(len(a)):
    avgs[i]['MLD'] = []
    avgs[i]['MLD_stdev'] = []
    avgs[i]['KED'] = []
    avgs[i]['KED_stdev'] = []
    avgs[i]['Phi_d'] = []
    avgs[i]['Phi_d_stdev'] = []
    for j in range(len(c)):
        if (j == 1 and i == 0) or (j == 1 and i == 1) or j != 1:
            avgs[i]['MLD'].append(average_data(data[i][j][7]))
            avgs[i]['MLD_stdev'].append(stdev_data(data[i][j][8]))
            avgs[i]['KED'].append(average_data(data[i][j][6]))
            avgs[i]['KED_stdev'].append(stdev_data(data[i][j][6]))
            avgs[i]['Phi_d'].append(average_data(data[i][j][3]))
            avgs[i]['Phi_d_stdev'].append(stdev_data(data[i][j][3]))

c_axis = [0.3, 1.0, 1.7, 2.4]
colors = ['orange', 'red', 'blue', 'green']
#MLD
plt.plot(c_axis, np.ones(len(c_axis)), linestyle='dashed', color='black')
for i in range(len(a)):
    for j in range(len(c)):
        if (j == 1 and i == 0) or (j == 1 and i == 1) or j != 1:
            plt.errorbar(c_axis[j], -avgs[i]['MLD'][j]/(H-z0), yerr=avgs[i]['MLD_stdev'][j]/(H-z0), capsize=5, marker='o', linestyle='None', color=colors[i], label=a[i])    
plt.xlabel('$U/\\sqrt{{z_0 \\Delta B}}$')
plt.ylabel('(Average Mixed Layer Depth)$/z_0$')
plt.ylim(plt.ylim()[::-1])
plt.grid()
plt.legend()
plt.savefig('MLD_figure.png')
plt.clf()

#KED
for i in range(len(a)):
    for j in range(len(c)):
        if (j == 1 and i == 0) or (j == 1 and i == 1) or j != 1:
            plt.errorbar(c_axis[j], avgs[i]['KED'][j]/phi_0, yerr=avgs[i]['KED_stdev'][j]/phi_0, capsize=5, marker='o', linestyle='None', color=colors[i], label=a[i])    
plt.xlabel('$U/\\sqrt{{z_0 \\Delta B}}$')
plt.ylabel('$\\varepsilon_k/(\\rho_0 \\sqrt{{z_0^{{9}}\\Delta B}})$')
plt.grid()
plt.legend()
plt.savefig('KED_figure.png')
plt.clf()

#phi_d
for i in range(len(a)):
    for j in range(len(c)):
        if (j == 1 and i == 0) or (j == 1 and i == 1) or j != 1:
            plt.errorbar(c_axis[j], avgs[i]['Phi_d'][j]/phi_0, yerr=avgs[i]['Phi_d_stdev'][j]/phi_0, capsize=5, marker='o', linestyle='None', color=colors[i], label=a[i])    
plt.xlabel('$U/\\sqrt{{z_0 \\Delta B}}$')
plt.ylabel('$\\Phi_d/(\\rho_0 \\sqrt{{z_0^{{9}}\\Delta B}})$')
plt.grid()
plt.legend()
plt.savefig('phid_figure.png')
plt.clf()