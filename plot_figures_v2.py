import json
import Constants as CON
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
#import dedalus
#import dedalus.public as de
import scipy.signal as sc
import seawater as sw
from matplotlib import colors
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import math
from uncertainties import ufloat
from scipy.ndimage.filters import gaussian_filter

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def keel(a):
    h = a*(H-z0)
    sigma = 3.9*h
    return -h*sigma**2/(sigma**2+4*(np.linspace(0, L, Nx)-l)**2)
d = 300
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
DB = 9.8*(sw.dens0(30,-2)-sw.dens0(28,-2))/sw.dens0(28,-2)
E_0 = (9.8*(sw.dens0(30,-2)-sw.dens0(28,-2))*(H-z0)**2) 
phi_0 = sw.dens0(28,-2)*np.sqrt(DB*(H-z0)**(7))
epsilon_0 = np.sqrt(DB**3*(H-z0))
t_0 = np.sqrt((H-z0)/DB)

#xbasis = de.Chebyshev('x', 1280, interval=(0, L))
#zbasis = de.Chebyshev('z', 640, interval=(0, H))
#domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)

def average_data(d, c):
    return np.mean(d[:])

def stdev_data(d, c):
    return np.std(d[:])

def stdev_log_data(d, c):
    return np.std(np.log10(d[:]))

def generate_modes(L_1, L_2, H_1, H_2):
	Nf_x = math.ceil(Nx/L*L_2)
	Ni_x = math.floor(Nx/L*L_1)
	Ni_z = math.ceil((1-H_2/H)*Nz)
	Nf_z = math.floor((1-H_1/H)*Nz)
	return Nf_x, Ni_x, Nf_z, Ni_z

def sort_rho_z(h5_file, Ni_x, Nf_x, Ni_z, Nf_z):
    with h5py.File(h5_file, mode='r') as f:
        rho = f['tasks']['rho'][0][Ni_x: Nf_x, Ni_z: Nf_z]
        rho_sort = np.reshape(-np.sort(-rho.flatten()), (Nf_x-Ni_x, Nf_z-Ni_z), order='F')
    return rho_sort

plt.rcParams.update({'font.size':14})

#x, z = domain.grids()
#a = ['a005', 'a009', 'a102', 'a200'] #heights
#c = ['c005', 'c100', 'c105', 'c200'] #speeds
sp = [[220, 260, 450, 450], [220, 260, 450, 450], [220, 260, 450, 450], [220, 260, 450, 450]]
a = ['a005', 'a095', 'a102', 'a200']
c = ['c005', 'c100', 'c105', 'c200']
labels_height = ['$\\eta=0.5$', '$\\eta=0.95$', '$\\eta=1.2$', '$\\eta=2.0$']
labels_height_Fr = ['$Fr=0.5$', '$Fr=1.0$', '$Fr=1.5$', '$Fr=2.0$']
labels_regime_up = ['Supercritical', 'LBR', 'Solitary Waves'] #Old: ['Supercritical', 'Rarefaction',  'Solitary Waves', 'Blocking']
labels_regime_down = ['Vortex Shedding','Quasi-laminar', 'Lee Waves'] #Old: ['Vortex Shedding', 'Stirred', 'Laminar Jump', 'Blocked', 'Lee Waves']
markers_labels_up = ['P', 'D', 'o']
markers_labels_down = ['*', '^', 's']
markers1 = [['D', 'D', 'D', 'P'], ['D', 'o', 'o', 'P'], ['D', 'o', 'o', 'P'], ['D', 'o', 'o', 'P']] #vortex shedding = star, bore & MSD = circle, blocking = square, bore & TD = triangle, MSD=diamond, lee = plus
markers2 = [['s', '^', '^', '*'], ['s', '^', '*', '*'], ['s', '^', '*', '*'], ['^', '^', '*', '*']] #vortex shedding = star, bore & MSD = circle, blocking = square, bore & TD = triangle, MSD=diamond, lee = plus
titles = ["a) Vortex Shedding", "b) Solitary waves & Turbulent Downstream", "c) Solitary waves & Minimum Stirring Downstream", "d) Minimum Stirring Downstream", "e) Blocking", "f) Lee waves"]
data = [[], [], [], []] #Height[Speeds]
salt_data_up = [[], [], [], []] #Height[Speeds]
salt_data_down = [[], [], [], []] #Height[Speeds]
avgs = [{}, {}, {}, {}] #a=0.5,0.9,1.2,2  Heights{MLD: [speeds], KED: [speeds], ...}
MLD_std = [[], [], [], []]
for i in range(len(a)):
    for j in range(len(c)):
        f = open('potentialdata_{0}_{1}-{2}.txt'.format(a[i]+c[j], "70", sp[i][j]), 'r')
        temp = f.readline()
        MLD_std[i].append(float(temp[:temp.find('\\')]))
        f.close()
        data[i].append(np.loadtxt('potentialdata_{0}_{1}-{2}.txt'.format(a[i]+c[j], "70", sp[i][j]), unpack=True, skiprows=2))
        
K_import_up = json.load(open('K_values_{0}-{1}.txt'.format(160, l)))
K_import_down = json.load(open('K_values_{0}-{1}.txt'.format(l, 920)))
z_mix_import_up = json.load(open('z_mix_values_{0}-{1}.txt'.format(160, l)))
z_mix_import_down = json.load(open('z_mix_values_{0}-{1}.txt'.format(l, 920)))

for i in range(len(a)):
    avgs[i]['time'] = []
    avgs[i]['MLD'] = []
    avgs[i]['MLD_stdev'] = []
    avgs[i]['KED_D'] = []
    avgs[i]['KED_D_stdev'] = []
    avgs[i]['Phi_d_D'] = []
    avgs[i]['Phi_d_D_stdev'] = []
    avgs[i]['KED_U'] = []
    avgs[i]['KED_U_stdev'] = []
    avgs[i]['Phi_d_U'] = []
    avgs[i]['Phi_d_U_stdev'] = []
    avgs[i]['K_p_U'] = []
    avgs[i]['K_p_D'] = []
    avgs[i]['K_p_D_series'] = []
    avgs[i]['K_p_U_series'] = []
    avgs[i]['z_mix_U'] = []
    avgs[i]['z_mix_D'] = []
    for j in range(len(c)):
        avgs[i]['MLD'].append(average_data(data[i][j][7], c[j]))
        avgs[i]['MLD_stdev'].append(MLD_std[i][j])
        avgs[i]['KED_D'].append(average_data(data[i][j][15], c[j]))
        avgs[i]['KED_D_stdev'].append(stdev_data(data[i][j][15], c[j]))
        avgs[i]['Phi_d_D'].append(average_data(data[i][j][13], c[j]))
        avgs[i]['Phi_d_D_stdev'].append(stdev_data(data[i][j][13], c[j]))
        avgs[i]['KED_U'].append(average_data(data[i][j][22], c[j]))
        avgs[i]['KED_U_stdev'].append(stdev_data(data[i][j][22], c[j]))
        avgs[i]['Phi_d_U'].append(average_data(data[i][j][20], c[j]))
        avgs[i]['Phi_d_U_stdev'].append(stdev_data(data[i][j][20], c[j]))
        avgs[i]['time'].append(data[i][j][0])
        avgs[i]['K_p_D_series'].append(data[i][j][16])
        avgs[i]['K_p_U_series'].append(data[i][j][23])
        # Import diffusivities
        avgs[i]['K_p_U'].append(K_import_up[a[i]+c[j]])
        avgs[i]['K_p_D'].append(K_import_down[a[i]+c[j]])
        avgs[i]['z_mix_U'].append(z_mix_import_up[a[i]+c[j]])
        avgs[i]['z_mix_D'].append(z_mix_import_down[a[i]+c[j]])

        


c_axis = [0.5, 1.0, 1.5, 2]
colors1 = ['#99c0ff', '#3385ff', '#0047b3', '#000a1a']
def MLD_downstream():
    #MLD Downstream
    plt.plot(c_axis, np.ones(len(c_axis)), linestyle='dashed', color='black')
    for i in range(len(a)):
        for j in range(len(c)):
            plt.errorbar(c_axis[j], -avgs[i]['MLD'][j]/(H-z0), yerr=avgs[i]['MLD_stdev'][j]/(H-z0), capsize=5, marker=markers2[i][j], linestyle='None', color=colors1[i], label=labels_height[i])    
            if j == len(c)-1:
                plt.plot([], [], marker='o', linestyle='None', color=colors1[i], label=labels_height[i])
    plt.xlabel('$U/\\sqrt{{z_0 \\Delta B}}$')
    plt.ylabel('(Average Mixed Layer Depth)$/z_0$')
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.ylim(plt.ylim()[::-1])
    plt.grid()
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig('MLD_figure.png', dpi=d, bbox_inches='tight')
    plt.clf()

def KED_downstream():
    #KED Downstream
    color_lines = []
    marker_lines = []
    for i in range(len(a)):
        for j in range(len(c)):
            plt.plot(c_axis[j], avgs[i]['KED_D'][j]/epsilon_0, marker=markers2[i][j], linestyle='None', color=colors1[i], label=labels_height[i])    
            #if j == len(c)-1:
            #    plt.plot([], [], marker='o', linestyle='None', color=colors1[i], label=labels[i])
    for i in range(len(a)):
        color_lines.append(plt.plot([], [], marker='X', linestyle='None', color=colors1[i], label=labels_height[i]))
    plt.plot([], [], marker='o', label=' ', color='white')
    for i in range(5):
        marker_lines.append(plt.plot([], [], marker=markers_labels_down[i], linestyle='None', color='black', label=labels_regime_down[i]))
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::1], handles[::1]))
    plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True, prop={'size': 10}, loc='upper left', ncol=2, handleheight=0.5)
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel(r'Downstream Dissipation Rate $\langle\overline{{\varepsilon_k}}\rangle_D/\xi$')
    plt.ylim(0, plt.ylim()[1])
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    plt.savefig('KED_Downstream_figure.png', dpi=d, bbox_inches='tight')
    plt.clf()

def KED_upstream():
    #KED Upstream
    color_lines = []
    marker_lines = []
    for i in range(len(a)):
        for j in range(len(c)):
            plt.plot(c_axis[j], avgs[i]['KED_U'][j]/epsilon_0, marker=markers1[i][j], linestyle='None', color=colors1[i], label=labels_height[i])    
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel(r'Upstream Dissipation Rate $\langle\overline{{\varepsilon_k}}\rangle_U/\xi$')
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    plt.ylim(0, plt.ylim()[1])
    for i in range(len(a)):
        color_lines.append(plt.plot([], [], marker='X', linestyle='None', color=colors1[i], label=labels_height[i]))
    for i in range(4):
        marker_lines.append(plt.plot([], [], marker=markers_labels_up[i], linestyle='None', color='black', label=labels_regime_up[i]))
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::1], handles[::1]))
    plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True, prop={'size': 10}, loc='upper left', ncol=2, handleheight=0.5)
    plt.savefig('KED_Upstream_figure.png', dpi=d, bbox_inches='tight')
    plt.clf()

def phi_d_upstream():
    #phi_d Upstream
    for i in range(len(a)):
        for j in range(len(c)):
            plt.errorbar(c_axis[j], avgs[i]['Phi_d_U'][j]/epsilon_0, yerr=avgs[i]['Phi_d_U_stdev'][j]/epsilon_0, capsize=5, marker=markers1[i][j], linestyle='None', color=colors1[i], label=labels_height[i])    
            if j == len(c)-1:
                plt.plot([], [], marker='o', linestyle='None', color=colors1[i], label=labels_height[i])
    plt.xlabel('$U/\\sqrt{{z_0 \\Delta B}}$')
    plt.ylabel('$\\Phi_d/(\\sqrt{{\\Delta B^3 z_0}})$')
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    plt.ylim(0, plt.ylim()[1])
    plt.title('Upstream')
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig('phid_Upstream_figure.png', dpi=d, bbox_inches='tight')
    plt.clf()

def phi_d_downstream():
    #phi_d Downstream
    for i in range(len(a)):
        for j in range(len(c)):
            plt.errorbar(c_axis[j], avgs[i]['Phi_d_D'][j]/epsilon_0, yerr=avgs[i]['Phi_d_D_stdev'][j]/epsilon_0, capsize=5, linestyle='None', marker=markers2[i][j], color=colors1[i], label=labels_height[i])    
            if j == len(c)-1:
                plt.plot([], [], marker='o', linestyle='None', color=colors1[i], label=labels_height[i])
    plt.xlabel('$U/\\sqrt{{z_0 \\Delta B}}$')
    plt.ylabel('$\\Phi_d/(\\sqrt{{\\Delta B^3 z_0}})$')
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    plt.ylim(0, plt.ylim()[1])
    plt.title('Downstream')
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig('phid_Downstream_figure.png', dpi=d, bbox_inches='tight')
    plt.clf()

def K_p_upstream():
    #K_p Upstream
    color_lines = []
    marker_lines = []
    for i in range(len(a)):
        for j in range(len(c)):
            plt.plot(c_axis[j], avgs[i]['K_p_U'][j]/mu, marker=markers1[i][j], linestyle='None', color=colors1[i], label=labels_height[i], ms=12)    
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel(r'Upstream Diapycnal Diffusivity $K_{{\rho}}^U/\mu$')
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    plt.yscale('log', nonposy='clip')
    plt.ylim(plt.ylim()[0], 1e5)
    for i in range(len(a)):
        color_lines.append(plt.plot([], [], marker='X', linestyle='None', color=colors1[i], label=labels_height[i]))
    for i in range(4):
        marker_lines.append(plt.plot([], [], marker=markers_labels_up[i], linestyle='None', color='black', label=labels_regime_up[i]))
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::1], handles[::1]))
    plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True, prop={'size': 10}, loc='upper left', ncol=2, handleheight=0.5)
    plt.title('(a) Upstream')
    plt.gcf().set_size_inches(8,6, forward=True)
    plt.savefig('Kp_Upstream_figure.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()

def K_p_downstream():
    color_lines = []
    marker_lines = []
    #K_p Downstream
    for i in range(len(a)):
        for j in range(len(c)):
            plt.plot(c_axis[j], avgs[i]['K_p_D'][j]/mu, linestyle='None', marker=markers2[i][j], color=colors1[i], label=labels_height[i], ms=12)    
    plt.xlabel('Froude number $Fr$')
    plt.ylabel(r'Downstream Diapycnal Diffusivity $K_{{\rho}}^D/\mu$')
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    plt.yscale('log')
    plt.ylim(plt.ylim()[0], 1e5)
    for i in range(len(a)):
        color_lines.append(plt.plot([], [], marker='X', linestyle='None', color=colors1[i], label=labels_height[i]))
    plt.plot([], [], marker='o', label=' ', color='white')
    for i in range(5):
        marker_lines.append(plt.plot([], [], marker=markers_labels_down[i], linestyle='None', color='black', label=labels_regime_down[i]))
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::1], handles[::1]))
    plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True, prop={'size': 10}, loc='upper left', ncol=2, handleheight=0.5)
    plt.title('(b) Downstream')
    plt.gcf().set_size_inches(8,6, forward=True)
    plt.savefig('Kp_Downstream_figure.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()

def K_p_upstream_var1():
    #K_p Upstream var1: (Fr, eta) space with colormap
    a_axis = [0.5, 0.95, 1.2, 2.0]
    color_lines = []
    marker_lines = []
    for i in range(len(a)):
        for j in range(len(c)):
            plt.scatter(c_axis[j], a_axis[i], c=np.log10(avgs[i]['K_p_D'][j]/mu), s=70, vmin=0, vmax=4, cmap='plasma', marker=markers1[i][j], label=labels_height[i], zorder=10)    
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel('Dimensionless Keel Draught $\\eta$')
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    for i in range(len(a)):
        color_lines.append(plt.plot([], [], marker='X', linestyle='None', color=colors1[i], label=labels_height[i]))
    for i in range(4):
        marker_lines.append(plt.plot([], [], marker=markers_labels_up[i], linestyle='None', color='black', label=labels_regime_up[i]))
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::1], handles[::1]))
    plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True, prop={'size': 10}, loc='upper left', ncol=2, handleheight=0.5)
    plt.title('(a) Upstream')
    plt.colorbar()
    plt.gcf().set_size_inches(8,6, forward=True)
    plt.savefig('Kp_Upstream_figure_var1.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()

def K_p_downstream_var1():
    #K_p downstream var1: (Fr, eta) space with colormap
    a_axis = [0.5, 0.95, 1.2, 2.0]
    color_lines = []
    marker_lines = []
    for i in range(len(a)):
        for j in range(len(c)):
            plt.scatter(c_axis[j], a_axis[i], c=np.log10(avgs[i]['K_p_D'][j]/mu), s=70, vmin=0, vmax=4, cmap='plasma', marker=markers2[i][j], label=labels_height[i], zorder=10)    
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel('Dimensionless Keel Draught $\\eta$')
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    for i in range(len(a)):
        color_lines.append(plt.plot([], [], marker='X', linestyle='None', color=colors1[i], label=labels_height[i]))
    for i in range(4):
        marker_lines.append(plt.plot([], [], marker=markers_labels_up[i], linestyle='None', color='black', label=labels_regime_down[i]))
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::1], handles[::1]))
    plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True, prop={'size': 10}, loc='upper left', ncol=2, handleheight=0.5)
    plt.title('(a) Downstream')
    plt.colorbar()
    plt.gcf().set_size_inches(8,6, forward=True)
    plt.savefig('Kp_Downstream_figure_var1.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()

colors2 = {'D': 'darkgreen', '^': '#663300', 's': 'red', 'o': '#ff3399', 'P': '#fcb900', '*': 'darkviolet'}

def K_p_upstream_var2():
    #K_p Upstream var2: (Fr, eta) space with marker sizes
    a_axis = [0.5, 0.95, 1.2, 2.0]
    color_lines = []
    marker_lines = []
    max_up = np.log10(avgs[-1]['K_p_U'][-1]/mu)
    for i in range(len(a)):
        for j in range(len(c)):
            marker_size = np.exp(2.0*np.log10(avgs[i]['K_p_U'][j]/mu)/max_up) * 55
            plt.scatter(c_axis[j], a_axis[i], c=np.log10(avgs[i]['K_p_D'][j]/mu), s=marker_size, vmin=0, vmax=4, cmap='plasma', marker=markers1[i][j], label=labels_height[i], zorder=10)    
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel('Dimensionless Keel Draught $\\eta$')
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    for i in range(len(a)):
        color_lines.append(plt.plot([], [], marker='X', linestyle='None', color=colors1[i], label=labels_height[i]))
    for i in range(4):
        marker_lines.append(plt.plot([], [], marker=markers_labels_up[i], linestyle='None', color='black', label=labels_regime_up[i]))
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::1], handles[::1]))
    plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True, prop={'size': 10}, loc='upper left', ncol=2, handleheight=0.5)
    plt.title('(a) Upstream')
    plt.gcf().set_size_inches(8,6, forward=True)
    plt.savefig('Kp_Upstream_figure_var2.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()

def K_p_downstream_var2():
    #K_p downstream var1: (Fr, eta) space with marker sizes
    a_axis = [0.5, 0.95, 1.2, 2.0]
    color_lines = []
    marker_lines = []
    max_down = np.log10(avgs[-1]['K_p_D'][-1]/mu)
    for i in range(len(a)):
        for j in range(len(c)):
            marker_size = np.exp(2.0*np.log10(avgs[i]['K_p_U'][j]/mu)/max_down) * 55
            plt.scatter(c_axis[j], a_axis[i], c=np.log10(avgs[i]['K_p_D'][j]/mu), s=marker_size, vmin=0, vmax=4, cmap='plasma', marker=markers2[i][j], label=labels_height[i], zorder=10)    
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel('Dimensionless Keel Draught $\\eta$')
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    for i in range(len(a)):
        color_lines.append(plt.plot([], [], marker='X', linestyle='None', color=colors1[i], label=labels_height[i]))
    for i in range(4):
        marker_lines.append(plt.plot([], [], marker=markers_labels_up[i], linestyle='None', color='black', label=labels_regime_down[i]))
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::1], handles[::1]))
    plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True, prop={'size': 10}, loc='upper left', ncol=2, handleheight=0.5)
    plt.title('(a) Downstream')
    plt.colorbar()
    plt.gcf().set_size_inches(8,6, forward=True)
    plt.savefig('Kp_Downstream_figure_var2.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()

def K_p_upstream_var3():
    #K_p Upstream var3: Horizontal axis is eta and coloring is Fr
    a_axis = [0.5, 0.95, 1.2, 2.0]
    color_lines = []
    marker_lines = []
    for i in range(len(a)):
        for j in range(len(c)):
            plt.plot(a_axis[i], avgs[i]['K_p_U'][j]/mu, marker=markers1[i][j], linestyle='None', color=colors1[j], label=labels_height_Fr[j], ms=12)    
    plt.xlabel('Dimensionless Keel Draught $\\eta$')
    plt.ylabel(r'Upstream Diapycnal Diffusivity $K_{{\rho}}^U/\mu$')
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    plt.yscale('log', nonposy='clip')
    plt.ylim(plt.ylim()[0], 1e5)
    for i in range(len(a)):
        color_lines.append(plt.plot([], [], marker='X', linestyle='None', color=colors1[i], label=labels_height_Fr[i]))
    for i in range(4):
        marker_lines.append(plt.plot([], [], marker=markers_labels_up[i], linestyle='None', color='black', label=labels_regime_up[i]))
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::1], handles[::1]))
    plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True, prop={'size': 10}, loc='upper left', ncol=2, handleheight=0.5)
    plt.title('(a) Upstream')
    plt.gcf().set_size_inches(8,6, forward=True)
    plt.savefig('Kp_Upstream_figure_var3.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()

def K_p_downstream_var3():
    #K_p Upstream var3: Horizontal axis is eta and coloring is Fr
    a_axis = [0.5, 0.95, 1.2, 2.0]
    color_lines = []
    marker_lines = []
    for i in range(len(a)):
        for j in range(len(c)):
            plt.plot(a_axis[i], avgs[i]['K_p_D'][j]/mu, marker=markers2[i][j], linestyle='None', color=colors1[j], label=labels_height_Fr[j], ms=12)    
    plt.xlabel('Dimensionless Keel Draught $\\eta$')
    plt.ylabel(r'Upstream Diapycnal Diffusivity $K_{{\rho}}^U/\mu$')
    plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    plt.yscale('log', nonposy='clip')
    plt.ylim(plt.ylim()[0], 1e5)
    for i in range(len(a)):
        color_lines.append(plt.plot([], [], marker='X', linestyle='None', color=colors1[i], label=labels_height_Fr[i]))
    for i in range(4):
        marker_lines.append(plt.plot([], [], marker=markers_labels_down[i], linestyle='None', color='black', label=labels_regime_down[i]))
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::1], handles[::1]))
    plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True, prop={'size': 10}, loc='upper left', ncol=2, handleheight=0.5)
    plt.title('(a) Downstream')
    plt.gcf().set_size_inches(8,6, forward=True)
    plt.savefig('Kp_Downstream_figure_var3.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()

def K_p_upstream_var4():
    #K_p upstream var4: (Fr, eta) space with height
    fig = plt.figure(figsize=(7, 6))
    ax = fig.gca(projection='3d')
    ax.view_init(elev=30, azim=225)
    a_axis = [0.5, 0.95, 1.2, 2.0] #eta
    block_width = 0.1
    block_depth = 0.1
    for i in range(len(a)):
        for j in range(len(c)):
            ax.bar3d(c_axis[j], a_axis[i], 0, block_width, block_depth, np.log10(avgs[i]['K_p_U'][j]/mu), color=colors2[markers1[i][j]], edgecolor='k', shade=True)
    plt.plot([], [], marker='s', linestyle='None', color=colors2['D'], mec='k', label='LBR')
    plt.plot([], [], marker='s', linestyle='None', color=colors2['o'], mec='k', label='Solitary Waves')
    plt.plot([], [], marker='s', linestyle='None', color=colors2['P'], mec='k', label='Supercritical')
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 11}, fancybox=True, shadow=True)
    ax.set_xlabel('Fr')
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('$\\eta$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\log (\overline{K}^U / \mu)$      ')
    ax.set_zlim(0, 4.5)
    fig.savefig('Kp_Upstream_figure_var4.pdf', format='pdf')
    plt.clf()

def K_p_downstream_var4():
    #K_p downstream var4: (Fr, eta) space with height
    fig = plt.figure(figsize=(7, 6))
    ax = fig.gca(projection='3d')
    ax.view_init(elev=30, azim=225)
    a_axis = [0.5, 0.95, 1.2, 2.0] #eta
    block_width = 0.1
    block_depth = 0.1
    for i in range(len(a)):
        for j in range(len(c)):
            ax.bar3d(c_axis[j], a_axis[i], 0, block_width, block_depth, np.log10(avgs[i]['K_p_D'][j]/mu), color=colors2[markers2[i][j]], edgecolor='k', shade=True)
    plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], mec='k', label='Lee Waves')
    plt.plot([], [], marker='s', linestyle='None', color=colors2['^'], mec='k', label='Quasi-laminar')
    plt.plot([], [], marker='s', linestyle='None', color=colors2['*'], mec='k', label='Vortex Shedding')
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 11}, fancybox=True, shadow=True)
    ax.set_xlabel('Fr')
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('$\\eta$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\log (\overline{K}^D / \mu)$       ')
    ax.set_zlim(0, 4.5)
    fig.savefig('Kp_Downstream_figure_var4.pdf', format='pdf')
    plt.clf()

def K_p_subplots_4():
    #var4, both up and downstream
    fig = plt.figure(figsize=(8,12))
    a_axis = [0.5, 0.95, 1.2, 2.0] #eta
    block_width, block_depth = 0.1, 0.1
    
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    ax.view_init(elev=30, azim=225)
    for i in range(len(a)):
        for j in range(len(c)):
            ax.bar3d(c_axis[j], a_axis[i], 0, block_width, block_depth, np.log10(avgs[i]['K_p_U'][j]/mu), color=colors2[markers1[i][j]], edgecolor='k', shade=True)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['D'], mec='k', label='LBR')
    ax.plot([], [], marker='s', linestyle='None', color=colors2['o'], mec='k', label='Solitary Waves')
    ax.plot([], [], marker='s', linestyle='None', color=colors2['P'], mec='k', label='Supercritical')
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 13}, fancybox=True, shadow=True)
    ax.set_xlabel('Fr')
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('$\\eta$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\log (\overline{K}^U / \mu)$        ')
    ax.set_zlim(0, 4.5)
    ax.set_title('(a)', loc='left')

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    ax.view_init(elev=30, azim=225)
    for i in range(len(a)):
        for j in range(len(c)):
            ax.bar3d(c_axis[j], a_axis[i], 0, block_width, block_depth, np.log10(avgs[i]['K_p_D'][j]/mu), color=colors2[markers2[i][j]], edgecolor='k', shade=True)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['s'], mec='k', label='Lee Waves')
    ax.plot([], [], marker='s', linestyle='None', color=colors2['^'], mec='k', label='Quasi-laminar')
    ax.plot([], [], marker='s', linestyle='None', color=colors2['*'], mec='k', label='Vortex Shedding')
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 13}, fancybox=True, shadow=True)
    ax.set_xlabel('Fr')
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('$\\eta$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\log (\overline{K}^D / \mu)$         ')
    ax.set_zlim(0, 4.5)
    ax.set_title('(b)', loc='left')

    plt.tight_layout()
    fig.savefig('Kp_subplots_var4.pdf', format='pdf')
    plt.clf()

def K_p_upstream_var5():
    f#K_p upstream var5: (Fr, eta) space with height and marker for regime
    fig = plt.figure(figsize=(7, 6))
    ax = fig.gca(projection='3d')
    ax.view_init(elev=30, azim=225)
    a_axis = [0.5, 0.95, 1.2, 2.0] #eta
    for i in range(len(a)):
        for j in range(len(c)):
            line_y = np.linspace(a_axis[i], 2.25, 50)
            line_x = np.full(shape=line_y.shape, fill_value=c_axis[j])
            line_z = np.full(shape=line_y.shape, fill_value=np.log10(avgs[i]['K_p_U'][j]/mu))
            plt.plot(line_x, line_y, line_z, color='gray', ls='dotted')
            markerline, stemlines, baseline = ax.stem([c_axis[j]], [a_axis[i]], [np.log10(avgs[i]['K_p_U'][j]/mu)], linefmt=colors2[markers1[i][j]], markerfmt=markers1[i][j])
            markerline.set_markeredgecolor(colors2[markers1[i][j]])
            markerline.set_markerfacecolor(colors2[markers1[i][j]])
            markerline.set_markersize(10)
    plt.plot([], [], marker='D', linestyle='None', color=colors2['D'], label='LBR')
    plt.plot([], [], marker='o', linestyle='None', color=colors2['o'], label='Solitary Waves')
    plt.plot([], [], marker='P', linestyle='None', color=colors2['P'], label='Supercritical')
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 11}, fancybox=True, shadow=True)
    ax.set_xlabel('Fr')
    ax.set_xlim(0.4, 2.1)
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('$\\eta$')
    ax.set_ylim(0.4, 2.1)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\log (\overline{K}^U / \mu)$       ')
    ax.set_zlim(0, 4.5)
    fig.savefig('Kp_Upstream_figure_var5.pdf', format='pdf')
    plt.clf()

def K_p_downstream_var5():
    #K_p downstream var5: (Fr, eta) space with height and marker for regime
    fig = plt.figure(figsize=(7, 6))
    ax = fig.gca(projection='3d')
    ax.view_init(elev=30, azim=225)
    a_axis = [0.5, 0.95, 1.2, 2.0] #eta
    for i in range(len(a)):
        for j in range(len(c)):
            line_y = np.linspace(a_axis[i], 2.25, 50)
            line_x = np.full(shape=line_y.shape, fill_value=c_axis[j])
            line_z = np.full(shape=line_y.shape, fill_value=np.log10(avgs[i]['K_p_D'][j]/mu))
            plt.plot(line_x, line_y, line_z, color='gray', ls='dotted')
            markerline, stemlines, baseline = ax.stem([c_axis[j]], [a_axis[i]], [np.log10(avgs[i]['K_p_D'][j]/mu)], linefmt=colors2[markers2[i][j]], markerfmt=markers2[i][j])
            markerline.set_markeredgecolor(colors2[markers2[i][j]])
            markerline.set_markerfacecolor(colors2[markers2[i][j]])
            markerline.set_markersize(10)
    plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], label='Lee Waves')
    plt.plot([], [], marker='^', linestyle='None', color=colors2['^'], label='Quasi-laminar')
    plt.plot([], [], marker='*', linestyle='None', color=colors2['*'], label='Vortex Shedding')
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 11}, fancybox=True, shadow=True)
    ax.set_xlabel('Fr')
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('$\\eta$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\log (\overline{K}^D / \mu)$      ')
    ax.set_zlim(0, 4.5)
    fig.savefig('Kp_Downstream_figure_var5.pdf', format='pdf')
    plt.clf()

"""
def regime_upstream():
    #vortex shedding = star, bore & MSD = circle, blocking = square, bore & TD = triangle, MSD=diamond, lee = plus
    #Regime layout upstream
    for i in range(len(a)):
        for j in range(len(c)):
            plt.errorbar(c_axis[j], [0.5, 0.95, 1.2, 2][i], capsize=5, linestyle='None', marker=markers1[i][j], color=colors2[markers1[i][j]])    
    plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], label='Blocked')
    plt.plot([], [], marker='o', linestyle='None', color=colors2['o'], label='Solitary waves')
    plt.plot([], [], marker='D', linestyle='None', color=colors2['D'], label='Rarefaction')
    plt.plot([], [], marker='d', linestyle='None', color=colors2['d'], label='Supercritical')
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel('Dimensionless Keel Depth $\\eta$')
    plt.xticks([0.5, 1, 1.5, 2])
    plt.yticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    plt.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(0.944, 0.73), prop={'size': 11}, fancybox=True, shadow=True)
    plt.savefig('regime_figure_upstream.png', dpi=d, bbox_inches='tight')
    plt.clf()

def regime_downstream():
    #Regime layout Downstream
    for i in range(len(a)):
        for j in range(len(c)):
            plt.errorbar(c_axis[j], [0.5, 0.95, 1.2, 2][i], capsize=5, linestyle='None', marker=markers2[i][j], color=colors2[markers2[i][j]])    
    plt.plot([], [], marker='P', linestyle='None', color=colors2['P'], label='Lee waves')
    plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], label='Blocking')
    plt.plot([], [], marker='p', linestyle='None', color=colors2['p'], label='Stirred')
    plt.plot([], [], marker='^', linestyle='None', color=colors2['^'], label='Laminar jump')
    plt.plot([], [], marker='*', linestyle='None', color=colors2['*'], label='Vortex shedding')
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel('Dimensionless Keel Depth $\\eta$')
    plt.xticks([0.5, 1, 1.5, 2])
    plt.yticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.grid()
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    plt.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(0.944, 0.73), prop={'size': 11}, fancybox=True, shadow=True)
    plt.savefig('regime_figure_downstream.png', dpi=d, bbox_inches='tight')
    plt.clf()
"""
def joint_regime():
    #Joint regime layout 
    shift = 0.032
    ms = {'o': 17.5, 'D': 15, 'P': 18, 's': 17, '^':18, '*':20.5}
    for i in range(len(a)):
        for j in range(len(c)):
            plt.plot(c_axis[j]-shift, [0.5, 0.95, 1.2, 2][i], markeredgecolor='k', linestyle='None', marker=markers1[i][j], color=colors2[markers1[i][j]], ms=ms[markers1[i][j]], zorder=10)    
#    line4, = plt.plot([], [], marker='D', linestyle='None', color=colors2['D'], label='LBR', ms=11)
    line3, = plt.plot([], [], marker='o', markeredgecolor='k', linestyle='None', color=colors2['o'], label='Solitary waves', ms=11)
    line2, = plt.plot([], [], marker='D', markeredgecolor='k', linestyle='None', color=colors2['D'], label='LBR', ms=11)
    line1, = plt.plot([], [], marker='P', markeredgecolor='k', linestyle='None', color=colors2['P'], label='Supercritical', ms=11)
    for i in range(len(a)):
        for j in range(len(c)):
            plt.plot(c_axis[j]+shift, [0.5, 0.95, 1.2, 2][i], markeredgecolor='k', linestyle='None', marker=markers2[i][j], color=colors2[markers2[i][j]], ms=ms[markers2[i][j]], zorder=10)    
    line9, = plt.plot([], [], marker='s', markeredgecolor='k', linestyle='None', color=colors2['s'], label='Lee Waves', ms=11)
#    line8, = plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], label='Blocked', ms=11)
    line7, = plt.plot([], [], marker='^', markeredgecolor='k', linestyle='None', color=colors2['^'], label='Quasi-laminar', ms=11)
#    line6, = plt.plot([], [], marker='p', linestyle='None', color=colors2['p'], label='Stirred', ms=11)
    line5, = plt.plot([], [], marker='*', markeredgecolor='k', linestyle='None', color=colors2['*'], label='Vortex Shedding', ms=11)
    first_legend = plt.legend(handles=[line1, line2, line3], loc='center right', bbox_to_anchor=(1.457, 0.65), prop={'size': 11}, fancybox=True, shadow=True)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=[line5, line7, line9], loc='center right', bbox_to_anchor=(1.48, 0.30), prop={'size': 11}, fancybox=True, shadow=True)
    plt.xlabel('$Fr$')
    plt.ylabel('$\\eta$    ', rotation=False)
    plt.xticks([0.5, 1, 1.5, 2])
    plt.yticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.text(2.525, 1.67, 'Upstream', ha='center', va='center')
    plt.text(2.53, 1.08, 'Downstream', ha='center', va='center')
    plt.grid()
    plt.gcf().set_size_inches(8,6, forward=True)
    plt.gca().set_aspect(1.3)
    plt.savefig('regime_layout.pdf', format='pdf', bbox_inches='tight')
    plt.clf()
"""
def joint_regime_ms():
    #Joint regime layout with ms
    shift = 0
    max_up = np.log10(avgs[-1]['K_p_U'][-1]/mu)
    max_down = np.log10(avgs[-1]['K_p_D'][-1]/mu)
    for i in range(len(a)):
        for j in range(len(c)):
            marker_size = np.exp(2.0*np.log10(avgs[i]['K_p_U'][j]/mu)/max_up) * 2.8
            shift += marker_size/8000
            plt.errorbar(c_axis[j]-shift, [0.5, 0.95, 1.2, 2][i], capsize=5, linestyle='None', marker=markers1[i][j], color=colors2[markers1[i][j]], ms=marker_size)    
    line4, = plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], label='Blocking', ms=11)
    line3, = plt.plot([], [], marker='o', linestyle='None', color=colors2['o'], label='Solitary waves', ms=11)
    line2, = plt.plot([], [], marker='D', linestyle='None', color=colors2['D'], label='Rarefaction', ms=11)
    line1, = plt.plot([], [], marker='P', linestyle='None', color=colors2['P'], label='Supercritical', ms=11)
    for i in range(len(a)):
        for j in range(len(c)):
            marker_size = np.exp(2.0*np.log10(avgs[i]['K_p_D'][j]/mu)/max_down) * 2.8
            shift += marker_size/8000
            plt.errorbar(c_axis[j]+shift, [0.5, 0.95, 1.2, 2][i], capsize=5, linestyle='None', marker=markers2[i][j], color=colors2[markers2[i][j]], ms=marker_size)    
    line9, = plt.plot([], [], marker='d', linestyle='None', color=colors2['d'], label='Lee waves', ms=11)
    line8, = plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], label='Blocked', ms=11)
    line7, = plt.plot([], [], marker='^', linestyle='None', color=colors2['^'], label='Laminar Jump', ms=11)
    line6, = plt.plot([], [], marker='p', linestyle='None', color=colors2['p'], label='Stirred', ms=11)
    line5, = plt.plot([], [], marker='*', linestyle='None', color=colors2['*'], label='Vortex shedding', ms=11)
    first_legend = plt.legend(handles=[line1, line2, line3, line4], loc='center right', bbox_to_anchor=(1.385, 0.73), prop={'size': 11}, fancybox=True, shadow=True)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=[line5, line6, line7, line8, line9], loc='center right', bbox_to_anchor=(1.4, 0.25), prop={'size': 11}, fancybox=True, shadow=True)
    plt.xlabel('Froude Number $Fr$ (Keel speed / Stratification strength)')
    plt.ylabel('Dimensionless Keel Depth $\\eta$')
    plt.xticks([0.5, 1, 1.5, 2])
    plt.yticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    plt.text(2.29, 1.87, 'Upstream')
    plt.text(2.25, 1.12, 'Downstream')
    plt.grid()
    plt.savefig('regime_layout_ms.png', dpi=d, bbox_inches='tight')
    plt.clf()

def Fr(z0, Si, Sf, U=0.2):
        dB = 9.8*(sw.dens0(Sf, -2) - sw.dens0(Si, -2))/sw.dens0(Si, -2)
        return U/np.sqrt(z0*dB)
def eta(z0, h):
    return h/z0
def sigma_rho(S, S_sigma):
    return S_sigma*np.sqrt((0.8)**2+9/4*(0.005)**2*S+4*(0.0004)**2*S**2)
def sigma_Fr(z0, Si, Sf, z0_sigma, Si_sigma, Sf_sigma, U=0.2):
    rhoi_sigma = sigma_rho(Si, Si_sigma)
    rhof_sigma = sigma_rho(Sf, Sf_sigma)
    rhoi = ufloat(sw.dens0(Si, -2), rhoi_sigma)
    rhof = ufloat(sw.dens0(Sf, -2), rhof_sigma)
    z0u = ufloat(z0, z0_sigma)
    dB = 9.8*(rhof-rhoi)/rhoi
    #dB = 9.8*(sw.dens0(Sf, -2) - sw.dens0(Si, -2))/sw.dens0(Si, -2)
    #dB_sigma = dB*np.sqrt((rhoi_sigma/sw.dens0(Si,-2))**2+(rhoi_sigma**2+rhof_sigma**2)/(sw.dens0(Sf,-2)-sw.dens0(Si,-2))**2)
    Fr = U/(z0u*dB)**0.5
    #z0dB_sigma = z0*dB*np.sqrt((z0_sigma/z0)**2+(dB_sigma/dB)**2)
    print(Fr.std_dev)
    return Fr.std_dev
    #return U/2*(z0*dB)**(-3/2)*z0dB_sigma

def arctic_plot(h, color1, color2):
    S_values = {'Chukchi Sea': {'Si': 29.1, 'Si_u': 1.00, 'Sf': 30.1, 'Sf_u': 0.1}, 
                'Southern Beaufort Sea': {'Si': 28.0, 'Si_u': 3.8, 'Sf': 30.5, 'Sf_u': 2.5}, 
                'Canada Basin': {'Si': 27.2, 'Si_u': 1.8, 'Sf': 30.1, 'Sf_u': 0.1}, 
                'Eurasian Basin': {'Si': 33.4, 'Si_u': 0.8, 'Sf': 33.8, 'Sf_u': 0.7}, 
                'Barents Sea': {'Si': 33.1, 'Si_u': 0.3, 'Sf': 34.5, 'Sf_u': 0.03}} #[Si, Si_error, Sf, Sf_error]
    z0_values = {'Chukchi Sea': {'z0': 12.3, 'z0_u': 4}, 
                'Southern Beaufort Sea': {'z0': 8.5, 'z0_u': 4.5}, 
                'Canada Basin': {'z0': 8.9, 'z0_u': 3.9},  
                'Eurasian Basin': {'z0': 22.3, 'z0_u': 11.3}, 
                'Barents Sea': {'z0': 17.7, 'z0_u': 12.2}}
    S_trends = {'Chukchi Sea': {'Si_t': 0.02, 'Sf_t': -0.07}, #[Sf_trend, Si_trend]
                'Southern Beaufort Sea': {'Si_t': 0.29, 'Sf_t': -0.04}, 
                'Canada Basin': {'Si_t': -0.11, 'Sf_t': -0.19}, 
                'Eurasian Basin': {'Si_t': -0.05, 'Sf_t': -0.07}, 
                'Barents Sea': {'Si_t': 0.02, 'Sf_t': 0.02,}} #Use summer trend for winter
    z0_trends = {'Chukchi Sea': {'z0_t': -0.43}, #[z0_winter_trend, z0_sumer_trend], using Winter trend
                'Southern Beaufort Sea': {'z0_t': 0.33}, 
                'Canada Basin': {'z0_t': -0.33}, 
                'Eurasian Basin': {'z0_t': -0.19}, 
                'Barents Sea': {'z0_t': 0.51}} #Barrents taken from ice free summer
    labels_region = {'Chukchi Sea': '*1', 'Southern Beaufort Sea': '2', 'Canada Basin': '3', 'Eurasian Basin': '4', 'Barents Sea': '*5'}
    #Summer
    for key in S_values.keys():
        print(key)
        value_Fr = Fr(z0_values[key]['z0'], S_values[key]['Si'], S_values[key]['Sf'])
        value_eta = eta(z0_values[key]['z0'], h=h)
        Fr_er = sigma_Fr(z0_values[key]['z0'], S_values[key]['Si'], S_values[key]['Sf'], z0_values[key]['z0_u'], S_values[key]['Si_u'], S_values[key]['Sf_u'])
        eta_er = eta(z0_values[key]['z0'], h=h)/z0_values[key]['z0']*z0_values[key]['z0_u']
        plt.gca().add_patch(patches.FancyBboxPatch(xy=(value_Fr-Fr_er, value_eta-eta_er), width=2*Fr_er, height=2*eta_er, linewidth=1, color=color2, fill='false', mutation_scale=0.05, alpha=0.15))
        scale = 1
        years = 5
        if key == 'Barents Sea': # Use winter salinity trend for summer salinity since summer is unavail
            Fr_diff = Fr(z0_values[key]['z0']+years*z0_trends[key]['z0_t'], S_values[key]['Si']+years*S_trends[key]['Si_t'], S_values[key]['Sf']+years*S_trends[key]['Sf_t'], U=0.2*(1+0.009*years))
            eta_diff = eta(z0_values[key]['z0']+years*z0_trends[key]['z0_t'], h=h)
            plt.arrow(x=value_Fr, y=value_eta, dx=Fr_diff-value_Fr, dy=eta_diff-value_eta, linestyle='-', linewidth=2.8, length_includes_head=True, zorder=3, head_width=0.03)
        elif key == 'Chukchi Sea': # Use winter MLD trend for summer MLD since summer is unavail
            Fr_diff = Fr(z0_values[key]['z0']+years*z0_trends[key]['z0_t'], S_values[key]['Si']+years*S_trends[key]['Si_t'], S_values[key]['Sf']+years*S_trends[key]['Sf_t'], U=0.2*(1+0.009*years))
            eta_diff = eta(z0_values[key]['z0']+years*z0_trends[key]['z0_t'], h=h)
            plt.arrow(x=value_Fr, y=value_eta, dx=Fr_diff-value_Fr, dy=eta_diff-value_eta, linestyle='-', linewidth=2.8, length_includes_head=True, zorder=3, head_width=0.03)
        else:
            Fr_diff = Fr(z0_values[key]['z0']+years*z0_trends[key]['z0_t'], S_values[key]['Si']+years*S_trends[key]['Si_t'], S_values[key]['Sf']+years*S_trends[key]['Sf_t'], U=0.2*(1+0.009*years))
            eta_diff = eta(z0_values[key]['z0']+years*z0_trends[key]['z0_t'], h=h)
            plt.arrow(x=value_Fr, y=value_eta, dx=Fr_diff-value_Fr, dy=eta_diff-value_eta, linewidth=2.8, length_includes_head=True, zorder=3, head_width=0.03)
        plt.text(value_Fr, value_eta, labels_region[key], fontsize=9.5, weight='bold', ha='center', va='center', zorder=10)
        plt.plot(value_Fr, value_eta, marker='s', color=color1, ms=13.5, zorder=6)
    

def joint_regime_arctic():
    #Joint regime Arctic layout 
    shift = 0
    max_up = np.log10(avgs[-1]['K_p_U'][-1]/mu)
    max_down = np.log10(avgs[-1]['K_p_D'][-1]/mu)
    for i in range(len(a)):
        for j in range(len(c)):
            marker_size = np.exp(2.0*np.log10(avgs[i]['K_p_U'][j]/mu)/max_up) * 2.8
            shift += marker_size/8000
            print(marker_size)
            plt.errorbar(c_axis[j]-shift, [0.5, 0.95, 1.2, 2][i], capsize=5, linestyle='None', marker=markers1[i][j], color=colors2[markers1[i][j]], ms=marker_size)    
    line4, = plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], label='Blocking', ms=11)
    line3, = plt.plot([], [], marker='o', linestyle='None', color=colors2['o'], label='Solitary waves', ms=11)
    line2, = plt.plot([], [], marker='D', linestyle='None', color=colors2['D'], label='Rarefaction', ms=11)
    line1, = plt.plot([], [], marker='P', linestyle='None', color=colors2['P'], label='Supercritical', ms=11)
    for i in range(len(a)):
        for j in range(len(c)):
            marker_size = np.exp(2.0*np.log10(avgs[i]['K_p_D'][j]/mu)/max_down) * 2.8
            shift += marker_size/8000
            plt.errorbar(c_axis[j]+shift, [0.5, 0.95, 1.2, 2][i], capsize=5, linestyle='None', marker=markers2[i][j], color=colors2[markers2[i][j]], ms=marker_size)    
    line9, = plt.plot([], [], marker='d', linestyle='None', color=colors2['d'], label='Lee waves', ms=11)
    line8, = plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], label='Blocked', ms=11)
    line7, = plt.plot([], [], marker='^', linestyle='None', color=colors2['^'], label='Laminar Jump', ms=11)
    line6, = plt.plot([], [], marker='p', linestyle='None', color=colors2['p'], label='Stirred', ms=11)
    line5, = plt.plot([], [], marker='*', linestyle='None', color=colors2['*'], label='Vortex shedding', ms=11)
    first_legend = plt.legend(handles=[line1, line2, line3, line4], loc='center right', bbox_to_anchor=(1.305, 0.73), prop={'size': 11}, fancybox=True, shadow=True)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=[line5, line6, line7, line8, line9], loc='center right', bbox_to_anchor=(1.32, 0.25), prop={'size': 11}, fancybox=True, shadow=True)
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel('Dimensionless Keel Draught $\\eta$')
    plt.text(2.90, 2.15, 'Upstream', ha='center', va='center')
    plt.text(2.90, 1, 'Downstream', ha='center', va='center')
    plt.xlim(0,2.7)
    plt.ylim(0,2.7)
    plt.grid(zorder=0)
    arctic_plot(h=7.45, color1='#cc0909', color2='red')
    arctic_plot(h=7.45*2.5, color1='cyan', color2='cyan')
    #Arctic stuff
    # Winter data is March data, likewise summer is July. All std and averages are taken from these two months
    # ML depth is ice covered July data
    # Winter ML salinity is April ice covered data
    # Summer ML salinity is July ice covered data
    # We reject Makaraov data as PFW did
    #plt.text(Fr1-0.03, eta1+0.01, '*', fontsize=8, weight='bold', ha='center', va='center', zorder=10)
    #plt.imshow([[0.6,0.6], [0.7, 0.7], [0.8, 0.8]], interpolation='bicubic', vmin=0.6, vmax=0.8, extent=(0, 2.5, 0, 2.5), alpha=0.3)
    plt.gcf().set_size_inches(8,6, forward=True)
    plt.savefig('regime_layout_regional.pdf', format='pdf', bbox_inches='tight')
    plt.clf()


#Heat maps
xbasis = de.Fourier('x', 1280, interval=(0, 640))
zbasis = de.Fourier('z', 640, interval=(0, 80))
domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)

class CyclicNormalize(colors.Normalize):
    def __init__(self, cmin=0, cmax=1, vmin=0, vmax=1, clip=False):
        self.cmin = cmin
        self.cmax = cmax
        colors.Normalize.__init__(self, vmin, vmax, clip=clip)

    def __call__(self, value, clip=False):
        x, y = [self.cmin, self.cmax], [0, 1]
        return np.ma.masked_array(np.interp(value, x, y, period=self.cmax - self.cmin))
 
def methods():
    with h5py.File('data-mixingsim-Test-00/data-mixingsim-Test-00_s1.h5', mode='r') as f:
        rho = f['tasks']['rho'][0]
        fig_j, ax_j = plt.subplots(figsize=(12,12))
        im_j = ax_j.imshow(np.transpose(rho), vmin=np.min(rho)-0.5, vmax=np.max(rho)+0.5, cmap='viridis', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        ax_j.set_yticks([-10, -8, -6, -4, -2, 0])
        ax_j.set_yticklabels(['10', '8', '6', '4', '2', '0'])
        plt.xlabel('$x/z_0$')
        plt.ylabel('$z/z_0$')
        plt.ylim(-6, 0)
        plt.xlim(60, 78)
        x = np.linspace(0, L, Nx)
        #keel = -h*sigma**2/(sigma**2+4*(x-l)**2)
        plt.fill_between(x/(H-z0), 0, keel(2)/(H-z0), facecolor="white")
        plt.plot(x/(H-z0), keel(2)/(H-z0), linewidth=0.5, color='black')
        plt.tight_layout()
        fig_j.set_dpi(d)
        plt.savefig('methods.png', bbox_inches='tight')
        plt.clf() 

def K_p_time():
    a_s = ['005', '095', '102', '200']
    c_s = ['005', '100', '105', '200']
    colors3 = {"200": "#FF1300", "102": "#0CCAFF", "095": "#29E91F", "005": "#a67acf"}
    styles = {"200": "solid", "105": (0, (1,1)), "100": "dashed", "005": (0, (3, 1, 1, 1))}
    fig, axs = plt.subplots(3,2, figsize=(8, 6))
    order = [[(0, 3), (1, 3), (2, 3), (3, 3)], [(1, 2), (2, 2), (3, 2)], [(1, 1), (2, 1), (3, 1)], [(0, 1), (0, 2)], [(3, 0)], [(0,0), (1, 0), (2, 0)]] #Regime[(a,c)]
    for i, ax in enumerate(fig.axes):
        for j in range(len(order[i])):
            ax.plot(avgs[order[i][j][0]]['time'][order[i][j][1]]/t_0, np.log10(avgs[order[i][j][0]]['K_p_U_series'][order[i][j][1]]/mu), color=colors3[a_s[order[i][j][0]]], linestyle=styles[c_s[order[i][j][1]]])
            ax.set_title(titles[i])
            ax.grid(True)
            if i % 2 == 0:
                ax.set_ylabel(r'$\log_{{10}}(\langle \overline{{K_\rho}}\rangle/\mu)$')
            if i > 3:
                ax.set_xlabel('$t/t_0$')  
    plt.suptitle('Upstream', y=1.05)
    plt.tight_layout()
    plt.savefig('Upstream_K_p_time_figure.png', bbox_inches='tight', dpi=d)
    plt.clf()

    fig, axs = plt.subplots(3,2, figsize=(8, 6))
    order = [[(0, 3), (1, 3), (2, 3), (3, 3)], [(1, 2), (2, 2), (3, 2)], [(1, 1), (2, 1), (3, 1)], [(0, 1), (0, 2)], [(3, 0)], [(0,0), (1, 0), (2, 0)]] #Regime[(a,c)]
    for i, ax in enumerate(fig.axes):
        for j in range(len(order[i])):
            ax.plot(avgs[order[i][j][0]]['time'][order[i][j][1]]/t_0, np.log10(avgs[order[i][j][0]]['K_p_D_series'][order[i][j][1]]/mu), color=colors3[a_s[order[i][j][0]]], linestyle=styles[c_s[order[i][j][1]]])
            ax.set_title(titles[i])
            ax.grid(True)
            if i % 2 == 0:
                ax.set_ylabel(r'$\log_{{10}}(\langle \overline{{K_\rho}}\rangle/\mu)$')
            if i > 3:
                ax.set_xlabel('$t/t_0$')  
    plt.suptitle('Downstream', y=1.05)
    plt.tight_layout()
    plt.savefig('Downstream_K_p_time_figure.png', bbox_inches='tight', dpi=d)
    plt.clf()

def test_heatmap():
    with h5py.File('regime_files/data-mixingsim-a005c005-00_s180.h5', mode='r') as f:
        L_1 = 160
        L_2 = 920
        Nf_x, Ni_x, Nf_z, Ni_z = generate_modes(L_1, L_2, 0, 80)
        xbasis = de.Chebyshev('x', Nf_x-Ni_x, interval=(L_1, L_2))
        zbasis = de.Chebyshev('z', Nf_z-Ni_z, interval=(0, 80))
        domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)
        x, z = domain.grids(domain.dealias)
        integrand1 = domain.new_field()
        integrand1.set_scales(domain.dealias)
        rho_ref = sort_rho_z('regime_files/data-mixingsim-a200c200-00_s320.h5', Ni_x, Nf_x, Ni_z, Nf_z)
        rho = f['tasks']['rho'][0][Ni_x:Nf_x, Ni_z:Nf_z]
        deriv2 = np.gradient(rho_ref, z[0], axis=1, edge_order=1)
        rho_z = np.gradient(rho, z[0], axis=1, edge_order=2)
        rho_x = np.gradient(rho, np.concatenate(x).ravel(), axis=0, edge_order=2)	
        nabla_rho = rho_x**2+rho_z**2
        #nabla_rho[nabla_rho < 1e-3] = 0
        integrand1['g'] = -9.8*mu*np.nan_to_num(1/deriv2, posinf=0, neginf=0, nan=0)*nabla_rho
        phi_d = de.operators.integrate(integrand1, 'x').evaluate()['g'][0]
        integrand = -9.8*np.nan_to_num(1/deriv2, posinf=0, neginf=0, nan=0)*(nabla_rho)/(sw.dens0(28,-2))
        N_sq = -9.8/sw.dens0(28,-2)*np.average(np.average(np.gradient(rho_ref, z[0], axis=1), axis=0))
        #f = 0.5*(1+np.tanh((z-32+16*sigma**2/(sigma**2+4*(x-l)**2))/0.01))
        Fr = abs(f['tasks']['u'][0][Ni_x:Nf_x, Ni_z:Nf_z])/np.sqrt(8*9.8*(rho-sw.dens0(28,-2))/sw.dens0(28,-2))
        fig_j, ax_j = plt.subplots(figsize=(16,12))
        im_j = ax_j.imshow(np.transpose(Fr), vmin=0, vmax=2, origin='lower', cmap='bwr', extent=(L_1/(H-z0), L_2/(H-z0), -80/(H-z0), 0))
        plt.ylim(plt.ylim()[::1])
        plt.xlim(L_1/(H-z0), L_2/(H-z0))
        fig_j.colorbar(im_j, orientation='horizontal', label='$dz_*/d\\rho |\\nabla\\rho|^2$')
        plt.plot(np.linspace(0, L, Nx)/(H-z0), keel(2)/(H-z0), color='red')
        plt.xlabel('$x/z_0$')
        plt.ylabel('$z/z_0$')
        plt.tight_layout()
        fig_j.set_dpi(d)
        plt.savefig('phid_integrand_region_a200c200e.png')
        plt.clf()
        cmap = plt.cm.hsv

        cyclicnorm = CyclicNormalize(cmin=100, cmax=100.2, vmin=np.min(rho), vmax=np.max(rho))

        #fig, ax = plt.subplots(figsize=(12,12))
        #pcm = ax.imshow(np.transpose(integrand/N_sq), vmin=1e1, vmax=1e3, cmap='plasma', origin='lower', norm=cyclicnorm, extent=(0, L/(H-z0), -H/(H-z0), 0))
        #fig.colorbar(pcm, orientation='horizontal')
        #fig.set_dpi(d)
        plt.savefig('testing.png')
        plt.tight_layout()
        plt.clf() 

def boundary_layer():
    plt.rcParams.update({'font.size':14})
    #Boundary layer figure
    u = []
    w = []
    v = []
    k = []
    rho = []
    #titles = ['(a) $t=132t_0$', '(b) $t=192t_0$']
    titles = ['(a)', '(b)']
    for i in range(2):
        with h5py.File('regime_files/data-mixingsim-a200c200-00_s{0}.h5'.format(['220', '320'][i]), mode='r') as f:
            u.append(f['tasks']['u'][0])
            w.append(f['tasks']['w'][0])
            v.append(np.sqrt(u[i]**2+w[i]**2))
            k.append(f['tasks']['vorticity'][0])
            rho.append(f['tasks']['rho'][0])
    fig, axes = plt.subplots(2,1, sharex='col')
    for i in range(len(axes)):
        N_sq = -9.81*gaussian_filter(np.gradient(rho[i], np.linspace(0, H, Nz), axis=1), 2)/sw.dens0(28,-2) 
        max_ind = np.argmax(N_sq, axis=1)
        z_Nvalues = []
        for j in range(len(N_sq)):
            z_Nvalues.append(-np.linspace(0, H, Nz)[Nz-max_ind[j]])

        pcm = axes[i].imshow(gaussian_filter(np.transpose(k[i]), 3.5), vmin=-0.2, vmax=0.2, cmap='bwr', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        axes[i].fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(2)/(H-z0), facecolor="white", zorder=10)
        axes[i].plot(np.linspace(0, L, Nx)/(H-z0), keel(2)/(H-z0), linewidth=0.5, color='black')
        print(np.transpose(u[i])[300])
        #axes[i].streamplot(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(u[i]), np.transpose(w[i]), color='black', density=3.1, linewidth=0.4, arrowsize=1)
        axes[i].plot(np.linspace(0, L, Nx)/(H-z0), np.array(z_Nvalues)/(H-z0), linewidth=2.5, color='black')
        axes[i].set_xlim(60, 75)
        axes[i].set_ylim(0, -4)
        axes[i].set_ylim(axes[i].get_ylim()[::-1])
        axes[i].set_yticks([0, -1, -2, -3, -4])
        axes[i].set_yticklabels(['0', '1', '2', '3', '4'])
        axes[i].set_xticks([60, 65, 70, 75])
        axes[i].set_ylabel('$z/z_0$')
        axes[i].set_aspect('auto')
        axes[i].text(73.7, -0.75, titles[i], fontsize=18, weight='bold', ha='center', va='center', zorder=10)
    axes[1].set_xlabel('$x/z_0$')
    fig.subplots_adjust(right=0.8, hspace=0.15)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label("Vorticity $q$ (s$^{{-1}}$)")
    fig.set_size_inches(8,6, forward=True)
    plt.savefig('boundarylayer_figure.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print('done')
    plt.clf()

def bore():
    #Bore figure
    us = []
    ws = []
    vs = []
    qs = []
    rhos = []
    titles = ['(a)', '(b)']
    for fi in ['regime_files/data-mixingsim-a200c100-00_s119.h5', 'regime_files/data-mixingsim-a200c100-00_s179.h5']:
        with h5py.File(fi, mode='r') as f:
            us.append(f['tasks']['u'][0])
            ws.append(f['tasks']['w'][0])
            vs.append(f['tasks']['u'][0]**2+f['tasks']['w'][0]**2)
            qs.append(f['tasks']['vorticity'][0])
            rhos.append(f['tasks']['rho'][0])
    fig, axes = plt.subplots(2,1, sharex='col')
    for i in range(len(axes)):
        N_sq = -9.81*gaussian_filter(np.gradient(rhos[i], np.linspace(0, H, Nz), axis=1), 10)/sw.dens0(28,-2) 
        max_ind = np.argmax(N_sq, axis=1)
        z_Nvalues = []
        for j in range(len(N_sq)):
            z_Nvalues.append(-np.linspace(0, H, Nz)[Nz-max_ind[j]])

        #pcm = axes[i].imshow(np.transpose(us[i])/0.69, vmin=-0.8, vmax=0.8, cmap='bwr', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        pcm = axes[i].imshow(gaussian_filter(np.transpose(qs[i]), 3), vmin=-0.2, vmax=0.2, cmap='bwr', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        #axes[i].streamplot(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(us[i]), np.transpose(ws[i]), color='black', density=2, linewidth=0.35, arrowsize=0.35)
        axes[i].fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(2)/(H-z0), facecolor="white", zorder=10)
        axes[i].plot(np.linspace(0, L, Nx)/(H-z0), keel(2)/(H-z0), linewidth=0.5, color='black') 
        axes[i].plot(np.linspace(0, L, Nx)/(H-z0), np.array(z_Nvalues)/(H-z0), linewidth=2.5, color='black')
        axes[i].set_xlim(20, 80)
        axes[i].set_ylim(0, -4)
        axes[i].set_ylim(axes[i].get_ylim()[::-1])
        axes[i].set_yticks([0, -2, -4])
        axes[i].set_yticklabels(['0', '2', '4'])
        axes[i].set_ylabel('$z/z_0$')
        axes[i].text(75, -0.5, titles[i], fontsize=18, weight='bold', ha='center', va='center', zorder=10)
        axes[i].set_aspect('auto')
    axes[1].set_xlabel('$x/z_0$')
    fig.set_dpi(d)
    fig.subplots_adjust(right=0.8, hspace=0.15)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label("Vorticity $q$ (s$^{{-1}}$)")
    plt.gcf().set_size_inches(8,6, forward=True)
    plt.savefig('bore_figure.pdf', format='pdf', bbox_inches='tight')
    plt.clf()

def regime_graphic():
    #density regimes
    titles1 = ['(a) Supercritical', '(b) LBR', '(c) Solitary Waves']
    titles2 = ['(a) Vortex Shedding', '(b) Quasi-laminar', '(c) Lee Waves']
    a_s = [2, 1.2, 1.2]
    c_s = [2, 0.5, 1.0]
    a_s2 = [2, 0.5, 0.5]
    c_s2 = [2, 1.0, 0.5]
    fs1 = ['regime_files/data-mixingsim-a200c200-00_s160.h5', 
        'regime_files/data-mixingsim-a102c005-00_s70.h5',
        'regime_files/data-mixingsim-a102c100-00_s180.h5'] 
    fs2 = ['regime_files/data-mixingsim-a200c200-00_s320.h5',
        'regime_files/data-mixingsim-a005c100-00_s240.h5', 
        'regime_files/data-mixingsim-a005c005-00_s180.h5']
    rhos1 = []
    us1 = []
    ws1 = []
    rhos2 = []
    us2 = []
    ws2 = []
    i = 0
    for fi in fs1:
        with h5py.File(fi, mode='r') as f:
            rhos1.append(f['tasks']['rho'][0])
            us1.append(f['tasks']['u'][0])
            ws1.append(f['tasks']['w'][0])
            plt.imshow(np.transpose(us1[i]), cmap='bwr', vmin=-0.4, vmax=0.4, extent=(0, L/(H-z0), -H/(H-z0), 0))
            plt.savefig('test_{0}.png'.format(i))
            plt.clf()
        i+=1
    for fi in fs2:
        with h5py.File(fi, mode='r') as f:
            rhos2.append(f['tasks']['rho'][0])
            us2.append(f['tasks']['u'][0])
            ws2.append(f['tasks']['w'][0])
    plt.rcParams.update({'font.size':14})
    #Upstream
    fig, axes = plt.subplots(1,3, figsize=(13, 3))
    paddingx = [-0.4, -0.1, 1, 1]
    paddingy = [-0.01, -0.01, -0.08, -0.09]
    ms = {'P': 13, 'D': 10, 'o': 13}
    for i in range(len(axes)):
        pcm = axes[i].imshow(np.transpose(rhos1[i])-sw.dens0(28,-2), vmin=0, vmax=2, cmap='viridis', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        #pcm = axes[i, j].imshow(np.transpose(us1[k][:, ::-1]), cmap='bwr', vmin=-0.4, vmax=0.4, extent=(0, L/(H-z0), -H/(H-z0), 0))
        axes[i].fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(a_s[i])/(H-z0), facecolor="white", zorder=10)
        speed = np.sqrt(np.transpose(us1[i])**2+np.transpose(ws1[i])**2)
        U = c_s[i]*np.sqrt((H-z0)*DB)
        lw = 2.2*speed/U
        axes[i].plot(np.linspace(0, L, Nx)/(H-z0), keel(a_s[i])/(H-z0), linewidth=0.5, color='black')
        c = axes[i].streamplot(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(us1[i]), np.transpose(ws1[i]), color='red', density=1.3, linewidth=lw, arrowsize=0.8, arrowstyle='->')
        c.lines.set_alpha(0.15)
        for x in axes[i].get_children():
            if type(x) == matplotlib.patches.FancyArrowPatch:
                x.set_alpha(0.15)
        axes[i].set_xticks([25, 35, 45, 55, 65, 75])
        axes[i].set_yticklabels(['4', '3', '2', '1', '0'])
        axes[i].set_xlim(20,75)
        axes[i].set_aspect("auto")
        axes[i].plot(53+0.8*len(titles1[i])+paddingx[i], paddingy[i]+0.4, markeredgecolor='k', ms=ms[['P', 'D', 'o'][i]], marker=['P', 'D', 'o'][i], color=colors2[['P', 'D', 'o'][i]], clip_on=False)
        axes[i].set_title(titles1[i])
        axes[i].set_ylim(0,-4)
        axes[i].set_ylim(axes[i].get_ylim()[::-1])
        axes[i].set_xlabel('$x/z_0$')
    axes[0].set_ylabel('$z/z_0$')
    plt.tight_layout()
    fig.subplots_adjust(right=0.8, hspace=0.6, wspace=0.3)
    cbar_ax = fig.add_axes([0.835, 0.18, 0.02, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label("$\\rho-\\rho_1$ (kg m$^{{-3}}$)")
    fig.set_dpi(d)
    plt.savefig('regime_graphic_figure_up.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()


    #Downstream
    fig, axes = plt.subplots(1,3, figsize=(13, 3))
    paddingx = [0.1, 0, 0.6]
    paddingy = [-0.01, -0.01, -0.08, -0.09]
    ms = {'*': 17.5, '^': 13, 's': 12}
    for i in range(len(axes)):
        pcm = axes[i].imshow(np.transpose(rhos2[i])-sw.dens0(28,-2), vmin=0, vmax=2, cmap='viridis', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        #pcm = axes[i, j].imshow(np.transpose(us1[k][:, ::-1]), cmap='bwr', vmin=-0.4, vmax=0.4, extent=(0, L/(H-z0), -H/(H-z0), 0))
        axes[i].fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(a_s2[i])/(H-z0), facecolor="white", zorder=10)
        speed = np.sqrt(np.transpose(us2[i])**2+np.transpose(ws2[i])**2)
        U = c_s2[i]*np.sqrt((H-z0)*DB)
        lw = 2*speed/U
        axes[i].plot(np.linspace(0, L, Nx)/(H-z0), keel(a_s2[i])/(H-z0), linewidth=0.5, color='black')
        c = axes[i].streamplot(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(us2[i]), np.transpose(ws2[i]), color='red', density=1.23, linewidth=lw, arrowsize=0.8, arrowstyle='->')
        c.lines.set_alpha(0.15)
        for x in axes[i].get_children():
            if type(x) == matplotlib.patches.FancyArrowPatch:
                x.set_alpha(0.15)
        axes[i].set_xticks([75, 85, 95, 105, 115])
        axes[i].set_yticklabels(['4', '3', '2', '1', '0'])
        axes[i].set_xlim(75,115)
        axes[i].set_aspect("auto")
        axes[i].plot(96+0.8*len(titles2[i])+paddingx[i], paddingy[i]+0.4, ms=ms[['*', '^', 's'][i]], markeredgecolor='k', marker=['*', '^', 's'][i], color=colors2[['*', '^', 's'][i]], clip_on=False)
        axes[i].set_title(titles2[i])
        axes[i].set_ylim(0,-4)
        axes[i].set_ylim(axes[i].get_ylim()[::-1])
        axes[i].set_xlabel('$x/z_0$')
    axes[0].set_ylabel('$z/z_0$')
    plt.tight_layout()
    fig.subplots_adjust(right=0.8, hspace=0.6, wspace=0.3)
    cbar_ax = fig.add_axes([0.835, 0.18, 0.02, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label("$\\rho-\\rho_1$ (kg m$^{{-3}}$)")
    fig.set_dpi(d)
    plt.savefig('regime_graphic_figure_down.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()

    fig = plt.figure(figsize=(13, 6))
    spec = GridSpec(ncols=6, nrows=2)
    ax1 = fig.add_subplot(spec[0, 0:2])
    ax2 = fig.add_subplot(spec[0, 2:4])
    ax3 = fig.add_subplot(spec[0, 4:])
    ax4 = fig.add_subplot(spec[1, 1:3])
    ax5 = fig.add_subplot(spec[1, 3:5])
    axes = [ax1, ax2, ax3, ax4, ax5]
    paddingx = [0, -7, -2, -6, -4.5]
    paddingy = [-0.05, -0.02, -0.045, -0.03, -0.02]
    for i in range(len(axes)):
        pcm = axes[i].imshow(np.transpose(rhos2[i])-sw.dens0(28,-2), vmin=0, vmax=2, cmap='viridis', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        axes[i].fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(a_s2[i])/(H-z0), facecolor="white", zorder=10)
        axes[i].plot(np.linspace(0, L, Nx)/(H-z0), keel(a_s2[i])/(H-z0), linewidth=0.5, color='black')
        speed = np.sqrt(np.transpose(us2[i])**2+np.transpose(ws2[i])**2)
        U = c_s2[i]*np.sqrt((H-z0)*DB)
        lw = 2*speed/U
        c = axes[i].streamplot(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(us2[i]), np.transpose(ws2[i]), color='red', density=1.23, linewidth=lw, arrowsize=0.8, arrowstyle='->')
        c.lines.set_alpha(0.15)
        for x in axes[i].get_children():
            if type(x) == matplotlib.patches.FancyArrowPatch:
                x.set_alpha(0.15)
        axes[i].set_xticks([75, 85, 95, 105, 115])
        axes[i].set_yticklabels(['4', '3', '2', '1', '0'])
        axes[i].set_xlim(75,115)
        axes[i].set_aspect("auto")
        axes[i].plot(112+0.10*len(titles2[i])+paddingx[i], 0.4+paddingy[i], ms=10, marker=['*', 'p', '^', 's', 'd'][i], color=colors2[['*', 'p', '^', 's', 'd'][i]], clip_on=False)
        print(i)
        axes[i].set_title(titles2[i])
        axes[i].set_ylim(0,-4)
        axes[i].set_ylim(axes[i].get_ylim()[::-1])
        axes[i].set_xlabel('$x/z_0$')
        if i == 0 or i == 3:
            axes[i].set_ylabel('$z/z_0$')
    plt.tight_layout()
    fig.subplots_adjust(right=0.8, hspace=0.6, wspace=0.57)
    cbar_ax = fig.add_axes([0.835, 0.18, 0.02, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label("$\\rho-\\rho_1$ (kg/m$^3$)")
    fig.set_dpi(d)
    plt.savefig('regime_graphic_figure_down.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()  
"""

def zmix():
    #zmix/z0, separate plots upstream/downstream, with keel
    a_axis = [0.5, 0.95, 1.2, 2.0] #eta
    block_width, block_depth = 0.1, 0.1

    figU = plt.figure(figsize=(7,6))
    ax = figU.gca(projection='3d')
    ax.view_init(elev=30, azim=225)
    for i in range(len(a)):
        for j in range(len(c)):
            ax.bar3d(c_axis[j], a_axis[i], 0, block_width, block_depth, z_mix_import_up[a[i]+c[j]], color=colors2[markers1[i][j]], edgecolor='k', shade=True)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['D'], mec='k', label='LBR')
    ax.plot([], [], marker='s', linestyle='None', color=colors2['o'], mec='k', label='Solitary Waves')
    ax.plot([], [], marker='s', linestyle='None', color=colors2['P'], mec='k', label='Supercritical')
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 13}, fancybox=True, shadow=True)
    ax.set_xlabel('Fr')
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('$\\eta$')
    ax.zaxis.set_rotate_label(False)
    plt.tight_layout()
    figU.savefig('zmix_Upstream_figure.pdf', format='pdf')
    plt.clf()

# RUN
#K_p_upstream()
#K_p_upstream_var1()
#K_p_downstream_var1()
#K_p_upstream_var2()
#K_p_downstream_var2()
#K_p_upstream_var3()
#K_p_downstream_var3()
#K_p_downstream()
#test_heatmap()
joint_regime()
#joint_regime_arctic()
#bore()
#boundary_layer()
#regime_graphic()
#K_p_downstream_var4()
#K_p_upstream_var4()
#K_p_downstream_var5()
#K_p_upstream_var5()
K_p_subplots_4()
zmix()
