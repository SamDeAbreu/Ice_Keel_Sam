import json
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

def average_data(d, c=0):
    return np.mean(d[22:])

def stdev_data(d, c):
    return np.std(d[22:])

def stdev_log_data(d, c):
    return np.std(np.log10(d[22:]))

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

plt.rcParams.update({'font.size':16})

#x, z = domain.grids()
#a = ['a005', 'a009', 'a102', 'a200'] #heights
#c = ['c005', 'c100', 'c105', 'c200'] #speeds
sp = [[220, 260, 450, 450], [220, 260, 450, 450], [220, 260, 450, 450], [220, 260, 450, 450]]
a = ['a005', 'a095', 'a102', 'a200']
c = ['c005', 'c100', 'c105', 'c200']
labels_height = ['$\\eta=0.5$', '$\\eta=0.95$', '$\\eta=1.2$', '$\\eta=2.0$']
labels_height_Fr = ['$Fr=0.5$', '$Fr=1.0$', '$Fr=1.5$', '$Fr=2.0$']
labels_regime_up = ['Unstable Supercritical', 'Stable Subcritical', 'Unstable Subcritical'] #Old: ['Supercritical', 'Rarefaction',  'Solitary Waves', 'Blocking']
labels_regime_down = ['Vortex Shedding','Fast-Laminar', 'Lee Waves'] #Old: ['Vortex Shedding', 'Stirred', 'Laminar Jump', 'Blocked', 'Lee Waves']
markers_labels_up = ['P', 'D', 'o']
markers_labels_down = ['*', '^', 's']
markers1 = [['D', 'D', 'o', 'P'], ['D', 'o', 'o', 'P'], ['D', 'o', 'o', 'P'], ['D', 'o', 'o', 'P']] #vortex shedding = star, bore & MSD = circle, blocking = square, bore & TD = triangle, MSD=diamond, lee = plus
markers2 = [['s', '^', '^', '*'], ['s', '^', '*', '*'], ['s', '^', '*', '*'], ['p', 'p', '*', '*']] #vortex shedding = star, bore & MSD = circle, blocking = square, bore & TD = triangle, MSD=diamond, lee = plus
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
phid_import_up = json.load(open('phi_d_values_{0}-{1}.txt'.format(160, l)))
phid_import_down = json.load(open('phi_d_values_{0}-{1}.txt'.format(l, 920)))
conv_id = {'a005': 'H05', 'a095': 'H09', 'a102': 'H12', 'a200': 'H20', 'c005': 'F05', 'c100': 'F10', 'c105': 'F15', 'c200': 'F20'}
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
    avgs[i]['Phi_d_D_series'] = []
    avgs[i]['Phi_d_U_series'] = []
    avgs[i]['z_mix_U'] = []
    avgs[i]['z_mix_D'] = []
    for j in range(len(c)):
        avgs[i]['MLD'].append(average_data(data[i][j][7], c[j]))
        avgs[i]['MLD_stdev'].append(MLD_std[i][j])
        avgs[i]['KED_D'].append(average_data(data[i][j][15], c[j]))
        avgs[i]['KED_D_stdev'].append(stdev_data(data[i][j][15], c[j]))
        avgs[i]['Phi_d_D'].append(average_data(phid_import_down[conv_id[c[j]]+conv_id[a[i]]][0], c[j][0]))
        avgs[i]['Phi_d_D_stdev'].append(stdev_data(phid_import_down[conv_id[c[j]]+conv_id[a[i]]][0], c[j][0]))
        avgs[i]['KED_U'].append(average_data(data[i][j][22], c[j]))
        avgs[i]['KED_U_stdev'].append(stdev_data(data[i][j][22], c[j]))
        avgs[i]['Phi_d_U'].append(average_data(phid_import_up[conv_id[c[j]]+conv_id[a[i]]][0], c[j]))
        avgs[i]['Phi_d_D_series'].append(phid_import_down[conv_id[c[j]]+conv_id[a[i]]])
        avgs[i]['Phi_d_U_series'].append(phid_import_up[conv_id[c[j]]+conv_id[a[i]]])
        avgs[i]['Phi_d_U_stdev'].append(stdev_data(phid_import_up[conv_id[c[j]]+conv_id[a[i]]][0], c[j]))
        avgs[i]['time'].append(data[i][j][0])
        avgs[i]['K_p_U_series'].append(K_import_up[conv_id[c[j]]+conv_id[a[i]]])
        avgs[i]['K_p_D_series'].append(K_import_down[conv_id[c[j]]+conv_id[a[i]]])
        # Import diffusivities
        avgs[i]['K_p_U'].append(average_data(K_import_up[conv_id[c[j]]+conv_id[a[i]]][0]))
        avgs[i]['K_p_D'].append(average_data(K_import_down[conv_id[c[j]]+conv_id[a[i]]][0]))
        avgs[i]['z_mix_U'].append(average_data(z_mix_import_up[conv_id[c[j]]+conv_id[a[i]]][0]))
        avgs[i]['z_mix_D'].append(average_data(z_mix_import_down[conv_id[c[j]]+conv_id[a[i]]][0]))

print('Label', 'K_U', 'Phi_U', 'z_mix_U', 'K_D', 'Phi_D', 'z_mix_D')
print(avgs[0]['Phi_d_U'][0], np.mean(json.load(open('phi_d_values_{0}-{1}.txt'.format(160, l)))['F05H05'][0][22:]))
for j in range(len(c)):
    for i in range(len(a)):
        print('{0}{1} & {2} & {3} & {4} & {5} & {6} & {7}\\\\'.format(a[i], c[j], round(avgs[i]['K_p_U'][j], 3), round(avgs[i]['Phi_d_U'][j]/avgs[0]['Phi_d_U'][0], 3), round(avgs[i]['z_mix_U'][j]/8, 3), round(avgs[i]['K_p_D'][j], 3), round(avgs[i]['Phi_d_D'][j]/avgs[0]['Phi_d_U'][0], 3), round(avgs[i]['z_mix_D'][j]/8, 3)))
        
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
    plt.xticks([0, 0.5, 1.0, 1.5])
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

def phi_d_subplots():
    #phi_d Upstream
    plt.rcParams.update({'font.size':18})
    fig = plt.figure(figsize=(13,5))
    a_axis = [0.5, 0.95, 1.2, 2.0] #eta
    block_width, block_depth = 0.1, 0.1
    phi_0_up = 9e-7
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.view_init(elev=30, azim=225)
    for i in range(len(a)):
        for j in range(len(c)):
            print(i, j, avgs[i]['Phi_d_U'][j]/avgs[0]['Phi_d_U'][0])
            ax.bar3d(x=c_axis[j]-block_width, y=a_axis[i]-block_width, z=block_depth, dx=block_width, dy=block_depth, dz=avgs[i]['Phi_d_U'][j]/avgs[0]['Phi_d_U'][0], color=colors2[markers1[i][j]], edgecolor='k', shade=True)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['D'], mec='k', label='Stable Subcritical', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['o'], mec='k', label='Unstable Subcritical', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['P'], mec='k', label='Unstable Supercritical', ms=11)
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.26,1), prop={'size': 15}, fancybox=True, shadow=True)
    ax.xaxis.set_rotate_label(False)
    ax.set_xlabel('\n$Fr$', linespacing=3.2)
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('\n$\\eta$', linespacing=3.2)
    ax.set_yticks([0.5, 1, 1.5, 2])
    ax.zaxis.set_rotate_label(False)
    ax.set_zlim(0, 3)
    #ax.set_zticks([0, 0.5, 1.0, 1.5])
    #ax.set_zticklabels(['$10^{{{0}}}$'.format(ii) for ii in [0, 0.5, 1.0, 1.5]])
    ax.set_zlabel(r'$\dfrac{\overline{\Phi}_U}{\overline{\Phi}_0}$       ')
    ax.set_title('(a)', loc='left')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.view_init(elev=30, azim=225)
    for i in range(len(a)):
        for j in range(len(c)):
            print('k', avgs[i]['Phi_d_D'][j]/avgs[0]['Phi_d_U'][0])
            ax.bar3d(c_axis[j]-block_width, a_axis[i]-block_width, block_depth, block_width, block_depth, avgs[i]['Phi_d_D'][j]/avgs[0]['Phi_d_U'][0], color=colors2[markers2[i][j]], edgecolor='k', shade=True)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['p'], mec='k', label='Diffusive BL', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['s'], mec='k', label='Lee Waves', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['^'], mec='k', label='Fast-laminar', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['*'], mec='k', label='Vortex Shedding', ms=11)
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.295,1.01), prop={'size': 15}, fancybox=True, shadow=True)
    ax.xaxis.set_rotate_label(False)
    ax.set_xlabel('\n$Fr$', linespacing=3.2)
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('\n$\\eta$', linespacing=3.2)
    ax.set_yticks([0.5, 1, 1.5, 2])
    ax.zaxis.set_rotate_label(False)
    ax.set_zlim(0,3)
    #ax.set_zticks([0, 0.5, 1.0, 1.5])
    #ax.set_zticklabels(['$10^{{{0}}}$'.format(ii) for ii in [0, 0.5, 1.0, 1.5]])
    ax.set_zlabel(r'$\dfrac{\overline{\Phi}_D}{\overline{\Phi}_0}$       ')
    ax.set_title('(b)', loc='left')

    plt.tight_layout()
    fig.savefig('phid_subplots_var4.pdf', format='pdf')
    plt.clf()

    plt.rcParams.update({'font.size':16})

def phi_d_downstream():
    #phi_d Downstream
    for i in range(len(a)):
        for j in range(len(c)):
            plt.errorbar(c_axis[j], avgs[i]['Phi_d_D'][j], yerr=avgs[i]['Phi_d_D_stdev'][j]/epsilon_0, capsize=5, linestyle='None', marker=markers2[i][j], color=colors1[i], label=labels_height[i])    
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

colors2 = {'D': 'darkgreen', '^': '#663300', 's': 'red', 'o': '#ff3399', 'P': 'darkviolet', '*': '#fcb900', 'p': '#0066ff'}
# P: #fcb900
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
            ax.bar3d(c_axis[j]-block_width, a_axis[i]-block_width, 0, block_width, block_depth, np.log10(avgs[i]['K_p_U'][j]/mu), color=colors2[markers1[i][j]], edgecolor='k', shade=True)
    plt.plot([], [], marker='s', linestyle='None', color=colors2['D'], mec='k', label='Stable Subcritical')
    plt.plot([], [], marker='s', linestyle='None', color=colors2['o'], mec='k', label='Unstable Subcritical')
    plt.plot([], [], marker='s', linestyle='None', color=colors2['P'], mec='k', label='Supercritical')
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 14}, fancybox=True, shadow=True)
    ax.xaxis.set_rotate_label(False)
    ax.set_xlabel('    $Fr$')
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('$\\eta$    ')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlim(0, 5)
    ax.set_zticks(range(5))
    ax.set_zticklabels([r'$10^{0}$'.format(ii) for ii in range(5)])
    ax.set_zlabel(r'$\overline{K}_U$')
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
            ax.bar3d(c_axis[j]-block_width, a_axis[i]-block_width, 0, block_width, block_depth, np.log10(avgs[i]['K_p_D'][j]/mu), color=colors2[markers2[i][j]], edgecolor='k', shade=True)
    plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], mec='k', label='Lee Waves')
    plt.plot([], [], marker='s', linestyle='None', color=colors2['^'], mec='k', label='Fast-laminar')
    plt.plot([], [], marker='s', linestyle='None', color=colors2['*'], mec='k', label='Vortex Shedding')
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 14}, fancybox=True, shadow=True)
    ax.xaxis.set_rotate_label(False)
    ax.set_xlabel('$    Fr$')
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('$\\eta$    ')
    ax.zaxis.set_rotate_label(False)
    #ax.set_zlabel(r'$\log (\overline{K}_D / \mu)$       ')
    ax.set_zlim(0, 5)
    ax.set_zticks(range(5))
    ax.set_zticklabels([r'$10^{0}$'.format(ii) for ii in range(5)])
    ax.set_zlabel(r'$\overline{K}_D$')
    fig.savefig('Kp_Downstream_figure_var4.pdf', format='pdf', bbox_inches='tight')
    plt.clf()

def K_p_subplots_4():
    #var4, both up and downstream
    fig = plt.figure(figsize=(6.5,9.75))
    a_axis = [0.5, 0.95, 1.2, 2.0] #eta
    block_width, block_depth = 0.1, 0.1
    
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    ax.view_init(elev=30, azim=225)
    for i in range(len(a)):
        for j in range(len(c)):
            print(c_axis[j]-block_width, a_axis[i]-block_width, avgs[i]['K_p_U'][j])
            ax.bar3d(c_axis[j]-block_width, a_axis[i]-block_width, block_depth, block_width, block_depth, avgs[i]['K_p_U'][j], color=colors2[markers1[i][j]], edgecolor='k', shade=True)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['D'], mec='k', label='Stable Subcritical', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['o'], mec='k', label='Unstable Subcritical', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['P'], mec='k', label='Unstable Supercritical', ms=11)
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.4,1), prop={'size': 15}, fancybox=True, shadow=True)
    ax.xaxis.set_rotate_label(False)
    ax.set_xlabel('\n$Fr$', linespacing=3.2)
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('\n$\\eta$', linespacing=3.2)
    ax.set_yticks([0.5, 1, 1.5, 2])
    ax.zaxis.set_rotate_label(False)
    ax.set_zlim(0, 3)
    #ax.set_zticks([0, 2.5, 5, 7.5, 10])
    #ax.set_zticklabels(['$10^{{{0}}}$'.format(ii) for ii in [0, 2.5, 5, 7.5, 10]])
    ax.set_zlabel(r'$\overline{K}_U$      ')
    ax.set_title('(a)', loc='left')

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    ax.view_init(elev=30, azim=225)
    for i in range(len(a)):
        for j in range(len(c)):
            ax.bar3d(c_axis[j]-block_width, a_axis[i]-block_width, block_depth, block_width, block_depth, avgs[i]['K_p_D'][j], color=colors2[markers2[i][j]], edgecolor='k', shade=True)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['s'], mec='k', label='Lee Waves', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['^'], mec='k', label='Fast-laminar', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['*'], mec='k', label='Vortex Shedding', ms=11)
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.22,1), prop={'size': 14}, fancybox=True, shadow=True)
    ax.xaxis.set_rotate_label(False)
    ax.set_xlabel('\n$Fr$', linespacing=3.2)
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('\n$\\eta$', linespacing=3.2)
    ax.set_yticks([0.5, 1, 1.5, 2])
    ax.zaxis.set_rotate_label(False)
    ax.set_zlim(0, 3)
    #ax.set_zticks([0, 2.5, 5, 7.5, 10])
    #ax.set_zticklabels(['$10^{{{0}}}$'.format(ii) for ii in [0, 2.5, 5, 7.5, 10]])
    ax.set_zlabel(r'$\overline{K}_D$      ')
    ax.set_title('(b)', loc='left')

    plt.tight_layout()
    fig.savefig('Kp_subplots_var4.pdf', format='pdf', bbox_inches='tight', pad_inches=0.8)
    plt.clf()

def K_p_upstream_var5():
    #K_p upstream var5: (Fr, eta) space with height and marker for regime
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
    ax.set_zlabel(r'$\log (\overline{K}_U / \mu)$       ')
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
    ax.set_zlabel(r'$\log (\overline{K}_D / \mu)$      ')
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
    ms = {'o': 17.5, 'D': 15, 'P': 18, 's': 17, '^': 18, '*': 20.5, 'p': 19.5}
    plt.figure(figsize=(6,5))
    for i in range(len(a)):
        for j in range(len(c)):
            plt.plot(c_axis[j]-shift, [0.5, 0.95, 1.2, 2][i], markeredgecolor='k', linestyle='None', marker=markers1[i][j], color=colors2[markers1[i][j]], ms=ms[markers1[i][j]], zorder=10)    
#    line4, = plt.plot([], [], marker='D', linestyle='None', color=colors2['D'], label='LBR', ms=11)
    line3, = plt.plot([], [], marker='D', markeredgecolor='k', linestyle='None', color=colors2['D'], label='Stable Subcritical', ms=ms['D'] * 0.75)
    line2, = plt.plot([], [], marker='o', markeredgecolor='k', linestyle='None', color=colors2['o'], label='Unstable Subcritical', ms=ms['o'] * 0.75)
    line1, = plt.plot([], [], marker='P', markeredgecolor='k', linestyle='None', color=colors2['P'], label='Unstable Supercritical', ms=ms['P'] * 0.75)
    for i in range(len(a)):
        for j in range(len(c)):
            plt.plot(c_axis[j]+shift, [0.5, 0.95, 1.2, 2][i], markeredgecolor='k', linestyle='None', marker=markers2[i][j], color=colors2[markers2[i][j]], ms=ms[markers2[i][j]], zorder=10)    
    line9, = plt.plot([], [], marker='s', markeredgecolor='k', linestyle='None', color=colors2['s'], label='Lee Waves', ms=ms['s'] * 0.75)
#    line8, = plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], label='Blocked', ms=11)
    line7, = plt.plot([], [], marker='^', markeredgecolor='k', linestyle='None', color=colors2['^'], label='Fast-Laminar', ms=ms['^'] * 0.75)
#    line6, = plt.plot([], [], marker='p', linestyle='None', color=colors2['p'], label='Stirred', ms=11)
    line5, = plt.plot([], [], marker='p', markeredgecolor='k', linestyle='None', color=colors2['p'], label='Diffusive BL', ms=ms['p'] * 0.75)
    line6, = plt.plot([], [], marker='*', markeredgecolor='k', linestyle='None', color=colors2['*'], label='Vortex Shedding', ms=ms['*'] * 0.75)
    first_legend = plt.legend(handles=[line1, line2, line3], loc='lower center', bbox_to_anchor=(0.27, -0.451), prop={'size': 13}, fancybox=True, shadow=True)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=[line6, line7, line9, line5], loc='lower center', bbox_to_anchor=(0.76, -0.52), prop={'size': 13}, fancybox=True, shadow=True)
    plt.xlabel('$Fr$')
    plt.ylabel('$\\eta$    ', rotation=False)
    plt.xlim(0.3, 2.2)
    plt.yticks([0.5, 1.0, 1.5, 2.0])
    plt.ylim(0.3, 2.2)
    plt.text(0.83, -0.04, 'Upstream', ha='center', va='center')
    plt.text(1.75, -0.05, 'Downstream', ha='center', va='center')
    plt.grid()
#    plt.gcf().set_size_inches(10,6, forward=True)
#    plt.gca().set_aspect(1.3)
    plt.savefig('regime_layout.pdf', format='pdf', bbox_inches='tight')
    plt.clf()

def joint_regime_ms():
    #Joint regime layout with ms
    shift = 0.032
    ms = {'o': 17.5, 'D': 15, 'P': 18, 's': 17, '^': 18, '*': 20.5}
    plt.figure(figsize=(6,5))
    max_up = np.log10(avgs[-1]['K_p_U'][-1]/mu)
    max_down = np.log10(avgs[-1]['K_p_D'][-1]/mu)
    for i in range(len(a)):
        for j in range(len(c)):
            marker_size = np.exp(2.2*np.log10(avgs[i]['K_p_U'][j]/mu)/max_up) * 2
            shift = (ms[markers1[i][j]]-13+12*marker_size)/6500
            plt.plot(c_axis[j]-shift, [0.5, 0.95, 1.2, 2][i], markeredgecolor='k', linestyle='None', marker=markers1[i][j], color=colors2[markers1[i][j]], ms=ms[markers1[i][j]]-13+marker_size, zorder=10)    
#    line4, = plt.plot([], [], marker='D', linestyle='None', color=colors2['D'], label='LBR', ms=11)
    line3, = plt.plot([], [], marker='D', markeredgecolor='k', linestyle='None', color=colors2['D'], label='Stable Subcritical', ms=ms['D'] * 0.75)
    line2, = plt.plot([], [], marker='o', markeredgecolor='k', linestyle='None', color=colors2['o'], label='Unstable Subcritical', ms=ms['o'] * 0.75)
    line1, = plt.plot([], [], marker='P', markeredgecolor='k', linestyle='None', color=colors2['P'], label='Unstable Supercritical', ms=ms['P'] * 0.75)
    for i in range(len(a)):
        for j in range(len(c)):
            marker_size = np.exp(2.7*np.log10(avgs[i]['K_p_D'][j]/mu)/max_down) * 1.5
            shift = (ms[markers1[i][j]]-13+12*marker_size)/6500
            plt.plot(c_axis[j]+shift, [0.5, 0.95, 1.2, 2][i], markeredgecolor='k', linestyle='None', marker=markers2[i][j], color=colors2[markers2[i][j]], ms=ms[markers2[i][j]]-13+marker_size, zorder=10)    
    line9, = plt.plot([], [], marker='s', markeredgecolor='k', linestyle='None', color=colors2['s'], label='Lee Waves', ms=ms['s'] * 0.75)
#    line8, = plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], label='Blocked', ms=11)
    line7, = plt.plot([], [], marker='^', markeredgecolor='k', linestyle='None', color=colors2['^'], label='Fast-Laminar', ms=ms['^'] * 0.75)
#    line6, = plt.plot([], [], marker='p', linestyle='None', color=colors2['p'], label='Stirred', ms=11)
    line5, = plt.plot([], [], marker='*', markeredgecolor='k', linestyle='None', color=colors2['*'], label='Vortex Shedding', ms=ms['*'] * 0.75)
    first_legend = plt.legend(handles=[line1, line2, line3], loc='lower center', bbox_to_anchor=(0.21, -0.5), prop={'size': 13}, fancybox=True, shadow=True)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=[line5, line7, line9], loc='lower center', bbox_to_anchor=(0.7, -0.5), prop={'size': 13}, fancybox=True, shadow=True)
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel('Dimensionless Keel Draft $\\eta$')
    plt.yticks([0.5, 1.0, 1.5, 2.0])
    plt.xlim(0.3,2.2)
    plt.ylim(0.3,2.2)
    plt.text(0.73, -0.13, 'Upstream', ha='center', va='center')
    plt.text(1.635, -0.13, 'Downstream', ha='center', va='center')
    plt.grid()
    plt.savefig('regime_layout_ms.pdf', dpi=d, bbox_inches='tight')
    plt.clf()

def Fr(z0, Si, Sf, U):
        dB = 9.8*(sw.dens0(Sf, -2) - sw.dens0(Si, -2))/sw.dens0(Si, -2)
        return U/np.sqrt(z0*dB)
def eta(z0, h):
    return h/z0
def sigma_rho(S, S_sigma): # Error in EOS (found the exact equation for EOS and derived error formula for leading order terms, needs to be checked)
    return S_sigma*np.sqrt((0.8)**2+9/4*(0.005)**2*S+4*(0.0004)**2*S**2)

def sigma_Fr(z0, Si, Sf, z0_sigma, Si_sigma, Sf_sigma, U):
    rhoi_sigma = sigma_rho(Si, Si_sigma)
    rhof_sigma = sigma_rho(Sf, Sf_sigma)
    rhoi = ufloat(sw.dens0(Si, -2), rhoi_sigma)
    rhof = ufloat(sw.dens0(Sf, -2), rhof_sigma)
    z0u = ufloat(z0, z0_sigma)
    dB = 9.8*(rhof-rhoi)/rhoi
    Fr = U/(z0u*dB)**0.5
    return Fr.std_dev

def arctic_plot(h, color1, color2):
    # Set salinity and ML depth base values (with uncertainty)
    # We organize the data into two categories: summer (denoted with i) and winter (f), each with uncertainties (u)
    # Summer corresponds to the July value for the particular region and April for winter
    
    # Salinity values taken from Figure 8 in PFW (visually estimated, somewhat subjective)
    S_values = {'Chukchi Sea': {'Si': 29.1, 'Si_u': 1.00, 'Sf': 30.1, 'Sf_u': 0.1}, 
                'Southern Beaufort Sea': {'Si': 28.0, 'Si_u': 3.8, 'Sf': 30.5, 'Sf_u': 2.5}, 
                'Canada Basin': {'Si': 27.2, 'Si_u': 1.8, 'Sf': 30.1, 'Sf_u': 0.1}, 
                'Eurasian Basin': {'Si': 33.4, 'Si_u': 0.8, 'Sf': 33.8, 'Sf_u': 0.7}, 
                'Barents Sea': {'Si': 33.1, 'Si_u': 0.3, 'Sf': 34.5, 'Sf_u': 0.03}} 
    # MLD values taken from Figure 6 in PFW
    # We only consider summer values (since our initial layer in our simulation is the "summer ML")
    z0_values = {'Chukchi Sea': {'z0': 12.3, 'z0_u': 4}, 
                'Southern Beaufort Sea': {'z0': 8.5, 'z0_u': 4.5}, 
                'Canada Basin': {'z0': 8.9, 'z0_u': 3.9},  
                'Eurasian Basin': {'z0': 22.3, 'z0_u': 11.3}, 
                'Barents Sea': {'z0': 17.7, 'z0_u': 12.2}}
    
    # Trends of salinity and MLD in units of [value] per year
    # Taken from Figure 14 in PFW
    # Salinity trends are taken from the ice-covered (ic) column
    S_trends = {'Chukchi Sea': {'Si_t': 0.02, 'Sf_t': -0.07}, 
                'Southern Beaufort Sea': {'Si_t': 0.29, 'Sf_t': -0.04}, 
                'Canada Basin': {'Si_t': -0.11, 'Sf_t': -0.19}, 
                'Eurasian Basin': {'Si_t': -0.05, 'Sf_t': -0.07}, 
                'Barents Sea': {'Si_t': 0.02, 'Sf_t': 0.02,}} # Using summer trend for winter (due to missing trend in PFW)
    # MLD trends are taken from the ice-covered (ic) summer column
    z0_trends = {'Chukchi Sea': {'z0_t': -0.43}, # Using Winter ic trend (due to missing trend in PFW)
                'Southern Beaufort Sea': {'z0_t': 0.33}, 
                'Canada Basin': {'z0_t': -0.33}, 
                'Eurasian Basin': {'z0_t': -0.19}, 
                'Barents Sea': {'z0_t': 0.51}} # Using Summer ice-free trend (due to missing trend in PFW)
    
    # All regions that have a trend not from ice-covered summer are marked with a (*)
    labels_region = {'Chukchi Sea': '*1', 'Southern Beaufort Sea': '2', 'Canada Basin': '3', 'Eurasian Basin': '4', 'Barents Sea': '*5'}
    regions = S_values.keys()
    
    U = 0.2 # Ice speed (fixed for every region right now)

    # Compute (Fr, eta) values for each region
    Fr_values = {}
    eta_values = {}
    for region in regions:
        # Compute Fr
        Fr_val = Fr(z0_values[region]['z0'], S_values[region]['Si'], S_values[region]['Sf'], U)
        Fr_er = sigma_Fr(z0_values[region]['z0'], S_values[region]['Si'], S_values[region]['Sf'], z0_values[region]['z0_u'], S_values[region]['Si_u'], S_values[region]['Sf_u'], U)
        Fr_values[region] = (Fr_val, Fr_er) # (Fr, Fr_error)

        # Compute eta
        eta_val = eta(z0_values[region]['z0'], h=h)
        eta_er = eta(z0_values[region]['z0'], h=h)/z0_values[region]['z0']*z0_values[region]['z0_u']
        eta_values[region] = (eta_val, eta_er) # (eta, eta_error)
    
    # Compute predicted (Fr, eta) values for each region. Note that the prediction is linear (only a rough first derivative is used)
    years = 5 # How many years into the future we predict
    U_rate = 1.009 # Ice speed trend (increase of 0.9% per year from Rampal I think?)
    Fr_pred_values = {}
    eta_pred_values = {}
    for region in regions:
        # Compute new predicted values with trends
        z0_new = z0_values[region]['z0']+years*z0_trends[region]['z0_t']
        S_i_new = S_values[region]['Si']+years*S_trends[region]['Si_t']
        S_f_new = S_values[region]['Sf']+years*S_trends[region]['Sf_t']
        U_new = U*U_rate**years

        # Store data
        Fr_pred = Fr(z0_new, S_i_new, S_f_new, U_new)
        eta_pred = eta(z0_new, h=h)
        Fr_pred_values[region] = Fr_pred
        eta_pred_values[region] = eta_pred

    # Plot the regional markers
    k = 0 # Counter for zordering
    for region in regions:
        # Import values
        Fr_value = Fr_values[region][0]
        Fr_er = Fr_values[region][1]
        eta_value = eta_values[region][0]
        eta_er = eta_values[region][1]
        Fr_pred_value = Fr_pred_values[region]
        eta_pred_value = eta_pred_values[region]
        
        # Plot error boxes
        plt.gca().add_patch(patches.FancyBboxPatch(xy=(Fr_value-Fr_er, eta_value-eta_er), width=2*Fr_er, height=2*eta_er, linewidth=1, color=color2, fill='false', mutation_scale=0.05, alpha=0.10))
        # Plot predictive arrows
        plt.arrow(x=Fr_value, y=eta_value, dx=Fr_pred_value-Fr_value, dy=eta_pred_value-eta_value, linestyle='-', linewidth=2.8, length_includes_head=True, zorder=98+k, head_width=0.03)
        # Plot marker boxes
        plt.plot(Fr_value, eta_value, marker='s', color=color1, ms=18, zorder=101+k, markeredgecolor='k')
        # Plot text in marker boxes
        plt.text(Fr_value, eta_value-0.012, labels_region[region], fontsize=15, weight='bold', ha='center', va='center', zorder=101+k)
        k += 1
    

def joint_regime_arctic():
    #Joint regime Arctic layout 
    shift = 0 # offset the marker for each point
    max_up = np.log10(avgs[-1]['K_p_U'][-1]/mu) # Scaling for marker size
    max_down = np.log10(avgs[-1]['K_p_D'][-1]/mu) # Scaling for marker size
    ms = {'o': 17.5, 'D': 15, 'P': 18, 's': 17, '^': 18, '*': 20.5, 'p':17}
    # Plot upstream markers
    for i in range(len(a)):
        for j in range(len(c)):
            marker_size = np.exp(1.9*np.log10(avgs[i]['K_p_U'][j]/mu)/max_up) * 4.0 # Chosen through trial and error (feel free to change)
            shift = (ms[markers1[i][j]]-13+12*marker_size)/6500
            plt.plot(c_axis[j]-shift, [0.5, 0.95, 1.2, 2][i], markeredgecolor='k', linestyle='None', marker=markers1[i][j], color=colors2[markers1[i][j]], ms=ms[markers1[i][j]]-13+marker_size, zorder=10)    
    line2, = plt.plot([], [], marker='o', markeredgecolor='k', linestyle='None', color=colors2['o'], label='Unstable Subcritical', ms=ms['o'] * 0.75)
    line3, = plt.plot([], [], marker='D', markeredgecolor='k', linestyle='None', color=colors2['D'], label='Stable Subcritical', ms=ms['D'] * 0.75)
    line1, = plt.plot([], [], marker='P', markeredgecolor='k', linestyle='None', color=colors2['P'], label='Unstable Supercritical', ms=ms['P'] * 0.75)
    # Plot downstream markers
    for i in range(len(a)):
        for j in range(len(c)):
            marker_size = np.exp(1.9*np.log10(avgs[i]['K_p_D'][j]/mu)/max_down) * 4.0 # Chosen through trial and error (feel free to change)
            shift = (ms[markers1[i][j]]-13+12*marker_size)/6500
            plt.plot(c_axis[j]+shift, [0.5, 0.95, 1.2, 2][i], markeredgecolor='k', linestyle='None', marker=markers2[i][j], color=colors2[markers2[i][j]], ms=ms[markers2[i][j]]-13+marker_size, zorder=10)    
    line9, = plt.plot([], [], marker='s', markeredgecolor='k', linestyle='None', color=colors2['s'], label='Lee Waves', ms=ms['s'] * 0.75)
    line7, = plt.plot([], [], marker='^', markeredgecolor='k', linestyle='None', color=colors2['^'], label='Fast-Laminar', ms=ms['^'] * 0.75)
    line5, = plt.plot([], [], marker='*', markeredgecolor='k', linestyle='None', color=colors2['*'], label='Vortex Shedding', ms=ms['*'] * 0.75)
    # Plot legends
    first_legend = plt.legend(handles=[line1, line2, line3], loc='lower center', bbox_to_anchor=(0.26, -0.37), prop={'size': 13}, fancybox=True, shadow=True)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=[line5, line7, line9], loc='lower center', bbox_to_anchor=(0.75, -0.37), prop={'size': 13}, fancybox=True, shadow=True)
    # Other plot details
    plt.xlim(0,2.2)
    plt.ylim(0,2.7)
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel('Dimensionless Keel Draft $\\eta$    ')
    plt.xticks([0.5, 1, 1.5, 2])
    plt.yticks([0.5, 1, 1.5, 2, 2.5])
    plt.text(0.58, -0.39, 'Upstream', ha='center', va='center')
    plt.text(1.66, -0.39, 'Downstream', ha='center', va='center')
    plt.grid(zorder=0)

    # Add Arctic region markers (the most importnat part)
    arctic_plot(h=7.45, color1='#b30000', color2='#b30000')
    arctic_plot(h=7.45*2.5, color1='blue', color2='blue')
    #Arctic stuff
    # Winter data is March data, likewise summer is July. All std and averages are taken from these two months
    # ML depth is ice covered July data
    # Winter ML salinity is April ice covered data
    # Summer ML salinity is July ice covered data
    # We reject Makaraov data as PFW did

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
    with h5py.File('regime_files/data-mixingsim-Test-00_s1.h5', mode='r') as f:
        rho = f['tasks']['rho'][0]
        fig_j, ax_j = plt.subplots(figsize=(12,12))
        im_j = ax_j.imshow(np.transpose(rho)-sw.dens0(28,-2), vmin=0, vmax=1.75, interpolation='bicubic', cmap='viridis', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        ax_j.set_yticks([-10, -5, 0])
        ax_j.set_yticklabels(['10', '5', '0'])
        plt.xlabel('$x/z_0$')
        plt.ylabel('$z/z_0$')
        #plt.ylim(-6, 0)
        #plt.xlim(60, 78)
        x = np.linspace(0, L, Nx)
        #keel = -h*sigma**2/(sigma**2+4*(x-l)**2)
        plt.fill_between(x/(H-z0), 0, keel(2+0.07)/(H-z0), facecolor="white")
        plt.plot(x/(H-z0), keel(2+0.07)/(H-z0), linewidth=0.5, color='black')
        plt.tight_layout()
        fig_j.set_dpi(d)
        plt.savefig('methods.pdf', bbox_inches='tight')
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
    with h5py.File('regime_files/data-mixingsim-a102c005-00_s70.h5', mode='r') as f:
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
        u = f['tasks']['u'][0][Ni_x:Nf_x, Ni_z:Nf_z]
        deriv2 = np.gradient(rho_ref, z[0], axis=1, edge_order=1)
        rho_z = np.gradient(rho, z[0], axis=1, edge_order=2)
        rho_x = np.gradient(rho, np.concatenate(x).ravel(), axis=0, edge_order=2)	
        u_z = np.gradient(u, z[0], axis=1)	
        nabla_rho = rho_x**2+rho_z**2
        #nabla_rho[nabla_rho < 1e-3] = 0
        integrand1['g'] = -9.8*mu*np.nan_to_num(1/deriv2, posinf=0, neginf=0, nan=0)*nabla_rho
        phi_d = de.operators.integrate(integrand1, 'x').evaluate()['g'][0]
        integrand = -9.8*np.nan_to_num(1/deriv2, posinf=0, neginf=0, nan=0)*(nabla_rho)/(sw.dens0(28,-2))
        N_sq = -9.8/sw.dens0(28,-2)*np.average(np.average(np.gradient(rho_ref, z[0], axis=1), axis=0))
        #f = 0.5*(1+np.tanh((z-32+16*sigma**2/(sigma**2+4*(x-l)**2))/0.01))
        #Fr = abs(f['tasks']['u'][0][Ni_x:Nf_x, Ni_z:Nf_z])/np.sqrt(8*9.8*(rho-sw.dens0(28,-2))/sw.dens0(28,-2))
        Ri = -9.8/sw.dens0(28,-2) * rho_z / (u_z**2)
        fig_j, ax_j = plt.subplots(figsize=(16,12))
        im_j = ax_j.imshow(np.transpose(Ri), vmin=0, vmax=1, origin='lower', cmap='viridis', extent=(L_1/(H-z0), L_2/(H-z0), 80/(H-z0), 0))
        plt.contour(np.linspace(L_1, L_2, Nf_x-Ni_x)/(H-z0), np.linspace(0, H, Nf_z-Ni_z)[::-1]/(H-z0), np.transpose(rho), [0.5*(sw.dens0(28,-2)+sw.dens0(30,-2))], linewidths=0.85, colors='black')
        plt.ylim(plt.ylim()[::1])
        plt.xlim(L_1/(H-z0), L_2/(H-z0))
        fig_j.colorbar(im_j, orientation='horizontal', label='$Ri$')
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
    plt.rcParams.update({'font.size':20})
    #Boundary layer figure
    u = []
    w = []
    v = []
    k = []
    rho = []
    #titles = ['(a) $t=132t_0$', '(b) $t=192t_0$']
    titles = ['(a)', '(b)']
    for i in range(2):
        with h5py.File('regime_files/data-mixingsim-a102c200-00_s{0}.h5'.format(['313', '410'][i]), mode='r') as f:
            u.append(f['tasks']['u'][0])
            w.append(f['tasks']['w'][0])
            v.append(np.sqrt(u[i]**2+w[i]**2))
            k.append(f['tasks']['vorticity'][0])
            rho.append(f['tasks']['rho'][0])
    fig, axes = plt.subplots(1,2, figsize=(18,3))
    for i in range(len(axes)):
        axes[i].contour(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(rho[i]), [0.5*(sw.dens0(28,-2)+sw.dens0(30,-2))], colors='k', linewidths=2.0)
        pcm = axes[i].imshow(np.transpose(k[i]), vmin=-0.2, vmax=0.2, cmap='bwr', origin='lower', interpolation='bicubic', extent=(0, L/(H-z0), -H/(H-z0), 0))
        axes[i].fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(1.2)/(H-z0), facecolor="white", zorder=10)
        axes[i].plot(np.linspace(0, L, Nx)/(H-z0), keel(1.2)/(H-z0), linewidth=0.5, color='black')
        print(np.transpose(u[i])[300])
        #axes[i].streamplot(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(u[i]), np.transpose(w[i]), color='black', density=3.1, linewidth=0.4, arrowsize=1)
        axes[i].set_xlim(60, 75)
        axes[i].set_ylim(0, -4)
        axes[i].set_ylim(axes[i].get_ylim()[::-1])
        axes[i].set_yticks([0, -1, -2, -3, -4])
        axes[i].set_yticklabels(['', '', '', '', ''])
        axes[i].set_xticks([60, 65, 70, 75])
        axes[i].set_xlabel('$x/z_0$')
        axes[i].set_aspect('auto')
        axes[i].yaxis.set_label_coords(axes[i].yaxis.get_label().get_position()[0]-0.1, axes[i].yaxis.get_label().get_position()[1]-0.1)
        axes[i].text(74.2, -0.5, titles[i], fontsize=18, weight='bold', ha='center', va='center', zorder=10)
    axes[0].set_yticklabels(['0', '1', '2', '3', '4'])
    axes[0].set_ylabel('$z/z_0$', rotation=0)
    for i in range(len(axes)):
        axes[i].yaxis.set_label_coords(axes[i].yaxis.get_label().get_position()[0]+0.01, axes[i].yaxis.get_label().get_position()[1]-0.1)
    fig.subplots_adjust(right=0.8, wspace=0.1)
    cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.78])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label("Vorticity $q$ (s$^{{-1}}$)")
    #fig.set_size_inches(8,6, forward=True)
    plt.savefig('boundarylayer_figure.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print('done')
    plt.clf()

def bore():
    plt.rcParams.update({'font.size':20})
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
    fig, axes = plt.subplots(1,2, figsize=(18,3))
    for i in range(len(axes)):
        axes[i].contour(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(rhos[i]), [0.5*(sw.dens0(28,-2)+sw.dens0(30,-2))], colors='k', linewidths=2.0)
        #pcm = axes[i].imshow(np.transpose(us[i])/0.69, vmin=-0.8, vmax=0.8, cmap='bwr', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        pcm = axes[i].imshow(np.transpose(qs[i]), vmin=-0.2, vmax=0.2, cmap='bwr', interpolation='bicubic', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        #axes[i].streamplot(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(us[i]), np.transpose(ws[i]), color='black', density=2, linewidth=0.35, arrowsize=0.35)
        axes[i].fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(2)/(H-z0), facecolor="white", zorder=10)
        axes[i].plot(np.linspace(0, L, Nx)/(H-z0), keel(2)/(H-z0), linewidth=0.5, color='black') 
        axes[i].set_xlim(20, 80)
        axes[i].set_ylim(0, -4)
        axes[i].set_ylim(axes[i].get_ylim()[::-1])
        axes[i].set_yticks([0, -1, -2, -3, -4])
        axes[i].set_yticklabels(['', '', '', '', ''])
        axes[i].set_xlabel('$x/z_0$')
        axes[i].text(75, -0.5, titles[i], fontsize=18, weight='bold', ha='center', va='center', zorder=10)
        axes[i].yaxis.set_label_coords(axes[i].yaxis.get_label().get_position()[0]-0.1, axes[i].yaxis.get_label().get_position()[1]-0.1)
        axes[i].set_aspect('auto')
    axes[0].set_yticklabels(['0', '1', '2', '3', '4'])
    axes[0].set_ylabel('$z/z_0$', rotation=0)
    for i in range(len(axes)):
        axes[i].yaxis.set_label_coords(axes[i].yaxis.get_label().get_position()[0]+0.01, axes[i].yaxis.get_label().get_position()[1]-0.1)
    fig.set_dpi(d)
    fig.subplots_adjust(right=0.8, wspace=0.1)
    cbar_ax = fig.add_axes([0.83, 0.1, 0.02, 0.78])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label("Vorticity $q$ (s$^{{-1}}$)")
    #plt.gcf().set_size_inches(8,6, forward=True)
    plt.savefig('bore_figure.pdf', format='pdf', bbox_inches='tight')
    plt.clf()


def streamQuiver(ax,sp,*args,spacing=None,n=5,**kwargs):
    """ Plot arrows from streamplot data  
    The number of arrows per streamline is controlled either by `spacing` or by `n`.
    See `lines_to_arrows`.
    """
    def curve_coord(line=None):
        """ return curvilinear coordinate """
        x=line[:,0]
        y=line[:,1]
        s     = np.zeros(x.shape)
        s[1:] = np.sqrt((x[1:]-x[0:-1])**2+ (y[1:]-y[0:-1])**2)
        s     = np.cumsum(s)                                  
        return s

    def curve_extract(line,spacing,offset=None):
        """ Extract points at equidistant space along a curve"""
        x=line[:,0]
        y=line[:,1]
        if offset is None:
            offset=spacing/2
        # Computing curvilinear length
        s = curve_coord(line)
        offset=np.mod(offset,s[-1]) # making sure we always get one point
        # New (equidistant) curvilinear coordinate
        sExtract=np.arange(offset,s[-1],spacing)
        # Interpolating based on new curvilinear coordinate
        xx=np.interp(sExtract,s,x);
        yy=np.interp(sExtract,s,y);
        return np.array([xx,yy]).T

    def seg_to_lines(seg):
        """ Convert a list of segments to a list of lines """ 
        def extract_continuous(i):
            x=[]
            y=[]
            # Special case, we have only 1 segment remaining:
            if i==len(seg)-1:
                x.append(seg[i][0,0])
                y.append(seg[i][0,1])
                x.append(seg[i][1,0])
                y.append(seg[i][1,1])
                return i,x,y
            # Looping on continuous segment
            while i<len(seg)-1:
                # Adding our start point
                x.append(seg[i][0,0])
                y.append(seg[i][0,1])
                # Checking whether next segment continues our line
                Continuous= all(seg[i][1,:]==seg[i+1][0,:])
                if not Continuous:
                    # We add our end point then
                    x.append(seg[i][1,0])
                    y.append(seg[i][1,1])
                    break
                elif i==len(seg)-2:
                    # we add the last segment
                    x.append(seg[i+1][0,0])
                    y.append(seg[i+1][0,1])
                    x.append(seg[i+1][1,0])
                    y.append(seg[i+1][1,1])
                i=i+1
            return i,x,y
        lines=[]
        i=0
        while i<len(seg):
            iEnd,x,y=extract_continuous(i)
            lines.append(np.array( [x,y] ).T)
            i=iEnd+1
        return lines

    def lines_to_arrows(lines,n=5,spacing=None,normalize=True):
        """ Extract "streamlines" arrows from a set of lines 
        Either: `n` arrows per line
            or an arrow every `spacing` distance
        If `normalize` is true, the arrows have a unit length
        """
        if spacing is None:
            # if n is provided we estimate the spacing based on each curve lenght)
            spacing = [ curve_coord(l)[-1]/n for l in lines]
        try:
            len(spacing)
        except:
            spacing=[spacing]*len(lines)

        lines_s=[curve_extract(l,spacing=sp,offset=sp/2)         for l,sp in zip(lines,spacing)]
        lines_e=[curve_extract(l,spacing=sp,offset=sp/2+0.01*sp) for l,sp in zip(lines,spacing)]
        arrow_x  = [l[i,0] for l in lines_s for i in range(len(l))]
        arrow_y  = [l[i,1] for l in lines_s for i in range(len(l))]
        arrow_dx = [le[i,0]-ls[i,0] for ls,le in zip(lines_s,lines_e) for i in range(len(ls))]
        arrow_dy = [le[i,1]-ls[i,1] for ls,le in zip(lines_s,lines_e) for i in range(len(ls))]

        if normalize:
            dn = [ np.sqrt(ddx**2 + ddy**2) for ddx,ddy in zip(arrow_dx,arrow_dy)]
            arrow_dx = [ddx/ddn for ddx,ddn in zip(arrow_dx,dn)] 
            arrow_dy = [ddy/ddn for ddy,ddn in zip(arrow_dy,dn)] 
        return  arrow_x,arrow_y,arrow_dx,arrow_dy 

    # --- Main body of streamQuiver
    # Extracting lines
    seg   = sp.lines.get_segments() # list of (2, 2) numpy arrays
    lines = seg_to_lines(seg)       # list of (N,2) numpy arrays
    # Convert lines to arrows
    ar_x, ar_y, ar_dx, ar_dy = lines_to_arrows(lines,spacing=spacing,n=n,normalize=True)
    # Plot arrows
    qv=ax.quiver(ar_x, ar_y, ar_dx, ar_dy, *args, angles='xy', **kwargs)
    return qv

def regime_graphic():
    #density regimes
    titles1 = ['(a) Unstable Supercritical', '(b) Unstable Subcritical', '(c) Stable Subcritical']
    #titles2 = ['(a) Vortex Shedding', '(b) Diffusive Boundary Layer', '(c) Fast-laminar', '(d) Lee Waves']
    titles2 = ['(a) Vortex Shedding', '(b) Fast-laminar', '(c) Lee Waves', '(d) Diffusive Boundary Layer']
    a_s = [1.2, 2, 1.2]
    c_s = [2, 1.0, 0.5]
    a_s2 = [2, 0.5, 0.5, 2]
    c_s2 = [2, 1.0, 0.5, 0.5]
    fs1 = ['regime_files/data-mixingsim-a102c200-00_s410.h5', 
           'regime_files/data-mixingsim-a200c100-00_s179.h5', 
           'regime_files/data-mixingsim-a102c005-00_s150.h5'] 
    fs2 = ['regime_files/data-mixingsim-a200c200-00_s320.h5',
           'regime_files/data-mixingsim-a005c100-00_s240.h5', 
           'regime_files/data-mixingsim-a005c005-00_s180.h5',
           'regime_files/data-mixingsim-a200c005-00_s160.h5']
    rhos1 = []
    us1 = []
    ws1 = []
    qs1 = []
    rhos2 = []
    us2 = []
    ws2 = []
    qs2 = []
    i = 0
    for fi in fs1:
        with h5py.File(fi, mode='r') as f:
            rhos1.append(f['tasks']['rho'][0])
            us1.append(f['tasks']['u'][0])
            ws1.append(f['tasks']['w'][0])
            qs1.append(f['tasks']['vorticity'][0])
        i+=1
    for fi in fs2:
        with h5py.File(fi, mode='r') as f:
            rhos2.append(f['tasks']['rho'][0])
            us2.append(f['tasks']['u'][0])
            ws2.append(f['tasks']['w'][0])
            qs2.append(f['tasks']['vorticity'][0])
    plt.rcParams.update({'font.size':14})
    #Upstream
    #fig, axes = plt.subplots(3,1, sharex='col')
    gs = GridSpec(2,4)
    gs.update(wspace=0.4)
    ax1 = plt.subplot(gs[0,:2],)
    ax2 = plt.subplot(gs[0, 2:])
    ax3 = plt.subplot(gs[1, 1:3])
    plt.gcf().set_size_inches(16,3.5)
    axes = [ax1, ax2, ax3]
    paddingx = [0.0, 0.4, 0.4, 1]
    paddingy = [-0.01, -0.04, -0.01, -0.09]
    ms = {'P': 13, 'D': 10, 'o': 13}
    rho1 = sw.dens0(28,-2)
    rho2 = sw.dens0(30,-2)
    lengths = [3, 3, 1.5]
    lis = [[[65, -0.6], [67.5, -0.5], [69, -0.43], [20, -0.5], [20, -1], [20, -1.5], [20, -2], [20, -2.5], [20, -3], [20, -3.5]],[[30,-0.3], [59, -0.3], [65.8, -1.3], [20, -0.1],[20, -0.5], [20, -1], [20, -1.5], [20, -2], [20, -2.5], [20, -3], [20, -3.5]],[[35, -0.2], [20, -0.15],[20, -0.5], [20, -1], [20, -1.5], [20, -2], [20, -2.5], [20, -3], [20, -3.5]]]
    for i in range(len(axes)):
        #pcm = axes[i].imshow(np.transpose(rhos1[i])-sw.dens0(28,-2), vmin=0, vmax=2, cmap='viridis', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        #lw = 0.8*(z_rhovalues-1022.3)/(z_rhovalues[-1]-1022.3)
        #axes[i].contour(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(rhos1[i]), z_rhovalues, linewidths=lw, colors='black')
        pcm = axes[i].imshow(np.transpose(rhos1[i])-rho1, vmin=0, vmax=1.75, cmap='viridis', interpolation='bicubic', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        #pcm = axes[i, j].imshow(np.transpose(us1[k][:, ::-1]), cmap='bwr', vmin=-0.4, vmax=0.4, extent=(0, L/(H-z0), -H/(H-z0), 0))
        axes[i].fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(a_s[i])/(H-z0), facecolor="white", zorder=10)
        speed = np.sqrt(np.transpose(us1[i])**2+np.transpose(ws1[i])**2)
        U = c_s[i]*np.sqrt((H-z0)*DB)
        lw = 1.3*speed/U
        axes[i].plot(np.linspace(0, L, Nx)/(H-z0), keel(a_s[i])/(H-z0), linewidth=0.5, color='black')
        c = axes[i].streamplot(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(us1[i]), np.transpose(ws1[i]), arrowstyle='-', color='red', linewidth=lw, broken_streamlines=False, start_points=lis[i], maxlength=lengths[i])
        c.lines.set_alpha(0.5)
        for x in axes[i].get_children():
            if type(x) == matplotlib.patches.FancyArrowPatch:
                x.set_alpha(0.5)
        streamQuiver(axes[i], c, spacing=31, headwidth=3, scale=25, color='red')
        axes[i].set_xticks([25, 35, 45, 55, 65, 75])
        axes[i].set_yticklabels(['', '', '', '', ''])
        axes[i].set_xlim(20,75)
        axes[i].set_aspect("auto")
        axes[i].plot(44.3+0.8*len(titles1[i])+paddingx[i]-1.9, paddingy[i]+0.65, markeredgecolor='k', ms=ms[['P', 'o', 'D'][i]], marker=['P', 'o', 'D'][i], color=colors2[['P', 'o', 'D'][i]], clip_on=False)
        axes[i].set_title(titles1[i])
        axes[i].set_ylim(0,-4)
        axes[i].set_ylim(axes[i].get_ylim()[::-1])
        axes[i].set_xlabel('$x/z_0$', rotation=0)
        axes[i].yaxis.set_label_coords(axes[i].yaxis.get_label().get_position()[0]-0.1, axes[i].yaxis.get_label().get_position()[1]-0.1)
    axes[0].set_yticklabels(['4', '2', '0', '1', '0'])
    axes[2].set_yticklabels(['4', '2', '0', '1', '0'])
    axes[2].set_ylabel('$z/z_0$', rotation=0)
    axes[0].set_ylabel('$z/z_0$', rotation=0)
    for i in range(len(axes)):
        axes[i].yaxis.set_label_coords(axes[i].yaxis.get_label().get_position()[0]+0.01, axes[i].yaxis.get_label().get_position()[1]-0.1)
    axes[2].set_xticklabels(['25', '35', '45', '55', '65', '75'])
    plt.tight_layout()
    plt.gcf().subplots_adjust(right=0.8, hspace=0.55, wspace=0.1)
    cbar_ax = plt.gcf().add_axes([axes[2].get_position().x0, -0.09, axes[2].get_position().x1 - axes[2].get_position().x0, 0.04])
    cbar = plt.gcf().colorbar(pcm, orientation='horizontal', cax=cbar_ax)
    cbar.set_label("$\\rho-\\rho_1$ (kg m$^{{-3}}$)")
    plt.gcf().set_dpi(d)
    plt.savefig('regime_graphic_figure_up.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()


    #Downstream
    fig, axes = plt.subplots(2,2, figsize=(14.5,4), sharex='col')
    paddingx = [-1.2, -0.6, 1.2, -5]
    paddingy = [-0.01, -0.01, -0.03, -0.01, -0.09]
    ms = {'*': 17.5, '^': 13, 's': 12, 'p':16}
    lis = [[[75, -2], [75, -2.5], [75, -3], [75, -3.5], [75, -4]], [[75, -1], [75, -1.5], [75, -2], [75, -2.5], [75, -3], [75, -3.5], [75, -4]], [[75, -1], [75, -1.5], [75, -2], [75, -2.5], [75, -3], [75, -3.5], [75, -4]], [[75, -2.5], [75, -3], [75, -3.5], [75, -4], [75, -4.5]]]
    axes = axes.flatten()
    for i in range(len(axes)):
        #pcm = axes[i].imshow(np.transpose(rhos2[i])-sw.dens0(28,-2), vmin=0, vmax=2, cmap='viridis', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        #lw = 1.1*(z_rhovalues-1022.3)/(z_rhovalues[-1]-1022.3)
        #axes[i].contour(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(rhos2[i]), z_rhovalues, linewidths=lw, colors='black')
        pcm = axes[i].imshow(np.transpose(rhos2[i])-rho1, vmin=0, vmax=1.75, cmap='viridis', interpolation='bicubic', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
        axes[i].fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(a_s2[i])/(H-z0), facecolor="white", zorder=10)
        speed = np.sqrt(np.transpose(us2[i])**2+np.transpose(ws2[i])**2)
        U = c_s2[i]*np.sqrt((H-z0)*DB)
        lw = 1.3*speed/U
        axes[i].plot(np.linspace(0, L, Nx)/(H-z0), keel(a_s2[i])/(H-z0), linewidth=0.5, color='black')
        c = axes[i].streamplot(np.linspace(0, L, Nx)/(H-z0), -np.linspace(0, H, Nz)[::-1]/(H-z0), np.transpose(us2[i]), np.transpose(ws2[i]), density=6, arrowstyle='-', color='red', linewidth=lw, broken_streamlines=False, start_points=lis[i], )
        c.lines.set_alpha(0.5)
        for x in axes[i].get_children():
            if type(x) == matplotlib.patches.FancyArrowPatch:
               x.set_alpha(0.5)
        streamQuiver(axes[i], c, spacing=40, headwidth=3, scale=25, color='red')
        axes[i].set_xticks([75, 85, 95, 105, 115])
        axes[i].set_xlim(75,115)
        axes[i].set_aspect("auto")
        axes[i].plot(91.6+0.8*len(titles2[i])+paddingx[i]-1.7, paddingy[i]+0.6, ms=ms[['*', '^', 's', 'p'][i]], markeredgecolor='k', marker=['*', '^', 's', 'p'][i], color=colors2[['*', '^', 's', 'p'][i]], clip_on=False)
        axes[i].set_title(titles2[i])
        axes[i].set_ylim(0,-4)
        axes[i].set_yticklabels(['4', '2', '0', '1', '0'])
        axes[i].yaxis.set_label_coords(axes[i].yaxis.get_label().get_position()[0]-0.1, axes[i].yaxis.get_label().get_position()[1]-0.1)
        axes[i].set_ylim(axes[i].get_ylim()[::-1])
    axes[0].set_ylabel('$z/z_0$', rotation=0)
    axes[2].set_ylabel('$z/z_0$', rotation=0)
    axes[2].set_xlabel('$x/z_0$')
    axes[3].set_xlabel('$x/z_0$')
    axes[1].set_yticklabels([])
    axes[3].set_yticklabels([])
    axes[2].set_xticklabels(['75', '85', '95', '105', '115'])
    axes[3].set_xticklabels(['75', '85', '95', '105', '115'])
    for i in range(len(axes)):
        axes[i].yaxis.set_label_coords(axes[i].yaxis.get_label().get_position()[0]+0.01, axes[i].yaxis.get_label().get_position()[1]-0.1)
    plt.tight_layout()
    plt.gcf().subplots_adjust(right=0.8, hspace=0.55, wspace=0.10)
    cbar_ax = plt.gcf().add_axes([0.82, 0.18, 0.02, 0.68])
    cbar = plt.gcf().colorbar(pcm, orientation='vertical', cax=cbar_ax)
    cbar.set_label("$\\rho-\\rho_1$ (kg m$^{{-3}}$)")
    fig.set_dpi(d)
    plt.savefig('regime_graphic_figure_down.pdf', format='pdf', dpi=d, bbox_inches='tight')
    plt.clf()

def zmix():
    #zmix/z0, separate plots upstream/downstream, with keel
    a_axis = [0.5, 0.95, 1.2, 2.0] #eta
    block_width, block_depth = 0.1, 0.1

    figU = plt.figure(figsize=(7.5,6))
    ax = figU.gca(projection='3d')
    ax.view_init(elev=30, azim=225)
    for i in range(len(a)):
        for j in range(len(c)):
            ax.bar3d(c_axis[j], a_axis[i], 0, block_width, block_depth, avgs[i]['z_mix_U'][j]/z0, color=colors2[markers1[i][j]], edgecolor='k', shade=True)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['D'], mec='k', label='LBR', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['o'], mec='k', label='Solitary Waves', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['P'], mec='k', label='Supercritical', ms=11)
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 15}, fancybox=True, shadow=True)
    ax.xaxis.set_rotate_label(False)
    ax.set_xlabel('\n$Fr$', linespacing=3.2)
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('\n$\\eta$', linespacing=3.2)
    ax.set_yticks([0.5, 1, 1.5, 2])
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('$z_{mix}/z_0$       ')
    ax.set_zlim(0, 0.42)
    ax.set_zticks([0, 0.1, 0.2, 0.3, 0.4])
    plt.tight_layout()
    figU.savefig('zmix_Upstream_figure.pdf', format='pdf')
    plt.clf()

    figD = plt.figure(figsize=(7.5,6))
    ax = figD.gca(projection='3d')
    ax.view_init(elev=30, azim=225)
    for i in range(len(a)):
        for j in range(len(c)):
            ax.bar3d(c_axis[j], a_axis[i], 0, block_width, block_depth, avgs[i]['z_mix_D'][j]/z0, color=colors2[markers2[i][j]], edgecolor='k', shade=True)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['s'], mec='k', label='Lee Waves', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['^'], mec='k', label='Quasi-laminar', ms=11)
    ax.plot([], [], marker='s', linestyle='None', color=colors2['*'], mec='k', label='Vortex Shedding', ms=11)
    handles, labels_temp = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_temp[::-1], handles[::-1]))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', prop={'size': 15}, fancybox=True, shadow=True)
    ax.xaxis.set_rotate_label(False)
    ax.set_xlabel('\n$Fr$', linespacing=3.2)
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel('\n$\\eta$', linespacing=3.2)
    ax.set_yticks([0.5, 1, 1.5, 2])
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('$z_{mix}/z_0$       ')
    ax.set_zlim(0, 0.42)
    ax.set_zticks([0, 0.1, 0.2, 0.3, 0.4])
    plt.tight_layout()
    figD.savefig('zmix_Downstream_figure.pdf', format='pdf')
    plt.clf()

def KED_analyze():
    #print('F20H20 Average upstream TOTAL energy dissipation: {0}  with KED = {1} and Phi = {2}'.format(avgs[3]['Phi_d_U'][3]+avgs[3]['KED_U'][3], avgs[3]['KED_U'][3], avgs[3]['Phi_d_U'][3]))
    #print('F20H20 Average downstream TOTAL energy dissipation: {0}  with KED = {1} and Phi = {2}\n'.format(avgs[3]['Phi_d_D'][3]+avgs[3]['KED_D'][3], avgs[3]['KED_D'][3], avgs[3]['Phi_d_D'][3]))
    #print('F15H20 Average upstream TOTAL energy dissipation: {0}  with KED = {1} and Phi = {2}'.format(avgs[3]['Phi_d_U'][2]+avgs[3]['KED_U'][2], avgs[3]['KED_U'][2], avgs[3]['Phi_d_U'][2]))
    #print('F15H20 Average downstream TOTAL energy dissipation: {0}  with KED = {1} and Phi = {2}'.format(avgs[3]['Phi_d_D'][2]+avgs[3]['KED_D'][2], avgs[3]['KED_D'][2], avgs[3]['Phi_d_D'][2]))
    fig, axs = plt.subplots(4,4, figsize=(10,6))
    plt.rcParams.update({'font.size':13})
    for i in range(len(a)):
        for j in range(len(c)):
            axs[i,j].plot(avgs[i]['K_p_D_series'][j][1]/t_0, avgs[i]['K_p_D_series'][j][0])
            axs[i,j].set_title(conv_id[c[j]]+conv_id[a[i]])
    axs[0, 0].set_ylabel('$K_D$')
    axs[1, 0].set_ylabel('$K_D$')
    axs[2, 0].set_ylabel('$K_D$')
    axs[3, 0].set_ylabel('$K_D$')
    axs[3, 0].set_xlabel('$t/t_0$')
    axs[3, 1].set_xlabel('$t/t_0$')
    axs[3, 2].set_xlabel('$t/t_0$')
    axs[3, 3].set_xlabel('$t/t_0$')
    #axs[3, 3].set_ylim(0,10)
    plt.tight_layout()
    plt.savefig('dnseries.png')
    plt.clf()

    fig, axs = plt.subplots(1,1, figsize=(10,6))
    for i in range(len(a)-1):
        for j in range(len(c)):
            print(avgs[i]['Phi_d_U_series'][j][1][33])
            axs.plot([0.5,0.95,1.2][i], avgs[i]['Phi_d_U_series'][j][0][33]/avgs[0]['Phi_d_U_series'][0][0][33], marker='o', linestyle='None', color=['k', 'b', 'r', 'aquamarine'][j])
    plt.ylim(0.6,1.1)
    plt.xlabel('$\\eta$')
    plt.ylabel('$\\Phi/\\Phi_0$')
    plt.title('Sam Upstream mixing $t=100.22t_0$')
    plt.savefig('test.png')
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
#joint_regime()
#joint_regime_ms()
#joint_regime_arctic()
bore()
boundary_layer()
regime_graphic()
#K_p_downstream_var4()
#K_p_upstream_var4()
#K_p_downstream_var5()
#K_p_upstream_var5()
#K_p_subplots_4()
phi_d_subplots()
#methods()
#zmix()
#KED_analyze()
