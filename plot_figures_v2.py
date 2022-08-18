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

def keel(a):
    h = a*(H-z0)
    sigma = 3.9*h
    return -h*sigma**2/(sigma**2+4*(np.linspace(0, L, Nx)-l)**2)
d = 100
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

xbasis = de.Chebyshev('x', 1280, interval=(0, L))
zbasis = de.Chebyshev('z', 640, interval=(0, H))
domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)

def average_data(d):
    return np.mean(d[-30:]) 

def stdev_data(d):
    return np.std(d[-30:])

def stdev_log_data(d):
    return np.std(np.log10(d[-30:]))

def sort_rho_z(h5_file, domain):
	with h5py.File(h5_file, mode='r') as f:
		x, z = domain.grids(domain.dealias)
		rho = f['tasks']['rho'][0]
		ind = z[0][np.argsort(np.argsort(-rho, axis=None))//(1280)]
		z_sort_height = np.reshape(ind, (1280, 640))
		rho_sort = np.reshape(-np.sort(-rho.flatten()), (1280, 640), order='F')
	return rho_sort, z_sort_height

plt.rcParams.update({'font.size':13})

x, z = domain.grids()
#a = ['a005', 'a009', 'a102', 'a200'] #heights
#c = ['c005', 'c100', 'c105', 'c200'] #speeds
sp = [[220, 260, 450, 450], [220, 260, 450, 450], [220, 260, 450, 450], [220, 260, 450, 450]]
a = ['a005', 'a095', 'a102', 'a200']
c = ['c005', 'c100', 'c105', 'c200']
labels_height = ['$h/z_0=0.5$', '$h/z_0=0.95$', '$h/z_0=1.2$', '$h/z_0=2.0$']
labels_regime_up = ['Supercritical', 'Rarefaction',  'Solitons', 'Blocked']
labels_regime_down = ['Vortex shedding', 'Stirred', 'Laminar jump', 'Blocked', 'Lee waves']
markers_labels_up = ['d', 'D', 'o', 's']
markers_labels_down = ['*', 'p', '^', 's', 'P']
markers1 = [['D', 'D', 'D', 'd'], ['D', 'o', 'o', 'd'], ['D', 'o', 'o', 'd'], ['s', 'o', 'o', 'd']] #vortex shedding = star, bore & MSD = circle, blocking = square, bore & TD = triangle, MSD=diamond, lee = plus
markers2 = [['P', '^', '^', '*'], ['P', '^', 'p', '*'], ['P', '^', 'p', '*'], ['s', '^', 'p', '*']] #vortex shedding = star, bore & MSD = circle, blocking = square, bore & TD = triangle, MSD=diamond, lee = plus
titles = ["a) Vortex Shedding", "b) Solitons & Turbulent Downstream", "c) Solitons & Minimum Stirring Downstream", "d) Minimum Stirring Downstream", "e) Blocking", "f) Lee waves"]
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
    avgs[i]['K_p_U_stdev'] = []
    avgs[i]['K_p_D'] = []
    avgs[i]['K_p_D_stdev'] = []
    avgs[i]['K_p_D_series'] = []
    avgs[i]['K_p_U_series'] = []
    for j in range(len(c)):
        avgs[i]['MLD'].append(average_data(data[i][j][7]))
        avgs[i]['MLD_stdev'].append(MLD_std[i][j])
        avgs[i]['KED_D'].append(average_data(data[i][j][15]))
        avgs[i]['KED_D_stdev'].append(stdev_data(data[i][j][15]))
        avgs[i]['Phi_d_D'].append(average_data(data[i][j][13]))
        avgs[i]['Phi_d_D_stdev'].append(stdev_data(data[i][j][13]))
        avgs[i]['KED_U'].append(average_data(data[i][j][22]))
        avgs[i]['KED_U_stdev'].append(stdev_data(data[i][j][22]))
        avgs[i]['Phi_d_U'].append(average_data(data[i][j][20]))
        avgs[i]['Phi_d_U_stdev'].append(stdev_data(data[i][j][20]))
        avgs[i]['K_p_U'].append(average_data(data[i][j][23]))
        avgs[i]['K_p_U_stdev'].append(stdev_data(data[i][j][23]))
        avgs[i]['K_p_D'].append(average_data(data[i][j][16]))
        avgs[i]['K_p_D_stdev'].append(stdev_data(data[i][j][16]))
        avgs[i]['K_p_D_series'].append(data[i][j][16])
        avgs[i]['K_p_U_series'].append(data[i][j][23])
        avgs[i]['time'].append(data[i][j][0])


c_axis = [0.5, 1.0, 1.5, 2]
colors1 = ['#4d94ff', '#005ce6', '#003380', '#001433']
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
plt.ylabel(r'Downstream Dissipation Rate $\langle\overline{{\varepsilon_k}}\rangle_D/\sqrt{{\Delta B^3 z_0}}$')
plt.ylim(0, plt.ylim()[1])
plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
plt.grid()



plt.savefig('KED_Downstream_figure.png', dpi=d, bbox_inches='tight')
plt.clf()

#KED Upstream
for i in range(len(a)):
    for j in range(len(c)):
        plt.plot(c_axis[j], avgs[i]['KED_U'][j]/epsilon_0, marker=markers1[i][j], linestyle='None', color=colors1[i], label=labels_height[i])    
plt.xlabel('Froude Number $Fr$')
plt.ylabel(r'Upstream Dissipation Rate $\langle\overline{{\varepsilon_k}}\rangle_U/\sqrt{{\Delta B^3 z_0}}$')
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

#K_p Upstream
for i in range(len(a)):
    for j in range(len(c)):
        plt.plot(c_axis[j], avgs[i]['K_p_U'][j]/mu, marker=markers1[i][j], linestyle='None', color=colors1[i], label=labels_height[i])    
plt.xlabel('Froude Number $Fr$')
plt.ylabel(r'Upstream Diapycnal Diffusivity $\langle \overline{{K_\rho}}\rangle_U/\mu$')
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
plt.savefig('Kp_Upstream_figure.png', dpi=d, bbox_inches='tight')
plt.clf()

#K_p Downstream
for i in range(len(a)):
    for j in range(len(c)):
        plt.plot(c_axis[j], avgs[i]['K_p_D'][j]/mu, linestyle='None', marker=markers2[i][j], color=colors1[i], label=labels_height[i])    
plt.xlabel('Froude number $Fr$')
plt.ylabel(r'Downstream Diapycnal Diffusivity $\langle \overline{{K_\rho}}\rangle_D/\mu$')
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
plt.savefig('Kp_Downstream_figure.png', dpi=d, bbox_inches='tight')
plt.clf()

"""#salt upstream
for i in range(len(a)):
    for j in range(len(c)):
        plt.errorbar(c_axis[j], salt_data_up[i][j], capsize=5, linestyle='None', marker=markers[i][j], color=colors1[i], label=labels[i])    
        if j == len(c)-1:
            plt.plot([], [], marker='o', linestyle='None', color=colors1[i], label=labels[i])
plt.xlabel('$U/\\sqrt{{z_0 \\Delta B}}$')
plt.ylabel('$R_{{avg}}/\\Delta C$')
plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
plt.grid()
plt.title('Upstream')
handles, labels_temp = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels_temp[::-1], handles[::-1]))
plt.legend(by_label.values(), by_label.keys())
plt.savefig('salt_Upstream_figure.png')
plt.clf()

#salt downstream
for i in range(len(a)):
    for j in range(len(c)):
        plt.errorbar(c_axis[j], salt_data_down[i][j], capsize=5, linestyle='None', marker=markers[i][j], color=colors1[i], label=labels[i])    
        if j == len(c)-1:
            plt.plot([], [], marker='o', linestyle='None', color=colors1[i], label=labels[i])
plt.xlabel('$U/\\sqrt{{z_0 \\Delta B}}$')
plt.ylabel('$R_{{avg}}/\\Delta C$')
plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
plt.grid()
plt.title('Downstream')
handles, labels_temp = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels_temp[::-1], handles[::-1]))
plt.legend(by_label.values(), by_label.keys())
plt.savefig('salt_Downstream_figure.png')
plt.clf()"""
colors2 = {'D': '#00b300', '^': '#0052cc', 's': '#6600cc', 'o': '#c908a6', 'P': '#ffa31a', '*': '#ff1a1a', 'p': 'black', 'd': '#800000'}
#vortex shedding = star, bore & MSD = circle, blocking = square, bore & TD = triangle, MSD=diamond, lee = plus
#Regime layout upstream
for i in range(len(a)):
    for j in range(len(c)):
        plt.errorbar(c_axis[j], [0.5, 0.95, 1.2, 2][i], capsize=5, linestyle='None', marker=markers1[i][j], color=colors2[markers1[i][j]])    
plt.plot([], [], marker='s', linestyle='None', color=colors2['s'], label='Blocked')
plt.plot([], [], marker='o', linestyle='None', color=colors2['o'], label='Solitons')
plt.plot([], [], marker='D', linestyle='None', color=colors2['D'], label='Rarefaction')
plt.plot([], [], marker='d', linestyle='None', color=colors2['d'], label='Supercritical')
plt.xlabel('Froude Number $Fr$')
plt.ylabel('$h/z_0$')
plt.xticks([0.5, 1, 1.5, 2])
plt.yticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
plt.grid()
plt.title('Upstream')
handles, labels_temp = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels_temp[::-1], handles[::-1]))
plt.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(0.944, 0.73), prop={'size': 11}, fancybox=True, shadow=True)
plt.savefig('regime_figure_upstream.png', dpi=d, bbox_inches='tight')
plt.clf()

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
plt.ylabel('$h/z_0$')
plt.xticks([0.5, 1, 1.5, 2])
plt.yticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
plt.grid()
handles, labels_temp = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels_temp[::-1], handles[::-1]))
plt.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(0.944, 0.73), prop={'size': 11}, fancybox=True, shadow=True)
plt.savefig('regime_figure_downstream.png', dpi=d, bbox_inches='tight')
plt.clf()

#Mixing efficiency upstream
for i in range(len(a)):
    for j in range(len(c)):
        M = avgs[i]['Phi_d_U'][j]
        e_k = avgs[i]['KED_U'][j]
        plt.errorbar(c_axis[j], M/(M+e_k), capsize=5, linestyle='None', marker=markers1[i][j], color=colors1[i], label=labels_height[i])    
        if j == len(c)-1:
            plt.plot([], [], marker='o', linestyle='None', color=colors1[i], label=labels_height[i])
plt.xlabel('$U/\\sqrt{{z_0 \\Delta B}}$')
plt.ylabel('$\\Gamma$')
plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
plt.grid()
plt.ylim(0, plt.ylim()[1])
plt.title('Upstream')
handles, labels_temp = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels_temp[::-1], handles[::-1]))
plt.legend(by_label.values(), by_label.keys())
plt.savefig('efficiency_Upstream_figure.png', dpi=d, bbox_inches='tight')
plt.clf()

#Mixing efficiency downstream
for i in range(len(a)):
    for j in range(len(c)):
        M = avgs[i]['Phi_d_D'][j]
        e_k = avgs[i]['KED_D'][j]
        plt.errorbar(c_axis[j], M/(M+e_k), capsize=5, linestyle='None', marker=markers2[i][j], color=colors1[i], label=labels_height[i])    
        if j == len(c)-1:
            plt.plot([], [], marker='o', linestyle='None', color=colors1[i], label=labels_height[i])
plt.xlabel('$U/\\sqrt{{z_0 \\Delta B}}$')
plt.ylabel('$\\Gamma$')
plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
plt.grid()
plt.ylim(0, plt.ylim()[1])
plt.title('Downstream')
handles, labels_temp = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels_temp[::-1], handles[::-1]))
plt.legend(by_label.values(), by_label.keys())
plt.savefig('efficiency_Downstream_figure.png', dpi=d, bbox_inches='tight')
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
 

with h5py.File('data-mixingsim-Test-00/data-mixingsim-Test-00_s1.h5', mode='r') as f:
    rho = f['tasks']['rho'][0]
    fig_j, ax_j = plt.subplots(figsize=(16,12))
    im_j = ax_j.imshow(np.transpose(rho), vmin=np.min(rho)-0.5, vmax=np.max(rho)+0.5, cmap='viridis', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
    ax_j.set_yticks([-10, -8, -6, -4, -2, 0])
    ax_j.set_yticklabels(['10', '8', '6', '4', '2', '0'])
    plt.xlabel('$x/z_0$')
    plt.ylabel('$z/z_0$')
    x = np.linspace(0, L, Nx)
    #keel = -h*sigma**2/(sigma**2+4*(x-l)**2)
    plt.fill_between(x/(H-z0), 0, keel(2)/(H-z0), facecolor="white")
    plt.plot(x/(H-z0), keel(2)/(H-z0), linewidth=0.5, color='black')
    plt.tight_layout()
    fig_j.set_dpi(80)
    plt.savefig('methods.png', bbox_inches='tight')
    plt.clf() 

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

plt.plot([], [], color=colors3['005'], marker='o', linestyle='None', label='$h/z_0=0.5$')
plt.plot([], [], color=colors3['095'], marker='o', linestyle='None', label='$h/z_0=0.95$')
plt.plot([], [], color=colors3['102'], marker='o', linestyle='None', label='$h/z_0=1.2$')
plt.plot([], [], color=colors3['200'], marker='o', linestyle='None', label='$h/z_0=2$')
plt.plot([], [], color='black', linestyle=styles['005'], label='$U/\\sqrt{{z_0 \\Delta B}}=0.5$')
plt.plot([], [], color='black', linestyle=styles['100'], label='$U/\\sqrt{{z_0 \\Delta B}}=1$')
plt.plot([], [], color='black', linestyle=styles['105'], label='$U/\\sqrt{{z_0 \\Delta B}}=1.5$')
plt.plot([], [], color='black', linestyle=styles['200'], label='$U/\\sqrt{{z_0 \\Delta B}}=2$')
plt.legend(fancybox=True, shadow=True, loc='center')
plt.savefig('temp2.png', dpi=d)

with h5py.File('regime_files/data-mixingsim-a102c005-00_s70.h5', mode='r') as f:
    rho = f['tasks']['rho'][0]
    u = f['tasks']['u'][0]
    w = f['tasks']['w'][0]
    x, z = domain.grids(domain.dealias)
    integrand1 = domain.new_field()
    integrand1.set_scales(domain.dealias)
    u_x = np.gradient(u, np.concatenate(x).ravel(), axis=0, edge_order=2)
    u_z = np.gradient(u, z[0], axis=1, edge_order=2)
    w_x = np.gradient(w, np.concatenate(x).ravel(), axis=0, edge_order=2)
    w_z = np.gradient(w, z[0], axis=1, edge_order=2)
    rho_ref, z_ref = sort_rho_z('regime_files/data-mixingsim-a102c005-00_s70.h5', domain)
    deriv = np.gradient(z_ref, z[0], axis=1, edge_order=2)/np.gradient(rho, z[0], axis=1, edge_order=2)+np.gradient(z_ref, np.concatenate(x).ravel(), axis=0, edge_order=2)/np.gradient(rho, np.concatenate(x).ravel(), axis=0, edge_order=2)
    deriv2 = np.gradient(rho_ref, z[0], axis=1, edge_order=1)
    rho_z = np.gradient(rho, z[0], axis=1, edge_order=2)
    rho_x = np.gradient(rho, np.concatenate(x).ravel(), axis=0, edge_order=2)	
    integrand1['g'] = -9.8*mu*np.nan_to_num(1/deriv2, posinf=0, neginf=0, nan=0)*(rho_x**2+rho_z**2)
    phi_d = de.operators.integrate(integrand1, 'x').evaluate()['g'][0]
    
    #rho_dev = rho - np.average(rho, axis=0)
    #rho_dev_x = np.gradient(rho_dev, np.concatenate(x).ravel(), axis=0, edge_order=2)
   # rho_dev_z = np.gradient(rho_dev, z[0], axis=1, edge_order=2)
   # integrand = np.average(rho_dev_x**2+rho_dev_z**2, axis=0)/(np.gradient(np.average(rho, axis=0), z[0], edge_order=2))**2
    #plt.plot(integrand, -(H-z[0])/(H-z0))
    #plt.ylim(plt.ylim()[::1])
    #plt.xlim(0, 1e5)
    #plt.savefig('jghjgjg.png')
    #plt.clf()
    integrand = mu*(2*u_x**2+2*w_z**2+(u_z+w_x)**2)/epsilon_0
    #integrand = -9.8*np.nan_to_num(1/deriv2, posinf=0, neginf=0, nan=0)*(rho_x**2+rho_z**2)/(sw.dens0(28,-2))
    #N_sq = -9.8/sw.dens0(28,-2)*np.average(np.average(np.gradient(rho_ref, z[0], axis=1), axis=0))
    #integrand = 2*mu*(u_x**2+w_z**2+u_z**2+w_x**2)/np.sqrt(DB**3*z0)
    #integrand = (-9.8/sw.dens0(28,-2)*rho_z)/u_z**2
    fig_j, ax_j = plt.subplots(figsize=(16,12))
    im_j = ax_j.imshow(np.transpose(integrand), vmin=0, vmax=1e-3, origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
    plt.ylim(plt.ylim()[::1])
    plt.xlim(64/(H-z0), (L-40)/(H-z0))
    fig_j.colorbar(im_j, orientation='horizontal', label='$dz_*/d\\rho |\\nabla\\rho|^2$')
    plt.plot(np.linspace(0, L, Nx)/(H-z0), keel(0.5)/(H-z0), color='red')
    plt.xlabel('$x/z_0$')
    plt.ylabel('$z/z_0$')
    plt.tight_layout()
    fig_j.set_dpi(300)
    plt.savefig('phid_integrand_region_a200c200e.png')
    plt.clf()
    cmap = plt.cm.hsv

    cyclicnorm = CyclicNormalize(cmin=100, cmax=100.2, vmin=np.min(rho), vmax=np.max(rho))

    fig, ax = plt.subplots(figsize=(12,12))
    pcm = ax.imshow(np.transpose(rho), vmin=1022, vmax=1025, cmap='plasma', origin='lower', norm=cyclicnorm, extent=(0, L/(H-z0), -H/(H-z0), 0))
    fig.colorbar(pcm, orientation='horizontal')
    fig.set_dpi(d)
    plt.savefig('testing.png')
    plt.tight_layout()
    plt.clf() 

#Bore figure
with h5py.File('regime_files/data-mixingsim-a200c100-00_s100.h5', mode='r') as f:
    u1 = f['tasks']['u'][0]
    w1 = f['tasks']['w'][0]
    v1 = np.sqrt(u1**2+w1**2)
with h5py.File('regime_files/data-mixingsim-a200c100-00_s119.h5', mode='r') as f:
    u2 = f['tasks']['u'][0]
    w2 = f['tasks']['w'][0]
    v2 = np.sqrt(u1**2+w1**2)
with h5py.File('regime_files/data-mixingsim-a200c100-00_s179.h5', mode='r') as f:
    u3 = f['tasks']['u'][0]
    w3 = f['tasks']['w'][0]
    v3 = np.sqrt(u1**2+w1**2)

fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex='col')
pcm = ax1.imshow(np.transpose(u1)/0.35, vmin=-0.40/0.35, vmax=0.40/0.35, cmap='bwr', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
pcm2 = ax2.imshow(np.transpose(u2)/0.35, vmin=-0.40/0.35, vmax=0.40/0.35, cmap='bwr', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
pcm3 = ax3.imshow(np.transpose(u3)/0.35, vmin=-0.40/0.35, vmax=0.40/0.35, cmap='bwr', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
fig.set_dpi(d)
ax1.streamplot(np.linspace(0, L, Nx)/(H-z0), -z[0][::-1]/(H-z0), np.transpose(u1), np.transpose(w1), color='black', density=2, linewidth=0.35, arrowsize=0.35)
ax2.streamplot(np.linspace(0, L, Nx)/(H-z0), -z[0][::-1]/(H-z0), np.transpose(u2), np.transpose(w2), color='black', density=2, linewidth=0.35, arrowsize=0.35)
ax3.streamplot(np.linspace(0, L, Nx)/(H-z0), -z[0][::-1]/(H-z0), np.transpose(u3), np.transpose(w3), color='black', density=2, linewidth=0.35, arrowsize=0.35)
ax1.fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(2)/(H-z0), facecolor="white", zorder=10)
ax2.fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(2)/(H-z0), facecolor="white", zorder=10)
ax3.fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(2)/(H-z0), facecolor="white", zorder=10)
ax1.plot(np.linspace(0, L, Nx)/(H-z0), keel(2)/(H-z0), linewidth=0.5, color='black')
ax2.plot(np.linspace(0, L, Nx)/(H-z0), keel(2)/(H-z0), linewidth=0.5, color='black')
ax3.plot(np.linspace(0, L, Nx)/(H-z0), keel(2)/(H-z0), linewidth=0.5, color='black')
fig.subplots_adjust(right=0.8, hspace=0.8)
ax1.set_xlim(20, 80)
ax2.set_xlim(20, 80)
ax3.set_xlim(20, 80)
ax1.set_ylim(0, -4)
ax2.set_ylim(0, -4)
ax3.set_ylim(0, -4)
ax1.set_ylim(ax1.get_ylim()[::-1])
ax2.set_ylim(ax2.get_ylim()[::-1])
ax3.set_ylim(ax3.get_ylim()[::-1])
ax1.set_yticks([0, -2, -4])
ax2.set_yticks([0, -2, -4])
ax3.set_yticks([0, -2, -4])
ax1.set_yticklabels(['0', '2', '4'])
ax2.set_yticklabels(['0', '2', '4'])
ax3.set_yticklabels(['0', '2', '4'])
ax3.set_xlabel('$x/z_0$')
ax3.set_ylabel('$z/z_0$')
ax2.set_ylabel('$z/z_0$')
ax1.set_ylabel('$z/z_0$')
ax1.set_title("a) $t=60t_0$")
ax2.set_title("b) $t=71t_0$")
ax3.set_title("c) $t=107t_0$")
ax1.set_aspect('auto')
ax2.set_aspect('auto')
ax3.set_aspect('auto')
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(pcm, cax=cbar_ax)
cbar.set_label("$u/U$")
plt.savefig('bore_figure.png', bbox_inches='tight')
plt.clf()

#density regimes
titles1 = ['Supercritical', 'Rarefaction', 'Solitons', 'Blocking']
titles2 = ['Vortex shedding', 'Stirred', 'Laminar jump', 'Blocked', 'Lee waves']
a_s = [2, 1.2, 1.2, 2]
a_s2 = [2, 0.95, 0.5, 2, 0.5]
fs1 = ['regime_files/data-mixingsim-a200c200-00_s160.h5',
      'regime_files/data-mixingsim-a102c005-00_s70.h5',
      'regime_files/data-mixingsim-a102c100-00_s180.h5',
      'regime_files/data-mixingsim-a200c005-00_s220.h5',]
fs2 = ['regime_files/data-mixingsim-a200c200-00_s320.h5',
      'regime_files/data-mixingsim-a095c105-00_s320.h5',
      'regime_files/data-mixingsim-a005c100-00_s240.h5',
      'regime_files/data-mixingsim-a200c005-00_s220.h5',
      'regime_files/data-mixingsim-a005c005-00_s180.h5']
rhos1 = []
rhos2 = []
for fi in fs1:
    with h5py.File(fi, mode='r') as f:
        rhos1.append(f['tasks']['rho'][0])
for fi in fs2:
    with h5py.File(fi, mode='r') as f:
        rhos2.append(f['tasks']['rho'][0])
plt.rcParams.update({'font.size':17})
#Upstream
fig, axes = plt.subplots(2,2, figsize=(13, 6))
k = 0
for i in range(len(axes)):
    for j in range(len(axes[0])):
        pcm = axes[i, j].imshow(np.transpose(rhos1[k])-sw.dens0(28,-2), vmin=0, vmax=2, cmap='viridis', origin='lower', extent=(0, L/(H-z0), H/(H-z0), 0))
        axes[i, j].fill_between(np.linspace(0, L, Nx)/(H-z0), 0, -keel(a_s[k])/(H-z0), facecolor="white", zorder=10)
        axes[i, j].plot(np.linspace(0, L, Nx)/(H-z0), -keel(a_s[k])/(H-z0), linewidth=0.5, color='black')
        axes[i, j].set_xticks([25, 50, 75])
        axes[i, j].set_xlim(20,75)
        axes[i, j].set_aspect("auto")
        axes[i, j].plot(72, 3.6, ms=15, marker=['d', 'D', 'p', 'o', 's'][k], color=colors2[['d', 'D', 'p', 'o', 's'][k]], clip_on=False)
        axes[i, j].set_title(titles1[k])
        axes[i, j].set_ylim(0,4)
        axes[i, j].set_ylim(axes[i, j].get_ylim()[::-1])
        if k == 2 or k == 3:
            axes[i, j].set_xlabel('$x/z_0$')
        if k == 0 or k == 2:
            axes[i, j].set_ylabel('$z/z_0$')
        k += 1
plt.tight_layout()
fig.subplots_adjust(right=0.8, hspace=0.6, wspace=0.3)
cbar_ax = fig.add_axes([0.835, 0.18, 0.02, 0.7])
cbar = fig.colorbar(pcm, cax=cbar_ax)
cbar.set_label("$\\rho-\\rho_1$")
fig.set_dpi(d)
plt.savefig('regime_graphic_figure_up.png', bbox_inches='tight')
plt.clf()


#Downstream
fig = plt.figure(figsize=(13, 6))
spec = GridSpec(ncols=6, nrows=2)
ax1 = fig.add_subplot(spec[0, 0:2])
ax2 = fig.add_subplot(spec[0, 2:4])
ax3 = fig.add_subplot(spec[0, 4:])
ax4 = fig.add_subplot(spec[1, 1:3])
ax5 = fig.add_subplot(spec[1, 3:5])
axes = [ax1, ax2, ax3, ax4, ax5]
for i in range(len(axes)):
    pcm = axes[i].imshow(np.transpose(rhos2[i])-sw.dens0(28,-2), vmin=0, vmax=2, cmap='viridis', origin='lower', extent=(0, L/(H-z0), H/(H-z0), 0))
    axes[i].fill_between(np.linspace(0, L, Nx)/(H-z0), 0, -keel(a_s2[i])/(H-z0), facecolor="white", zorder=10)
    axes[i].plot(np.linspace(0, L, Nx)/(H-z0), -keel(a_s2[i])/(H-z0), linewidth=0.5, color='black')
    axes[i].set_xticks([75, 95, 115])
    axes[i].set_xlim(75,115)
    axes[i].set_aspect("auto")
    axes[i].plot(112, 3.6, ms=15, marker=['*', 'p', '^', 's', 'P'][i], color=colors2[['*', 'p', '^', 's', 'P'][i]], clip_on=False)
    print(i)
    axes[i].set_title(titles2[i])
    axes[i].set_ylim(0,4)
    axes[i].set_ylim(axes[i].get_ylim()[::-1])
    axes[i].set_xlabel('$x/z_0$')
    if i == 0 or i == 3:
        axes[i].set_ylabel('$z/z_0$')
plt.tight_layout()
fig.subplots_adjust(right=0.8, hspace=0.6, wspace=0.57)
cbar_ax = fig.add_axes([0.835, 0.18, 0.02, 0.7])
cbar = fig.colorbar(pcm, cax=cbar_ax)
cbar.set_label("$\\rho-\\rho_1$")
fig.set_dpi(d)
plt.savefig('regime_graphic_figure_down.png', bbox_inches='tight')
plt.clf()
