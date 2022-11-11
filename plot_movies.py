import Constants as CON
import h5py
import os
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
import math
import pathlib
from scipy.ndimage.filters import gaussian_filter

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

id_conv = {'005': '05', '100': '10', '095': '095', '102': '12', '105': '15', '200': '20'}
num_conv = {'005': 0.5, '095': 0.95, '100': 1.0, '105': 1.5, '102': 1.2, '200': 2.0}

sim_length = {'a200c200': 450, 'a102c200': 450, 'a095c200': 450, 'a005c200': 450, 
              'a200c105': 450, 'a102c105': 450, 'a095c105': 450, 'a005c105': 450,
              'a200c100': 260, 'a102c100': 260, 'a095c100': 260, 'a005c100': 260,
              'a200c005': 220, 'a102c005': 220, 'a095c005': 220, 'a005c005': 220}
def plot_density(rho, a, c, T, N, folder):
    fig, ax = plt.subplots(figsize=(8,4))
    pcm = ax.imshow(np.transpose(rho)-sw.dens0(28,-2), vmin=0, vmax=2, cmap='viridis', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
    plt.fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(num_conv[a])/(H-z0), facecolor="white", zorder=10)
    plt.plot(np.linspace(0, L, Nx)/(H-z0), keel(num_conv[a])/(H-z0), linewidth=0.5, color='black')
    plt.gca().set_yticklabels(['4', '3', '2', '1', '0'])
    plt.xlim(20,115)
    plt.title('F{0}H{1} Density $\\rho$ ($t={2}t_0$)'.format(id_conv[c], id_conv[a], round(T/t_0, 2)))
    plt.ylim(0,-4)
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel('$x/z_0$')
    plt.ylabel('$z/z_0$')
    ax.set_aspect('auto')
    #cbar_ax = fig.add_axes([0.835, 0.18, 0.02, 0.7])
    cbar = fig.colorbar(pcm, orientation='horizontal', pad=0.2)
    #plt.tight_layout()
    cbar.set_label("$\\rho-\\rho_1$ (kg/m$^3$)")
    if N < 100:
        if N < 10:
            N_str = '00'+str(N)
        else:
            N_str = '0'+str(N)
    else:
        N_str = str(N)

    plt.savefig(pathlib.Path(folder+'img{0}.png'.format(N_str)).absolute(), dpi=200)
    plt.clf()
    plt.close()

def plot_density_contours_vorticity(rho, q, a, c, T, N, folder):
    fig, ax = plt.subplots(figsize=(8,4))
    rho1 = sw.dens0(28,-2)
    rho2 = sw.dens0(30,-2)
    # N^2 max
    N_sq = -9.81*gaussian_filter(np.gradient(rho, np.linspace(0, H, Nz), axis=1)/sw.dens0(28,-2), 4) 
    max_ind = np.argmax(N_sq, axis=1)
    z_Nvalues = []
    for j in range(len(N_sq)):
        z_Nvalues.append(-np.linspace(0, H, Nz)[(Nz-1)-max_ind[j]])
    z_rhovalues = []
    #rho_i = [rho1+0.005, rho2-0.005]
    for i in range(9):
        z_rhovalues.append([])
        rho_i = rho1+(rho2-rho1)*(i+1)/10
        min_ind = np.argmin(abs(rho-rho_i), axis=1)
        for k in range(Nx):
            z_rhovalues[i].append(-np.linspace(0, H, Nz)[(Nz-1)-min_ind[k]])
        plt.plot(np.linspace(0, L, Nx)/(H-z0), gaussian_filter(np.array(z_rhovalues[i])/(H-z0), 5), color='black', linewidth=0.6)
    plt.plot(np.linspace(0, L, Nx)/(H-z0), gaussian_filter(np.array(z_Nvalues)/(H-z0), 5), linewidth=1.5, color='black')
    pcm = ax.imshow(gaussian_filter(np.transpose(q), 4), vmin=-0.2, vmax=0.2, cmap='bwr', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
    plt.fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(num_conv[a])/(H-z0), facecolor="white", zorder=10)
    plt.plot(np.linspace(0, L, Nx)/(H-z0), keel(num_conv[a])/(H-z0), linewidth=0.5, color='black')
    plt.gca().set_yticklabels(['4', '3', '2', '1', '0'])
    plt.xlim(20,115)
    plt.title('F{0}H{1} Vorticity $q$ with Density $\\rho$ contours ($t={2}t_0$)'.format(id_conv[c], id_conv[a], round(T/t_0, 2)))
    plt.ylim(0,-4)
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel('$x/z_0$')
    plt.ylabel('$z/z_0$')
    ax.set_aspect('auto')
    #cbar_ax = fig.add_axes([0.835, 0.18, 0.02, 0.7])
    cbar = fig.colorbar(pcm, orientation='horizontal', pad=0.2)
    #plt.tight_layout()
    cbar.set_label("Vorticity $q$ (s$^{{-1}}$)")
    if N < 100:
        if N < 10:
            N_str = '00'+str(N)
        else:
            N_str = '0'+str(N)
    else:
        N_str = str(N)

    plt.savefig(pathlib.Path(folder+'img{0}.png'.format(N_str)).absolute(), dpi=200)
    plt.clf()
    plt.close()


def plot_uvelocity(u, a, c, T, N, folder):
    fig, ax = plt.subplots(figsize=(8,4))
    U = num_conv[c]*np.sqrt((H-z0)*DB)
    pcm = ax.imshow(np.transpose(u)/U, vmin=-1, vmax=1, cmap='bwr', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
    plt.fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(num_conv[a])/(H-z0), facecolor="white", zorder=10)
    plt.plot(np.linspace(0, L, Nx)/(H-z0), keel(num_conv[a])/(H-z0), linewidth=0.5, color='black')
    ax.set_yticklabels(['4', '3', '2', '1', '0'])
    plt.xlim(20,115)
    plt.title('F{0}H{1} Horizontal Velocity $u$ ($t={2}t_0$)'.format(id_conv[c], id_conv[a], round(T/t_0, 2)))
    plt.ylim(0,-4)
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel('$x/z_0$')
    plt.ylabel('$z/z_0$')
    ax.set_aspect('auto')
    cbar = fig.colorbar(pcm, orientation='horizontal', pad=0.2)
    cbar.set_label("$u/U$")
    if N < 100:
        if N < 10:
            N_str = '00'+str(N)
        else:
            N_str = '0'+str(N)
    else:
        N_str = str(N)
    plt.savefig(folder+'img{0}.png'.format(N_str), dpi=200)
    plt.clf()
    plt.close()

def plot_density_contours_velocity(rho, u, a, c, T, N, folder):
    fig, ax = plt.subplots(figsize=(8,4))
    rho1 = sw.dens0(28,-2)
    rho2 = sw.dens0(30,-2)
    U = num_conv[c]*np.sqrt((H-z0)*DB)
    # N^2 max
    N_sq = -9.81*gaussian_filter(np.gradient(rho, np.linspace(0, H, Nz), axis=1)/sw.dens0(28,-2), 4) 
    max_ind = np.argmax(N_sq, axis=1)
    z_Nvalues = []
    for j in range(len(N_sq)):
        z_Nvalues.append(-np.linspace(0, H, Nz)[(Nz-1)-max_ind[j]])
    z_rhovalues = []
    #rho_i = [rho1+0.005, rho2-0.005]
    for i in range(9):
        z_rhovalues.append([])
        rho_i = rho1+(rho2-rho1)*(i+1)/10
        min_ind = np.argmin(abs(rho-rho_i), axis=1)
        for k in range(Nx):
            z_rhovalues[i].append(-np.linspace(0, H, Nz)[(Nz-1)-min_ind[k]])
        plt.plot(np.linspace(0, L, Nx)/(H-z0), gaussian_filter(np.array(z_rhovalues[i])/(H-z0), 5), color='black', linewidth=0.6)
    plt.plot(np.linspace(0, L, Nx)/(H-z0), gaussian_filter(np.array(z_Nvalues)/(H-z0), 5), linewidth=1.5, color='black')
    pcm = ax.imshow(np.transpose(u)/U, vmin=-1, vmax=1, cmap='bwr', origin='lower', extent=(0, L/(H-z0), -H/(H-z0), 0))
    plt.fill_between(np.linspace(0, L, Nx)/(H-z0), 0, keel(num_conv[a])/(H-z0), facecolor="white", zorder=10)
    plt.plot(np.linspace(0, L, Nx)/(H-z0), keel(num_conv[a])/(H-z0), linewidth=0.5, color='black')
    plt.gca().set_yticklabels(['4', '3', '2', '1', '0'])
    plt.xlim(20,115)
    plt.title('F{0}H{1} Horizontal Velocity $u$ with Density $\\rho$ contours ($t={2}t_0$)'.format(id_conv[c], id_conv[a], round(T/t_0, 2)))
    plt.ylim(0,-4)
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel('$x/z_0$')
    plt.ylabel('$z/z_0$')
    ax.set_aspect('auto')
    #cbar_ax = fig.add_axes([0.835, 0.18, 0.02, 0.7])
    cbar = fig.colorbar(pcm, orientation='horizontal', pad=0.2)
    #plt.tight_layout()
    cbar.set_label("$u/U$")
    if N < 100:
        if N < 10:
            N_str = '00'+str(N)
        else:
            N_str = '0'+str(N)
    else:
        N_str = str(N)

    plt.savefig(pathlib.Path(folder+'img{0}.png'.format(N_str)).absolute(), dpi=200)
    plt.clf()
    plt.close()

def create_pngs():
    for dir_a in ['200']:
        for dir_c in ['200']:
            print(dir_a, dir_c)
            direct = '/gpfs/fs0/scratch/n/ngrisoua/samdeab/mixing/new/data-mixingsim-a{0}c{1}-00/'.format(dir_a, dir_c)
            for i in range(70, sim_length['a'+dir_a+'c'+dir_c], 1):
                with h5py.File(pathlib.Path(direct+'data-mixingsim-a{0}c{1}-00_s{2}.h5'.format(dir_a, dir_c, i)).absolute(), mode='r') as f:
                    t = f['tasks']['rho'].dims[0]['sim_time'][0]
                    folder = '/gpfs/fs0/scratch/n/ngrisoua/samdeab/mixing/fig_pngs/a{0}c{1}/density/'.format(dir_a, dir_c,)
                    plot_density_contours_velocity(f['tasks']['rho'][0], f['tasks']['u'][0], dir_a, dir_c, t, i, folder)
                    folder = '/gpfs/fs0/scratch/n/ngrisoua/samdeab/mixing/fig_pngs/a{0}c{1}/uvelocity/'.format(dir_a, dir_c,)
                    plot_density_contours_vorticity(f['tasks']['rho'][0], f['tasks']['vorticity'][0], dir_a, dir_c, t, i, folder)

def create_movies():
    for dir_a in ['005', '095', '102', '200']:
        for dir_c in ['005', '100', '105', '200']:
            arg = 'cd /gpfs/fs0/scratch/n/ngrisoua/samdeab/mixing/fig_pngs/a{0}c{1}/density/ ; ffmpeg -y -framerate 20 -start_number 70 -i img%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p F{1}H{0}_density.mp4'.format(dir_a, dir_c)
            os.system(arg)
            arg = 'cd /gpfs/fs0/scratch/n/ngrisoua/samdeab/mixing/fig_pngs/a{0}c{1}/uvelocity/ ; ffmpeg -y -framerate 20 -start_number 70 -i img%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p F{1}H{0}_uvelocity.mp4'.format(dir_a, dir_c)
            os.system(arg)

create_pngs()
create_movies()
