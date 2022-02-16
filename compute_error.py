"""
Usage:
	mp4_plots.py <files>... [--fileno=<str>] [--splice=<str>]

Written by: Sam De Abreu August 2021
"""
import Constants as CON
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import dedalus
import dedalus.public as de


from docopt import docopt
args = docopt(__doc__)
files = args['<files>'] #=h5 file location for analysis
fileno = args['--fileno'] #Name of outputted mp4s
splice = args['--splice'] #Specified files for analysis
splice0 = int(splice[:splice.find('-')])
splice1 = int(splice[splice.find('-')+1:])

Nx = CON.Nx
Nz = CON.Nz
H = CON.H
L = CON.L
l = CON.l
h = CON.h
z0 = CON.z0
h_z = CON.H-z0
sigma = 6


def sort_h5files(h5_files, splice0, splice1):
	#Sort h5 files
	temp_list = []
	temp_list2 = []
	for filename in h5_files:
		temp_list.append(int(filename[filename.find("_")+2:filename.find(".")]))
	for i in range(splice0, splice1+1):
		temp_list2.append(h5_files[temp_list.index(i)])
	h5_files = temp_list2
	return h5_files

def compute_depth(h5file, i):
    with h5py.File(h5file, mode='r') as f:
        u = f['tasks']['u'][i][203]
        z = f['tasks']['u'].dims[2][0][:]
        ind = np.argmax(-np.diff(u))
        return -z[Nz-ind]

def plot_depths():
    data = np.loadtxt('depth_{0}_{1}-{2}.txt'.format(fileno, splice0, splice1), unpack=True)
    time = data[0]
    Eb = data[1]
    plt.plot(time, Eb, label="Depth", marker='o')
    #plt.plot(time, Ea, label="Available potential energy")
    #plt.plot(time, Ep, label="Total potential energy")
    plt.xlabel("z (m)")
    plt.ylabel("u (m/s)")
    plt.title("Immobile Water Layer velocity at x = "+str(L*203/640)+"m")
    #plt.legend()
    plt.savefig('depth_{0}_{1}-{2}_1.png'.format(fileno, splice0, splice1))

def create_data_file():
    h5_files = sort_h5files(files, splice0, splice1)
    file = open('depth_{0}_{1}-{2}.txt'.format(fileno, splice0, splice1), 'a+')
    for h5file in h5_files:
        #with h5py.File(h5file, mode='r') as f: 
        #    times = f['tasks']['u'].dims[0]['sim_time'][:]
        #    for i in range(len(times)):
        #        file.write(str(times[i])+'\t'+str(compute_depth(h5file, i))+'\n')
        #    print(h5file)
        with h5py.File(h5_files[-1], mode='r') as f: 
            u = f['tasks']['u'][0][203]
            z = f['tasks']['u'].dims[2][0][:]
            ind = np.argmax(-np.diff(u))
            for i in range(0, ind):
                file.write(str(f['tasks']['u'].dims[2][0][:][i])+'\t'+str(f['tasks']['u'][0][203][i])+'\n')
    file.close()
create_data_file()
plot_depths()
