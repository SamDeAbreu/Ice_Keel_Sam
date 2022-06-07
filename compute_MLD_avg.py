"""
Usage:
	mp4_plots.py <files>... [--fileno=<str>] [--splice=<str>]

Written by: Sam De Abreu August 2021
"""
import Constants as CON
import h5py
import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
plt.ioff()
import dedalus
import dedalus.public as de
import scipy.signal as sc

from docopt import docopt
args = docopt(__doc__)
files = args['<files>'] #=h5 file location for analysis
fileno = args['--fileno'] #Name of outputted mp4s
splice = args['--splice'] #Specified files for analysis
splice0 = int(splice[:splice.find('-')])
splice1 = int(splice[splice.find('-')+1:])


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

def compute_mixedlayerdepth(h5_file):
	with h5py.File(h5_file, mode='r') as f:
		data = f['tasks']['rho'][0]
		posz = []
		low = math.ceil((CON.l+CON.sigma)*CON.Nx/CON.L)
		for mx in range(low, CON.Nx):
			rho_min = data[mx][-1]
			diff = data[mx] - rho_min
			i = min(np.argwhere(diff <= 0.1))[0]
			posz.append(-CON.H*(1-i/CON.Nz))
	return np.mean(posz), np.stdev(posz)

h5_files = sort_h5files(files, splice0, splice1)
file = open('MLD_avg_{0}.txt'.format(fileno), 'a+')
for h5file in h5_files[::2]:
    with h5py.File(h5file, mode='r') as f: 
        times = f['tasks']['u'].dims[0]['sim_time'][:]
        print(h5file)
        file.write(str(times[0])+'\t'+str(compute_mixedlayerdepth(h5file))+ '\n')
file.close()