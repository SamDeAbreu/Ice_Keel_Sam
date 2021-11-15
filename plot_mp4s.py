"""
Usage:
	mp4_plots.py <files>... [--output=<dir>] [--fileno=<str>] [--task=<str>] [--splice=<str>]

Written by: Rosalie Cormier, August 2021
"""

#Used for plotting simulated data

#Hasn't been run in parallel

#Assumes a folder './figures' will be used to save results

#Can add masking of ice keel if desired

##############################
import sys
import h5py
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib import rc
import gc
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 14})

from docopt import docopt
args = docopt(__doc__)
h5_files = args['<files>']
fileno = args['--fileno']
specific_task = args['--task']
splice = args['--splice']
splice0 = int(splice[:splice.find('-')])
splice1 = int(splice[splice.find('-')+1:])
import pathlib
output_path = pathlib.Path(args['--output']).absolute()

#Space-dependent variables that can be accessed directly - there are others not included here
tasks_legend = ['u', 'w', 'C', 'vorticity', 'dsfx', 'dsfz']
task_titles_legend = ['x-Velocity', 'z-Velocity', 'Salinity', 'Vorticity ($\hat{y}$)', 'Diffusive Salinity Flux, x-Component', 'Diffusive Salinity Flux, z-Component']
tasks = specific_task.split(",")
task_titles = []
for task in tasks:
	task_titles.append(task_titles_legend[tasks_legend.index(task)])
#Constants that can be accessed directly - there are others not included here
consts = ['energy', 'salt']
const_titles = ['Total Energy', 'Total Salt']

Nx = 640
Nz = 320
H = 15
L = 110
l = 70
h = 7
#Sort h5 files
temp_list = []
temp_list2 = []
for filename in h5_files:
	temp_list.append(int(filename[filename.find("_")+2:filename.find(".")]))
for i in range(len(h5_files)):
	temp_list2.append(h5_files[temp_list.index(i+1)])
h5_files = temp_list2[splice0:splice1]
#Dsets have size of tasks x timesteps x 2 x nx x nz
dsets = []
for task in tasks:

	task_tseries = []

	for filename in h5_files:

		with h5py.File(filename, mode='r') as f:

			dset = f['tasks'][task]
			if len(dset.shape) != 3:
				raise ValueError("This only works for 3D datasets")
			task_grid = np.array(dset[()], dtype=np.float64) #The [()] notation returns all data from an h5 object
			#x_scale = f['scales']['x']['1.0']
			#x_axis = np.array(x_scale[()], dtype=np.float64)
			#z_scale = f['scales']['z']['1.0']
			#z_axis = np.array(z_scale[()], dtype=np.float64)
			t_scale = f['scales']['sim_time']
			t_axis = np.array(t_scale[()], dtype=np.float64)
			for i in range(len(t_axis)):
				time_slice = [t_axis[i], task_grid[i]]
				task_tseries.append(time_slice)
			del dset
			del task_grid
			del t_scale
			del t_axis
			del f
			del time_slice
			gc.collect()
		print("Filename: "+filename+"  Task: "+task)
	dsets.append(task_tseries)

#Find length of time series
t_len = len(dsets[0])

x = np.linspace(0, L, Nx)
z = np.linspace(-15, 0, Nz)
file = open('AverageSalt.txt', 'a+')
#Vertical modes: Graphically, going from the bottom is the lowest mode
#Integrate from 0-h with special method. Then h-H with plain simple EZ kekW method
#z = H-H/Nz*i = H(1-i/Nz)
#H>z>h => EZ method    h>z>0 => Hard method
for filename in [h5_files[0]]:
	#['C'][time=0][x][z]
	with h5py.File(filename, mode='r') as f:
		for mz in range(175, Nz):
			z_cord = -H*(1-mz/Nz)
			if z_cord < -h:
				x_length = 35
				i = math.floor(Nx*x_length/L)
				total_salt = 0
				for mx in range(i):
					total_salt += f['tasks']['C'][0][mx][mz]
				average_salt = total_salt/i
				print(average_salt, z_cord)
			else:
				x_length = l/2-(-72*np.log((-z_cord)/h))**0.5
				i = math.floor(Nx*x_length/L)
				total_salt = 0
				for mx in range(i):
					total_salt += f['tasks']['C'][0][mx][mz]
				average_salt = total_salt/i
				file.write(str(average_salt) + '\n')
				print(x_length, z_cord, average_salt)	
file.close()
exit()
#Plot and animate all the tasks
for j in range(len(tasks)):

	task_name = tasks[j]
	task_title = task_titles[j]

	#Set bounds (adjust as needed) and colormaps
	if task_name == 'u':
		vmin, vmax = -0.5, 0.5
		cmap = 'seismic'
		label = 'm/s'
		color = 'k'
		ticks = [-0.5, -0.25, 0, 0.25, 0.5]
	elif task_name == 'w':
		vmin, vmax = -0.5, 0.5
		cmap = 'seismic'
		label = 'm/s'
		color = 'k'
		ticks = [-0.5, -0.25, 0, 0.25, 0.5]
	elif task_name == 'C':
		vmin, vmax = 25.5, 28.5
		cmap = 'viridis'
		label = 'psu'
		color = 'k'
		ticks = [25.5, 26.5, 27.5, 28.5]
	elif task_name == 'p':
		vmin, vmax = 0, 300
		cmap = 'viridis'
		label = 'Pa'
		color = 'k'
		ticks = [0, 100, 200, 300]
	elif task_name == 'f':
		vmin, vmax = 0, 1
		cmap = 'gray'
		label = ''
		color = 'r'
		ticks = [0, 1]
	elif task_name == 'ct':
		vmin, vmax = -2, 2
		cmap = 'viridis'
		label = 'psu/s'
		color = 'k'
		ticks = [-2, -1, 0, 1, 2]
	elif task_name == 'q':
		vmin, vmax = -2, 2
		cmap = 'PuOr'
		label = 's$^{-1}$'
		color = 'k'
		ticks = [-2, -1, 0, 1, 2]
	elif task_name == 'B':
		vmin, vmax = -300, 0
		cmap = 'viridis'
		label = '?'
		color = 'k'
		ticks = [-300, -200, -100, 0]
	elif task_name == 'dsfx' or 'dsfz':
		vmin, vmax = -0.001, 0.001
		cmap = 'BrBG'
		label = 'psu * m/s'
		color = 'k'
		ticks = [-0.001, 0, 0.001]

	keel = -h * np.exp(-((x-35)**2)/(2*6**2))

	fig_j, ax_j = plt.subplots()
	im_j = ax_j.imshow(dsets[j][0][1].transpose(), vmin=vmin, vmax=vmax, cmap=cmap, extent=(0, L, -15, 0), origin='lower', animated=True)
	plt.plot(x, keel, linewidth=0.5, color=color)
	plt.xlim(10,L-10)
	fig_j.colorbar(im_j, label=label, orientation='horizontal', ticks=ticks)
	ax_j.set_title(task_title)
	ax_j.set_xlabel('x (m)')
	ax_j.set_ylabel('z (m)')
	plt.tight_layout()

	def init():
		im_j.set_array(dsets[j][0][1].transpose())
		return [im_j]

	def animate(i):
		im_j.set_array(dsets[j][i][1].transpose())
		return [im_j]
	ani = animation.FuncAnimation(fig_j, animate, init_func=init, frames=t_len, interval=30, blit=True)
	name = r'{0}/anim_{1}_{2}_{3}-{4}.mp4'.format(str(output_path), task_name, fileno, str(splice0), str(splice1))
	ani.save(name)
#############################

# #Plot and animate velocity field

# u_series = dsets[0]
# w_series = dsets[1]

# v = np.zeros((t_len, Nz, Nx))
# for t in range(t_len):
# 	u_t = np.array(dsets[0][t][1].transpose())
# 	w_t = np.array(dsets[1][t][1].transpose())
# 	v_t = np.zeros((Nx, Nz))
# 	v_t = (u_t**2 + w_t**2)**0.5
# 	v[t] = v_t

# fig_v, ax_v = plt.subplots()
# im_v = ax_v.imshow(v[0], vmin=0, vmax=0.4, cmap='Purples', extent=(0, L, -15, 0), origin='lower', animated=True)
# plt.streamplot(x, z, u_series[0][1].transpose(), w_series[0][1].transpose(), color='r', density=0.7, linewidth=0.5)
# plt.plot(x, keel, linewidth=0.5, color='k')
# plt.xlim(10,L-10)
# fig_v.colorbar(im_v, label='m/s', orientation='horizontal', ticks=[0, 0.1, 0.2, 0.3, 0.4])
# ax_v.set_title('Velocity')
# ax_v.set_xlabel('x (m)')
# ax_v.set_ylabel('z (m)')
# plt.tight_layout()

# def init():
# 	im_v.set_array(v[0])
# 	u = u_series[0][1].transpose()
# 	w = w_series[0][1].transpose()
# 	stream = ax_v.streamplot(x, z, u, w, color='r', density=0.7, linewidth=0.5)
# 	return [im_v, stream]

# def animate(i):
# 	ax_v.collections = []
# 	ax_v.patches = []
# 	im_v.set_array(v[i])
# 	u = u_series[i][1].transpose()
# 	w = w_series[i][1].transpose()
# 	stream = ax_v.streamplot(x, z, u, w, color='r', density=0.7, linewidth=0.5)
# 	return [im_v, stream]

# ani = animation.FuncAnimation(fig_v, animate, init_func=init, frames=t_len, interval=30, blit=False)
# ani.save('figures/anim_v_{0}.mp4'.format(fileno))

#############################

# #Plot and animate diffusive salinity flux field

# dsfx_series = dsets[4]
# dsfz_series = dsets[5]

# dsf = np.zeros((t_len, Nz, Nx))
# for t in range(t_len):
# 	dsfx_t = np.array(dsets[4][t][1].transpose())
# 	dsfz_t = np.array(dsets[5][t][1].transpose())
# 	dsf_t = np.zeros((Nx, Nz))
# 	dsf_t = (dsfx_t**2 + dsfz_t**2)**0.5
# 	dsf[t] = dsf_t

# fig_d, ax_d = plt.subplots()
# im_d = ax_d.imshow(dsf[0], vmin=0, vmax=0.5, cmap='Greens', extent=(0, L, -15, 0), origin='lower', animated=True)
# plt.streamplot(x, z, dsfx_series[0][1].transpose(), dsfz_series[0][1].transpose(), color='c', density=0.7, linewidth=0.5)
# 	#Probably better suited to a quiver plot than a streamplot
# plt.plot(x, keel, linewidth=0.5, color='k')
# fig_d.colorbar(im_d, label='psu * m/s', orientation='horizontal', ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
# ax_d.set_title('Diffusive Salinity Flux')
# ax_d.set_xlabel('x (m)')
# ax_d.set_ylabel('z (m)')
# plt.tight_layout()

# def init():
# 	im_d.set_array(dsf[0])
# 	dsfx = dsfx_series[0][1].transpose()
# 	dsfz = dsfz_series[0][1].transpose()
# 	stream = ax_d.streamplot(x, z, dsfx, dsfz, color='c', density=0.7, linewidth=0.5)
# 	return [im_d, stream]

# def animate(i):
# 	ax_d.collections = []
# 	ax_d.patches = []
# 	im_d.set_array(v[i])
# 	dsfx = dsfx_series[i][1].transpose()
# 	dsfz = dsfz_series[i][1].transpose()
# 	stream = ax_d.streamplot(x, z, dsfx, dsfz, color='c', density=0.7, linewidth=0.5)
# 	return [im_d, stream]

# ani = animation.FuncAnimation(fig_d, animate, init_func=init, frames=t_len, interval=30, blit=False)
# ani.save('figures/anim_dsf_{0}.mp4'.format(fileno))

##############################

#Collect time series data for constants

for task in consts:

	file = open('{0}_tseries_{1}.txt'.format(task, fileno), 'a+')

	for filename in h5_files:

		with h5py.File(filename, mode='r') as f:
			try:
				for i in range(15):
					value = f['tasks'][task][i][0][0]
					file.write(str(value) + '\n')
			except(IndexError):
				pass
	file.close()

#A separate file is used for plotting these time series
