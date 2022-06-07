"""
Usage:
	mp4_plots.py <files>... [--output=<dir>] [--fileno=<str>] [--task=<str>] [--splice=<str>] [--time=<str>]

Written by: Rosalie Cormier, August 2021
"""

#Used for plotting simulated data

#Hasn't been run in parallel

#Assumes a folder './figures' will be used to save results

#Can add masking of ice keel if desired

##############################
import Constants as CON
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
import dedalus
import dedalus.public as de
from dedalus.extras import flow_tools, plot_tools
from dedalus.tools import post
from dedalus.core.operators import GeneralFunction

#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 14})

from docopt import docopt
args = docopt(__doc__)
files = args['<files>'] #=h5 file location for analysis
fileno = args['--fileno'] #Name of outputted mp4s
specific_task = args['--task'] #Specified tasks
splice = args['--splice'] #Specified files for analysis
avg_salt_time = float(args['--time']) #A specific time to compute average salt
splice0 = int(splice[:splice.find('-')])
splice1 = int(splice[splice.find('-')+1:])
import pathlib
output_path = pathlib.Path(args['--output']).absolute()

#Space-dependent variables that can be accessed directly - there are others not included here
tasks_legend = ['u', 'w', 'C', 'vorticity', 'dsfx', 'dsfz', 'f']
task_titles_legend = ['x-Velocity', 'z-Velocity', 'Salinity ($h/z_0={0}$, $U/\\sqrt{{z_0g^\\prime}}$=${1}$)'.format(CON.a, CON.c), 'Vorticity ($\hat{y}$)', 'Diffusive Salinity Flux, x-Component', 'Diffusive Salinity Flux, z-Component', 'Keel mask']
tasks = specific_task.split(",")
task_titles = []
for task in tasks:
	task_titles.append(task_titles_legend[tasks_legend.index(task)])
#Constants that can be accessed directly - there are others not included here
consts = ['energy', 'salt', "ked_rate"]
const_titles = ['Total Energy', 'Total Salt', "Energy dissipation"]
#Constants for the plotting (keep the same as nl_strat_simulation.py)
Nx = CON.Nx
Nz = CON.Nz
H = CON.H
L = CON.L
l = CON.l
h = CON.h
z0 = CON.z0
h_z = CON.H-z0
sigma = CON.sigma

xbasis = de.Chebyshev('x', Nx, interval=(0, L))
zbasis = de.Chebyshev('z', Nz, interval=(0, H))
domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)

x, z = domain.grids()
wall_mask = 0.5*(np.tanh((x-10)/0.025)+1)-0.5*(np.tanh((x-100)/0.025)+1)

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
def read_h5files(h5_files):
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
	return dsets
def animate_data(dsets):
	#Find length of time series
	t_len = len(dsets[0])
	x = np.linspace(0, L, Nx)
	z = np.linspace(-H, 0, Nz)
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
			vmin, vmax = 27.5, 30.5
			cmap = 'viridis'
			label = 'psu'
			color = 'k'
			ticks = [28, 29, 30]
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
		elif task_name == 'vorticity':
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

		#keel = -(h) * np.exp(-((x-l/2)**2)/(2*6**2))
		keel = -h*sigma**2/(sigma**2+4*(x-l)**2)

		fig_j, ax_j = plt.subplots(figsize=(8,8))
		im_j = ax_j.imshow(dsets[j][0][1].transpose(), vmin=vmin, vmax=vmax, cmap=cmap, extent=(0, L/(H-z0), -H/(H-z0), 0), origin='lower', animated=True)
		#plt.axvline(x=l/2)
		#plt.axvline(x=l/2+sigma)
		plt.fill_between(x/(H-z0), 0, keel/(H-z0), facecolor="white")
		plt.plot(x/(H-z0), keel/(H-z0), linewidth=0.5, color=color)
		#plt.xlim(2, plt.xlim()[1]-2)
		fig_j.colorbar(im_j, label=label, orientation='horizontal', ticks=ticks)
		ax_j.set_title(task_title+' Time: {0}s-{1}s'.format(int(splice0*CON.save_freq*CON.save_max*CON.dt), int(splice1*CON.save_freq*CON.save_max*CON.dt)))
		ax_j.set_xlabel('$x/z_0$')
		ax_j.set_ylabel('$z/z_0$')
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
def compute_froude_number(h5_file):
	with h5py.File(h5_file, mode='r') as f:
		h_data = compute_mixedlayerdepth(h5_file)
		g = 9.8*((1022-1020)/1020)
		i_1 = math.floor(Nx*(l/2)/L)
		i_2 = math.floor(Nx*(l/2+sigma)/L)
		i_3 = math.floor(Nx*(l/2+2*sigma)/L)
		Fr_1 = 0
		Fr_2 = 0
		Fr_3 = 0
		for mx in range(Nx):
			#i_h = math.floor(-Nz*h_data[1][mx]/L)
			i_h = h_data[0][mx]
			h_d = h_data[1][mx]+h * np.exp(-((np.linspace(0,L,Nx)[mx]-l/2)**2)/(2*6**2))
			if mx <= i_1:
				Fr_1 += (f['tasks']['u'][0][mx][i_h]**2)**0.5/(g*abs(h_d))**0.5
			elif i_1 < mx <= i_2:
				Fr_2 += (f['tasks']['u'][0][mx][i_h]**2)**0.5/(g*abs(h_d))**0.5
			elif i_2 < mx <= i_3:
				Fr_3 += (f['tasks']['u'][0][mx][i_h]**2)**0.5/(g*abs(h_d))**0.5
		return (Fr_1/i_1, Fr_2/(i_2-i_1), Fr_3/(i_3-i_2))
def compute_mixedlayerdepth(h5_file):
	#Creates file for ML
	file = open('MLD_{0}.txt'.format(fileno), 'a+')
	with h5py.File(h5_file, mode='r') as f:
		data = f['tasks']['rho'][0]
		indz = []
		posz = []
		posx = []
		x = np.linspace(0, L, Nx)
		for mx in range(Nx):
			posx.append(mx*L/Nx)
			rho_min = data[mx][-1]
			diff = data[mx] - rho_min
			i = min(np.argwhere(diff <= 0.1))[0]
			posz.append(-H*(1-i/Nz))
			indz.append(i)
			file.write(str(-H*(1-i/Nz))+"\n")
	file.close()
	return (indz, posz, posx)
def plot_avg_salt(avg_salt_time, h5_files):
	file_number = round(avg_salt_time/(3e-2*15))
	with h5py.File(h5_files[file_number-1], mode='r') as f:
		plt.clf()
		z = np.linspace(-H, 0, Nz)
		plt.plot(-1 * np.tanh((z+(h_z)) / 1e-1) + 29, z, label="Initial")
		plt.plot(f['tasks']['avg_salt_prof1'][0][0], z, label="Upstream")
		plt.plot(f['tasks']['avg_salt_prof2'][0][0], z, label="Downstream")
		plt.xlabel("Average Salinity (psu)")
		plt.ylabel("z (m)")
		plt.legend()
		plt.title("Average salinity vs depth ($h={0}$m, $U={1}$m/s)".format(h, CON.U))
		plt.savefig('AverageSalinity_{0}.png'.format(str(file_number)))
def plot_mixedlayerdepth(h5_file):
	with h5py.File(h5_file):
		data = compute_mixedlayerdepth(h5_file)
		plt.clf()
		plt.plot(data[2], data[1], label="Mixed Layer Depth")
		plt.plot(data[2], (z0-H)*np.ones(len(data[0])), label="Initial", marker="_")
		plt.plot(data[2], -h*sigma**2/(sigma**2+4*(np.array(data[2])-l)**2), linestyle='dashed')
		plt.ylim(-H+15,0)
		plt.xlim(10,L-10)
		plt.legend()
		plt.xlabel("x (m)")
		plt.ylabel("Mixed layer depth (m)")
		plt.title("Mixed layer depth ($h={0}$m, $U={1}$m/s)".format(h, CON.U))
		plt.savefig('MixedLayerDepth_{0}_.png'.format(int(h5_file[h5_file.find("_")+2:h5_file.find(".")])))
def animate_velocity_field():
	#Plot and animate velocity field

	u_series = dsets[0]
	w_series = dsets[1]

	v = np.zeros((t_len, Nz, Nx))
	for t in range(t_len):
		u_t = np.array(dsets[0][t][1].transpose())
		w_t = np.array(dsets[1][t][1].transpose())
		v_t = np.zeros((Nx, Nz))
		v_t = (u_t**2 + w_t**2)**0.5
		v[t] = v_t

	fig_v, ax_v = plt.subplots()
	im_v = ax_v.imshow(v[0], vmin=0, vmax=0.4, cmap='Purples', extent=(0, L, -15, 0), origin='lower', animated=True)
	plt.streamplot(x, z, u_series[0][1].transpose(), w_series[0][1].transpose(), color='r', density=0.7, linewidth=0.5)
	plt.plot(x, keel, linewidth=0.5, color='k')
	plt.xlim(10,L-10)
	fig_v.colorbar(im_v, label='m/s', orientation='horizontal', ticks=[0, 0.1, 0.2, 0.3, 0.4])
	ax_v.set_title('Velocity')
	ax_v.set_xlabel('x (m)')
	ax_v.set_ylabel('z (m)')
	plt.tight_layout()

	def init():
		im_v.set_array(v[0])
		u = u_series[0][1].transpose()
		w = w_series[0][1].transpose()
		stream = ax_v.streamplot(x, z, u, w, color='r', density=0.7, linewidth=0.5)
		return [im_v, stream]

	def animate(i):
		ax_v.collections = []
		ax_v.patches = []
		im_v.set_array(v[i])
		u = u_series[i][1].transpose()
		w = w_series[i][1].transpose()
		stream = ax_v.streamplot(x, z, u, w, color='r', density=0.7, linewidth=0.5)
		return [im_v, stream]

	ani = animation.FuncAnimation(fig_v, animate, init_func=init, frames=t_len, interval=30, blit=False)
	ani.save('figures/anim_v_{0}.mp4'.format(fileno))
def animate_dsfx_field():
	#Plot and animate diffusive salinity flux field
	dsfx_series = dsets[4]
	dsfz_series = dsets[5]

	dsf = np.zeros((t_len, Nz, Nx))
	for t in range(t_len):
		dsfx_t = np.array(dsets[4][t][1].transpose())
		dsfz_t = np.array(dsets[5][t][1].transpose())
		dsf_t = np.zeros((Nx, Nz))
		dsf_t = (dsfx_t**2 + dsfz_t**2)**0.5
		dsf[t] = dsf_t

	fig_d, ax_d = plt.subplots()
	im_d = ax_d.imshow(dsf[0], vmin=0, vmax=0.5, cmap='Greens', extent=(0, L, -15, 0), origin='lower', animated=True)
	plt.streamplot(x, z, dsfx_series[0][1].transpose(), dsfz_series[0][1].transpose(), color='c', density=0.7, linewidth=0.5)
		#Probably better suited to a quiver plot than a streamplot
	plt.plot(x, keel, linewidth=0.5, color='k')
	fig_d.colorbar(im_d, label='psu * m/s', orientation='horizontal', ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
	ax_d.set_title('Diffusive Salinity Flux')
	ax_d.set_xlabel('x (m)')
	ax_d.set_ylabel('z (m)')
	plt.tight_layout()

	def init():
		im_d.set_array(dsf[0])
		dsfx = dsfx_series[0][1].transpose()
		dsfz = dsfz_series[0][1].transpose()
		stream = ax_d.streamplot(x, z, dsfx, dsfz, color='c', density=0.7, linewidth=0.5)
		return [im_d, stream]

	def animate(i):
		ax_d.collections = []
		ax_d.patches = []
		im_d.set_array(v[i])
		dsfx = dsfx_series[i][1].transpose()
		dsfz = dsfz_series[i][1].transpose()
		stream = ax_d.streamplot(x, z, dsfx, dsfz, color='c', density=0.7, linewidth=0.5)
		return [im_d, stream]

	ani = animation.FuncAnimation(fig_d, animate, init_func=init, frames=t_len, interval=30, blit=False)
	ani.save('figures/anim_dsf_{0}.mp4'.format(fileno))
def compute_time_series(consts, h5_files):
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
	rho_ref, z_ref = sort_rho_z(h5_file)
	with h5py.File(h5_file, mode='r') as f:
		diff = domain.new_field()
		diff['g'] = rho_ref
		result = diff.differentiate('z')['g']
		#print(f['tasks']['rho'][0][20])
		#print(z_ref[20])
		#print(diff.differentiate('z')['g'][400])
		
		integrand = domain.new_field()
		integrand['g'] = 9.8*f['tasks']['nabla_rho_sq'][0]/diff['g']*wall_mask
		
		E_b_t = de.operators.integrate(integrand, 'x', 'z')
		
		return E_b_t.evaluate()['g'][0][0]
h5_files = sort_h5files(files, splice0, splice1)
dset = read_h5files(h5_files)
#plot_avg_salt(avg_salt_time, h5_files)
#plot_mixedlayerdepth(h5_files[-1])
#print(compute_froude_number(h5_files[-1]))
animate_data(dset)
#compute_time_series(consts, h5_files)