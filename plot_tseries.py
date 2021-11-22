"""
Written by: Rosalie Cormier, August 2021
"""

import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp

#Both of these values must match those used in the simulation
dt = 5e-4
freq = 30

tasks = ['energy', 'salt', 'ked_rate'] #Can add more
task_titles = ['Total Energy', 'Total Salt', "Energy Dissipation"]
ylabels = ['Energy (J)', 'Salt (g)']

numbers = ['Test']
	#Modify as needed

for j in range(len(tasks)):

	task = tasks[j]
	task_title = task_titles[j]
	ylabel = ylabels[j]

	task_data = []

	for number in numbers:

		file = '{0}_tseries_{1}.txt'.format(task, number)

		y_data_i = np.loadtxt(file)
		task_data = np.concatenate([task_data, y_data_i])

	iterations = len(task_data)
	total_time = dt*freq*iterations

	t_data = np.linspace(0, total_time, iterations)

	plt.figure()
	plt.scatter(t_data, task_data, s=5, c='r')
	plt.xlabel('Time (s)')
	plt.ylabel(ylabel)
	plt.title(task_title)
	plt.savefig('{0}_fig'.format(task))
	plt.close()
