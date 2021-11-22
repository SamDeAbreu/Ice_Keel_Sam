"""
Written by: Sam De Abreu, November 2021
"""

import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp

#Both of these values must match those used in the simulation
dt = 5e-4
freq = 30

#file = 'AverageSalt.txt'.format(task, number)
salt_data_i = np.loadtxt('averageSalt_10.0_Test.txt', unpack=True)
plt.plot(salt_data_i[0], salt_data_i[1], label="Upstream")
plt.xlabel("z (m)")
plt.ylabel("Average Salt (Salinity)")
plt.title("Average Salt vs z at t=0 (Profile 1)")
plt.plot(salt_data_i[0], salt_data_i[2], label="Downstream")
plt.xlabel("z (m)")
plt.ylabel("Average Salt (Salinity)")
plt.legend()
plt.title("Average Salt vs z at t="+str(salt_data_i[3][0])+"s")
plt.show()