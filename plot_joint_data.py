import Constants as CON
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
import dedalus
import dedalus.public as de
import scipy.signal as sc

#scales
rho = 1022-1020
d = 6
u = 0.3
E = rho*u**2*d**3
E_s = rho*u**3*d**2
colors = ['#FF1300', '#9F0D0D', '#0CCAFF', '#097CD8', '#29E91F', '#0F9208']
labels = ['v04h15', 'v02h15', 'v04h10.5', 'v02h10.5', 'v04h7.5', 'v02h7.5']
z = np.linspace(-CON.H, 0, CON.Nz)

#KE dissipation

#colors = ['9F0D0D', '28508D', '0F9208']
i = 0
for f in ['v04z34h15', 'v02z34h15', 'v04z34h105', 'v02z34h105', 'v04z34h705', 'v02z34h705']:
    data = np.loadtxt('potentialenergy_'+f+'_1-1334.txt', unpack=True)
    plt.plot(data[0], data[7]/E_s, color=colors[i], label=labels[i])
    i += 1
plt.grid()
plt.xlabel('Simulation Seconds (s)')
plt.ylabel("$\\varepsilon_{{k}}/(\\Delta\\rho U^3\\sigma^2)$")
plt.title('Rate of Dissipation of Kinetic Energy')
plt.savefig('KE_dissipation.png')

plt.clf()
i = 0
#Background energy + vertical bouyancy flux
fig, (ax1, ax2) = plt.subplots(2,1, sharex='col')
for f in ['v04z34h15', 'v02z34h15', 'v04z34h105', 'v02z34h105', 'v04z34h705', 'v02z34h705']:
    data = np.loadtxt('potentialenergy_'+f+'_1-1334.txt', unpack=True)
    if 'v04' in f:
        ax1.plot(data[0], data[7]/E_s, color=colors[i], label=labels[i])
        ax2.plot(data[0], data[5]/E_s, color=colors[i], label=labels[i])
    i += 1
    ax2.plot(data[0], np.ones(len(data[0]))*0, linestyle='dashed', color='black', zorder=0, linewidth=1.6)
ax1.set_ylabel("$\\varepsilon_{{k}}/(\\Delta\\rho U^3\\sigma^2)$")
ax1.set_title('Rate of Kinetic Energy Dissipation and Downstream Vertical Buoyancy Flux')
ax1.grid()
ax2.set_ylabel("$\\phi_z/(\\Delta\\rho U^3\\sigma^2)$")
ax2.set_xlabel("Simulation Seconds (s)")
ax2.grid()
plt.tight_layout()
plt.savefig('EkEnergy_Vflux.png')

plt.clf()
i = 0
#Rate of mixing
for f in ['v04z34h15', 'v02z34h15', 'v04z34h105', 'v02z34h105', 'v04z34h705', 'v02z34h705']:
    data = np.loadtxt('potentialenergy_'+f+'_1-1334.txt', unpack=True)
    plt.plot(data[0], data[4]/E_s, color=colors[i], label=labels[i])
    i += 1
plt.grid()
plt.xlabel('Simulation Seconds (s)')
plt.ylabel("$\\Phi_{{d}}/(\\Delta\\rho U^3\\sigma^2)$")
plt.title('Rate of Diffusive Mixing in the Vertical Direction Time Series')
plt.savefig('Mixing_rates.png')

plt.clf()
i = 0
#MLD
x = np.linspace(0, CON.L, CON.Nx)
for f in ['v04z34h15', 'v02z34h15', 'v04z34h105', 'v02z34h105', 'v04z34h705', 'v02z34h705']:
    data = np.loadtxt('potentialenergy_'+f+'_1-1334.txt', unpack=True)
    MLD = np.loadtxt('MLD_{0}.txt'.format(f), unpack=True)
    plt.plot(x, MLD[:640], color=colors[i], label=labels[i])
    i += 1
plt.fill_between(x, 0, -(15) * np.exp(-((x-70/2)**2)/(2*6**2)), facecolor='w')
plt.fill_between(x, 0, -(15) * np.exp(-((x-70/2)**2)/(2*6**2)), facecolor='#FF1300', alpha=0.3)
plt.fill_between(x, 0, -(10.5) * np.exp(-((x-70/2)**2)/(2*6**2)), facecolor='w')
plt.fill_between(x, 0, -(10.5) * np.exp(-((x-70/2)**2)/(2*6**2)), facecolor='#0CCAFF', alpha=0.3)
plt.fill_between(x, 0, -(7.5) * np.exp(-((x-70/2)**2)/(2*6**2)), facecolor='w')
plt.fill_between(x, 0, -(7.5) * np.exp(-((x-70/2)**2)/(2*6**2)), facecolor='#0F9208', alpha=0.3)
plt.plot(x, -10*np.ones(len(x)), linestyle='dashed', color='black', label='Initial MLD')
plt.legend(loc='lower right')
plt.grid()
plt.xlim(10, 100)
plt.ylim(-20, 0)
plt.xlabel('x (m)')
plt.ylabel("Mixed Layer Depth (m)")
plt.title('Mixed Layer depth of Every Simulation ($t=2000$s)')
plt.savefig('MLDs.png')

plt.clf()
""" #MLD time downstream evolution
i = 0
for f in ['v04z34h15', 'v02z34h15', 'v04z34h105', 'v02z34h105', 'v04z34h705', 'v02z34h705']:
    data = np.loadtxt('MLD_avg_'+f+'.txt', unpack=True)
    plt.plot(data[0], data[1], color=colors[i], label=labels[i])
    i += 1
plt.grid()
plt.xlabel('Simulation Seconds (s)')
plt.ylabel("Average Downstream Mixed Layer Depth (m)")
plt.title('Average MLD Downstream Time Series')
plt.savefig('MLD_downstream_avgs.png')

plt.clf()
#MLD time downstream evolution
i = 0
for f in ['v04z34h15', 'v02z34h15', 'v04z34h105', 'v02z34h105', 'v04z34h705', 'v02z34h705']:
    data = np.loadtxt('MLD_upstream_avg_'+f+'.txt', unpack=True)
    plt.plot(data[0], data[1], color=colors[i], label=labels[i])
    i += 1
plt.grid()
plt.xlabel('Simulation Seconds (s)')
plt.ylabel("Average Upstream Mixed Layer Depth (m)")
plt.title('Average MLD Upstream Time Series')
plt.savefig('MLD_upstream_avgs.png') """

fig, (ax1, ax2) = plt.subplots(2,1, sharex='col')
i = 0
for f in ['v04z34h15', 'v02z34h15', 'v04z34h105', 'v02z34h105', 'v04z34h705', 'v02z34h705']:
    data = np.loadtxt('MLD_upstream_avg_'+f+'.txt', unpack=True)
    ax1.plot(data[0], data[1], color=colors[i], label=labels[i])
    data = np.loadtxt('MLD_avg_'+f+'.txt', unpack=True)
    ax2.plot(data[0], data[1], color=colors[i], label=labels[i])
    i += 1
ax1.grid()
ax1.set_ylabel('Upstream MLD (m)')
ax2.set_xlabel('Simulation Seconds (s)')
ax2.set_ylabel('Downstream MLD (m)')
ax2.grid()
ax1.set_title('Average Mixed Layer Depth Upstream & Downstream Time Series')
plt.tight_layout()
plt.savefig('MLD_Time_avgs.png')


plt.clf()
i = 0
li = ['2223', '1334', '1334', '1334', '1331', '1334']
#Average salinity downstream
for fo in ['v04z34h15', 'v02z34h15', 'v04z34h105', 'v02z34h105', 'v04z34h705', 'v02z34h705']:
    with h5py.File('data-mixingsim-{0}-00/data-mixingsim-{0}-00_s{1}.h5'.format(fo, li[i]), mode='r') as f:
        plt.plot(f['tasks']['avg_salt_prof2'][0][0], z, label=labels[i], color=colors[i])  
    i += 1
plt.plot(-1 * np.tanh((z+(CON.H-CON.z0)) / 1e-1) + 29, z, label="Initial profile", linestyle='dashed', color='black')
plt.grid()
plt.xlabel('Salinity (psu)')
plt.ylabel("Depth (m)")
plt.legend(loc='lower left')
plt.title('Average Downstream Salinity with respect to Depth')
plt.savefig('avg_downstream_salinity.png')

plt.clf()
i = 0
li = ['2223', '1334', '1334', '1334', '1331', '1334']
#Average salinity upstream
for fo in ['v04z34h15', 'v02z34h15', 'v04z34h105', 'v02z34h105', 'v04z34h705', 'v02z34h705']:
    with h5py.File('data-mixingsim-{0}-00/data-mixingsim-{0}-00_s{1}.h5'.format(fo, li[i]), mode='r') as f:
        plt.plot(f['tasks']['avg_salt_prof1'][0][0], z, label=fo, color=colors[i])  
    i += 1
plt.plot(-1 * np.tanh((z+(CON.H-CON.z0)) / 1e-1) + 29, z, label="Initial", linestyle='dashed', color='black')
plt.grid()
plt.xlabel('Salinity (psu)')
plt.ylabel("Depth (m)")
plt.title('Average Upstream Salinity with respect to Depth')
plt.savefig('avg_upstream_salinity.png')

