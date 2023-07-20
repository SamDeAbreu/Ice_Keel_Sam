"""
Performs all calculations for all region (Fr, eta) predictions and generates Figure 9.
Author: Sam De Abreu (2023)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from uncertainties import ufloat
import seawater as sw
plt.rcParams.update({'font.size':16})


# Graphing constants
a = ['a005', 'a095', 'a102', 'a200']
c = ['c005', 'c100', 'c105', 'c200']
c_axis = [0.5, 1.0, 1.5, 2]
markers1 = [['D', 'D', 'o', 'P'], ['D', 'o', 'o', 'P'], ['D', 'o', 'o', 'P'], ['D', 'o', 'o', 'P']] 
markers2 = [['s', '^', '^', '*'], ['s', '^', '*', '*'], ['s', '^', '*', '*'], ['p', 'p', '*', '*']] 
colors2 = {'D': 'darkgreen', '^': '#663300', 's': 'red', 'o': '#ff3399', 'P': 'darkviolet', '*': '#fcb900', 'p': '#0066ff'}
conv_id = {'a005': 'H05', 'a095': 'H09', 'a102': 'H12', 'a200': 'H20', 'c005': 'F05', 'c100': 'F10', 'c105': 'F15', 'c200': 'F20'}

# Import mixing data
Phi_d_import_up = json.load(open('Phi_d_values_{0}-{1}.txt'.format(160, 600)))
Phi_d_import_down = json.load(open('Phi_d_values_{0}-{1}.txt'.format(600, 920)))
avgs = [{}, {}, {}, {}]
for i in range(len(a)):
    avgs[i]['Phi_d_U'] = []
    avgs[i]['Phi_d_D'] = []
    for j in range(len(c)):
        avgs[i]['Phi_d_U'].append(np.mean(Phi_d_import_up[conv_id[c[j]]+conv_id[a[i]]][0][22:]))
        avgs[i]['Phi_d_D'].append(np.mean(Phi_d_import_down[conv_id[c[j]]+conv_id[a[i]]][0][22:]))

# Methods for computing values related to the figures

def Fr(z0, Si, Sf, U):
        # Computes the Fr number (Si and Sf are the summer and winter ML salinities, respectively)
        dB = 9.8*(sw.dens0(Sf, -2) - sw.dens0(Si, -2))/sw.dens0(Si, -2)
        return U/np.sqrt(z0*dB)

def eta(z0, h):
    # Computes eta
    return h/z0

def EOS(t, S):
    # EOS (see https://www.desmos.com/calculator/emvyrt6rmc)
    # Needed to write out explicitly as sw.dens0 runs into errors when used with ufloat
    a = [999.84, 6.79e-2, -9.09e-3, 1.001e-4, -1.12e-6, 6.54e-9]
    b = [8.24e-1, -4.1e-3, 7.64e-5, -8.25e-7, 5.38e-9]
    c = [-5.72e-3, 1.02e-4, -1.65e-6]
    d = [4.83e-4]
    p_w = a[0]+a[1]*t+a[2]*t**2+a[3]*t**3+a[4]*t**4+a[5]*t**5
    return p_w + (b[0]+b[1]*t+b[2]*t**2+b[3]*t**3+b[4]*t**4)*S+(c[0]+c[1]*t+c[2]*t**2)*S**(3/2)+d[0]*S**2

def sigma_rho(S, S_sigma):
    # Computes the std_dev in density from a given salinity and salinity std_Dev
    S_val = ufloat(S, S_sigma)
    rho_val = EOS(-2, S_val)
    return rho_val.std_dev

def sigma_Fr(z0, Si, Sf, z0_sigma, Si_sigma, Sf_sigma, U):
    # Computes the std_dev in the Fr number (assumed no U std_dev)
    rhoi_sigma = sigma_rho(Si, Si_sigma)
    rhof_sigma = sigma_rho(Sf, Sf_sigma)
    rhoi = ufloat(sw.dens0(Si, -2), rhoi_sigma)
    rhof = ufloat(sw.dens0(Sf, -2), rhof_sigma)
    z0u = ufloat(z0, z0_sigma)
    dB = 9.8*(rhof-rhoi)/rhoi
    Fr = U/(z0u*dB)**0.5
    return Fr.std_dev

# Methods for drawing the figures

def arctic_plot(h, color1, color2):
    # Plot a keel of height h on the arctic plot 
    
    # Set salinity and ML depth base values (with uncertainty which is the std_dev)
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
    
    # Trends of salinity, MLD, and wind speed in units of [value] per year
    # Taken from Figure 14 in PFW
    # Salinity trends are taken from the ice-covered (ic) column
    S_trends = {'Chukchi Sea': {'Si_t': 0.02, 'Sf_t': -0.07}, 
                'Southern Beaufort Sea': {'Si_t': 0.29, 'Sf_t': -0.04}, 
                'Canada Basin': {'Si_t': -0.11, 'Sf_t': -0.19}, 
                'Eurasian Basin': {'Si_t': -0.05, 'Sf_t': -0.07}, 
                'Barents Sea': {'Si_t': 0.02, 'Sf_t': 0,}} # Non significant trend in winter ML
    # MLD trends are taken from the ice-covered (ic) summer column
    z0_trends = {'Chukchi Sea': {'z0_t': 0}, # Non significant trend 
                'Southern Beaufort Sea': {'z0_t': 0.33}, 
                'Canada Basin': {'z0_t': -0.33}, 
                'Eurasian Basin': {'z0_t': -0.19}, 
                'Barents Sea': {'z0_t': 0}} # Non significant trend
    # Wind trends are taken from ice_covered (ic) summer column
    wind_trends = {'Chukchi Sea': {'u_t': 0.16},
                   'Southern Beaufort Sea': {'u_t': 0.08},
                   'Canada Basin': {'u_t': 0}, # Non significant trend
                   'Eurasian Basin': {'u_t': 0}, # Non significant trend
                   'Barents Sea': {'u_t': 0}} # Non significant trend
    
    # All regions that have a non significant trend are marked with a (*)
    labels_region = {'Chukchi Sea': '*1', 'Southern Beaufort Sea': '2', 'Canada Basin': '*3', 'Eurasian Basin': '*4', 'Barents Sea': '*5'}
    regions = S_values.keys()
    
    U = 0.2 # Ice speed (fixed for every region)

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
    years = {'Chukchi Sea': 5, 'Southern Beaufort Sea': 5, 'Canada Basin': 15, 'Eurasian Basin': 15, 'Barents Sea': 15} # How many years into the future we predict
    ls = {'Chukchi Sea': '-', 'Southern Beaufort Sea': '-', 'Canada Basin': (0,(5,3)), 'Eurasian Basin': (0,(5,3)), 'Barents Sea': (0,(5,3))}
    #U_rate = 1.009 # Ice speed trend (increase of 0.9% per year from Rampal)
    Fr_pred_values = {}
    eta_pred_values = {}
    for region in regions:
        # Compute new predicted values with trends
        z0_new = z0_values[region]['z0']+years[region]*z0_trends[region]['z0_t']
        S_i_new = S_values[region]['Si']+years[region]*S_trends[region]['Si_t']
        S_f_new = S_values[region]['Sf']+years[region]*S_trends[region]['Sf_t']
        #U_new = U*U_rate**years
        U_new = U + years[region]*0.02*wind_trends[region]['u_t'] # Sea ice speed 2% of wind speed

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
        print(h, region, (eta_pred_value - eta_value)/(years[region]))
        # Plot error boxes
        plt.gca().add_patch(patches.FancyBboxPatch(xy=(Fr_value-Fr_er, eta_value-eta_er), width=2*Fr_er, height=2*eta_er, linewidth=1, color=color2, fill='false', mutation_scale=0.05, alpha=0.13))
        # Plot predictive arrows
        if region == 'Canada Basin' and h == 7.45*2.5:
            plt.arrow(x=Fr_value, y=eta_value, dx=Fr_pred_value-Fr_value, dy=eta_pred_value-eta_value, linestyle='--', linewidth=2.8, length_includes_head=True, zorder=98+k, head_width=0.03, head_length=0.03)
        else:
            plt.annotate('', xy=(Fr_pred_value+0.02, eta_pred_value), xytext=(Fr_value, eta_value), arrowprops = dict(arrowstyle="->", color='k', linewidth=2.8, ls=ls[region]), zorder=98+k)
        #plt.arrow(x=Fr_value, y=eta_value, dx=Fr_pred_value-Fr_value, dy=eta_pred_value-eta_value, linestyle='-', linewidth=2.8, length_includes_head=True, zorder=98+k, head_width=0.03, head_length=0.03)
        # Plot marker boxes
        plt.plot(Fr_value, eta_value, marker='s', color=color1, ms=18, zorder=101+k, markeredgecolor='k')
        # Plot text in marker boxes
        plt.text(Fr_value, eta_value-0.012, labels_region[region], fontsize=14, weight='bold', ha='center', va='center', zorder=101+k)
        k += 1

def joint_regime_arctic():
    #Joint regime Arctic layout 
    shift = 0 # offset the marker for each point
    scale = 12 # Scaling for marker size 
    Phi_0 = avgs[0]['Phi_d_U'][0] # Normalize mixing values
    ms = {'o': 17.5, 'D': 15, 'P': 18, 's': 17, '^': 18, '*': 20.5, 'p':17}
    # Plot upstream markers
    for i in range(len(a)):
        for j in range(len(c)):
            marker_size = avgs[i]['Phi_d_U'][j]/Phi_0 * scale # Chosen through trial and error (feel free to change)
            #marker_size = 14
            shift = (ms[markers1[i][j]]-13+12*marker_size)/6500
            plt.plot(c_axis[j]-shift, [0.5, 0.95, 1.2, 2][i], markeredgecolor='k', linestyle='None', marker=markers1[i][j], color=colors2[markers1[i][j]], ms=ms[markers1[i][j]]-13+marker_size, zorder=10)    
    line2, = plt.plot([], [], marker='o', markeredgecolor='k', linestyle='None', color=colors2['o'], label='Unstable Subcritical', ms=ms['o'] * 0.75)
    line3, = plt.plot([], [], marker='D', markeredgecolor='k', linestyle='None', color=colors2['D'], label='Stable Subcritical', ms=ms['D'] * 0.75)
    line1, = plt.plot([], [], marker='P', markeredgecolor='k', linestyle='None', color=colors2['P'], label='Unstable Supercritical', ms=ms['P'] * 0.75)
    # Plot downstream markers
    for i in range(len(a)):
        for j in range(len(c)):
            marker_size = avgs[i]['Phi_d_D'][j]/Phi_0 * scale # Chosen through trial and error (feel free to change)
            #marker_size = 14
            shift = (ms[markers1[i][j]]-13+12*marker_size)/6500
            plt.plot(c_axis[j]+shift, [0.5, 0.95, 1.2, 2][i], markeredgecolor='k', linestyle='None', marker=markers2[i][j], color=colors2[markers2[i][j]], ms=ms[markers2[i][j]]-13+marker_size, zorder=10)    
    line11, = plt.plot([], [], marker='p', markeredgecolor='k', linestyle='None', color=colors2['p'], label='Diffusive BL', ms=ms['p'] * 0.75)
    line9, = plt.plot([], [], marker='s', markeredgecolor='k', linestyle='None', color=colors2['s'], label='Lee Waves', ms=ms['s'] * 0.75)
    line7, = plt.plot([], [], marker='^', markeredgecolor='k', linestyle='None', color=colors2['^'], label='Fast-Laminar', ms=ms['^'] * 0.75)
    line5, = plt.plot([], [], marker='*', markeredgecolor='k', linestyle='None', color=colors2['*'], label='Vortex Shedding', ms=ms['*'] * 0.75)
    # Plot legends
    first_legend = plt.legend(handles=[line1, line2, line3], loc='lower center', bbox_to_anchor=(0.26, -0.42), prop={'size': 13}, fancybox=True, shadow=True)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=[line5, line7, line9, line11], loc='lower center', bbox_to_anchor=(0.75, -0.475), prop={'size': 13}, fancybox=True, shadow=True)
    # Other plot details
    plt.xlim(0,2.2)
    plt.ylim(0,2.7)
    plt.xlabel('Froude Number $Fr$')
    plt.ylabel('Dimensionless Keel Draft $\\eta$    ')
    plt.xticks([0.5, 1, 1.5, 2])
    plt.yticks([0.5, 1, 1.5, 2, 2.5])
    plt.text(0.58, -0.49, 'Upstream', ha='center', va='center')
    plt.text(1.66, -0.49, 'Downstream', ha='center', va='center')
    plt.grid(zorder=0)

    # Add Arctic region markers (the most importnat part)
    arctic_plot(h=7.45, color1='#e59866', color2='#e59866')
    arctic_plot(h=7.45*2.5, color1='cyan', color2='cyan')

    plt.gcf().set_size_inches(8,6, forward=True)
    plt.savefig('regime_layout_regional.pdf', format='pdf', bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    joint_regime_arctic() # Create figure
