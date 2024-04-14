# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:02:38 2024

@author: mmanc
"""

import numpy as np

from einsteinpy.geodesic import Nulllike
from einsteinpy.plotting import StaticGeodesicPlotter, GeodesicPlotter
#%%
position = [2.5, np.pi / 2, 0.]
momentum = [0., 0., -2.]
a = 0.99 # 0.99
steps = 7440 # As close as we can get before the integration becomes highly unstable 7440
delta = 0.0005
omega = 0.01
suppress_warnings = True

geod = Nulllike(
    metric="Kerr",
    metric_params=(a,),
    position=position,
    momentum=momentum,
    steps=steps,
    delta=delta,
    return_cartesian=True,
    omega=omega,
    suppress_warnings=suppress_warnings
)

gpl = GeodesicPlotter(ax=None, bh_colors=('#000', '#FFC'), draw_ergosphere=True)
#matplotlib nbagg
gpl.plot(geod, color = 'blue')
gpl.show()
#%%
sgpl = StaticGeodesicPlotter(bh_colors=("red", "blue"))
sgpl.plot(geod, color="blue")
sgpl.show()
#%%
sgpl = StaticGeodesicPlotter(bh_colors=("red", "blue"))
sgpl.plot2D(geod, coordinates=(1, 2), figsize=(10, 10), color="indigo") # Plot X vs Y
sgpl.show()
#%%
import matplotlib.pyplot as plt
trajectory = geod.trajectory[1]
x1_list = []
x2_list = []
x3_list = []
iterations = []
for i in range(0,steps):
    x1 = trajectory[i][1] # X1 values
    x2 = trajectory[i][2] # X2 values
    x3 = trajectory[i][3]
    ite = i # keep the iteartions
    x1_list.append(x1)
    x2_list.append(x2)
    x3_list.append(x3)
    iterations.append(ite)
# plotting the results
plt.plot(iterations, x1_list, color = 'red', label = r'$X_1$ (cartesian)')
plt.plot(iterations, x2_list, color = 'blue', label = r'$X_2$ (cartesian)')
plt.plot(iterations, x3_list, color = 'purple', label = r'$X_3$ (cartesian)')
plt.legend(loc = 'upper right', bbox_to_anchor = (1.5, 1))
plt.title(r'$X_1$, $X_2$ and $X_3$ in cartesian')
plt.xlabel(r'Affine parameter $\lambda$')
plt.ylabel('Coordinates')
plt.show()

#%%
gpl._draw_bh(a)
gpl.show()
#%%
ax = plt.axes(projection = '3d')
ax.grid()
ax.plot3D(x1_list, x2_list, x3_list, color = 'blue', label = 'orbit')
ax.set_title("Particle's trajectory")
#gpl.plot(Timelike(metric = 'Kerr', metric_params=(a,), position = [], momentum = []))

ax.set_xlabel(r'$\frac{x}{R_s}$', labelpad=20)
ax.set_ylabel(r'$\frac{y}{R_s}$', labelpad=20)
ax.set_zlabel(r'$\frac{z}{R_s}$', labelpad=20)

plt.show()
#%%
# determining the valuue of r
r_list = []
for i in range(0, steps):
    r = np.sqrt(x1_list[i]**2 + x2_list[i]**2 + x3_list[i]**2)
    r_list.append(r)

plt.plot(iterations, r_list)
plt.title(r'r against the affine parameter $\lambda$')
plt.xlabel(r'Affine parameter $\lambda$')
plt.ylabel('r')
plt.show()

#%%
# Using my own plot

trajectory = geod.trajectory[1]
x1_list = []
x2_list = []
x3_list = []
iterations = []
for i in range(0,steps):
    x1 = trajectory[i][1] # X1 values
    x2 = trajectory[i][2] # X2 values
    x3 = trajectory[i][3]
    ite = i # keep the iteartions
    x1_list.append(x1)
    x2_list.append(x2)
    x3_list.append(x3)
    iterations.append(ite)

Rs = 2

def ergosphere(r, a, theta):
    return (Rs + np.sqrt(Rs**2 - 4*a**2*np.cos(theta)**2))/2
def horizon(r, a):
    return (r + np.sqrt(r**2 - 4*a**3))/2

def kerr_black_hole(a):
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    r_bh = horizon(Rs, a)
    r_ergo = ergosphere(Rs, a, phi)
    
    # Convert spherical coordinates to Cartesian coordinates
    x_ergo = r_ergo * np.sin(theta) * np.cos(phi)
    y_ergo = r_ergo * np.sin(theta) * np.sin(phi)
    z_ergo = r_ergo * np.cos(theta)
    
    x_bh = r_bh * np.sin(theta) * np.cos(phi)
    y_bh = r_bh * np.sin(theta) * np.sin(phi)
    z_bh = r_bh * np.cos(theta)
    
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ergosphere
    ax.plot_surface(x_ergo, y_ergo, z_ergo, color='gray', alpha=0.5)
    ax.plot([], [], [], 'o',color = 'gray', label='Ergosphere', alpha = 0.5)
    # Plot event horizon
    ax.plot_surface(x_bh, y_bh, z_bh, color='black', alpha = 0.8)
    ax.plot([], [], [], 'ko', label='Outer horizon', alpha = 0.8)
    
    #ax1 = plt.axes(projection = '3d')
    ax.grid()
    ax.plot3D(x1_list, x2_list, x3_list, color = 'blue', label = 'Orbit')
    ax.set_title(f"Particle's trajectory around the Kerr black hole (a={a})")
    #gpl.plot(Timelike(metric = 'Kerr', metric_params=(a,), position = [], momentum = []))

    ax.set_xlabel(r'$\frac{x}{R_s}$', labelpad=20)
    ax.set_ylabel(r'$\frac{y}{R_s}$', labelpad=20)
    ax.set_zlabel(r'$\frac{z}{R_s}$', labelpad=20)
    
    
    
    ax.legend()
    limx = 5
    limy = 5
    limz = 5
    
    ax.set_xlim3d(-limx, limx)
    ax.set_ylim3d(-limy, limy)
    ax.set_zlim3d(-limz, limz)
    
    plt.show()
    
kerr_black_hole(a)
