# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 00:34:58 2024

@author: mmanc
"""

#%%
import einsteinpy
einsteinpy.utils.exceptions.BasedError()


#%%
import numpy as np
import time 
# =============================================================================
# Timelike
# =============================================================================

start_time = time.time()

from einsteinpy.geodesic import Timelike
from einsteinpy.plotting import StaticGeodesicPlotter, GeodesicPlotter
b = 0.767851
# Constant Radius Orbit
position = [10, np.pi/ 3, 0.] # [4, np.pi / 3, 0.] original
momentum = [0, b, -2.] # [0., 0.767851, 2.]
a = 0.99 # 0.99
steps = 10000 # set to 10000 for pi/4
delta = 0.005 #set to 0.5 by default
omega = 0.01
suppress_warnings = True

geod = Timelike(
    metric="Kerr",
    metric_params = (a,),
    position=position,
    momentum=momentum,
    steps=steps,
    delta=delta,  
    return_cartesian=True,
    omega=omega,
    suppress_warnings=suppress_warnings
)

end_time = time.time()
ellapse = end_time - start_time
print('Time ellapse is', ellapse, 'seconds')
#%%
step = 7915
# Using my own plot
import matplotlib.pyplot as plt
trajectory = geod.trajectory[1]
x1_list = []
x2_list = []
x3_list = []
iterations = []
for i in range(0,step):
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

def kerr_orbit_3D(a):
    
    trajectory = geod.trajectory[1]
    x1_list = []
    x2_list = []
    x3_list = []
    iterations = []
    for i in range(0,step):
        x1 = trajectory[i][1] # X1 values
        x2 = trajectory[i][2] # X2 values
        x3 = trajectory[i][3]
        ite = i # keep the iteartions
        x1_list.append(x1)
        x2_list.append(x2)
        x3_list.append(x3)
        iterations.append(ite)
    
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
    ax.plot3D(x1_list, x2_list, x3_list, color = 'blue', label = 'Time-like geodesic')
    ax.set_title(f"3D plot of time-like geodisic, Kerr balck hole (a = {a})")

    ax.set_xlabel(r'$\frac{X}{R_s}$', labelpad=20)
    ax.set_ylabel(r'$\frac{Y}{R_s}$', labelpad=20)
    ax.set_zlabel(r'$\frac{Z}{R_s}$', labelpad=20)
    
    b1 = position[0] - 4
    ax.legend()
    limx = b1
    limy = b1
    limz = b1
    
    ax.set_xlim3d(-limx, limx)
    ax.set_ylim3d(-limy, limy)
    ax.set_zlim3d(-limz, limz)
    
    plt.show()

def kerr_orbit_2D(a):
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    r_bh = horizon(Rs, a)
    r_ergo = ergosphere(Rs, a, phi)
    
    # Convert spherical coordinates to Cartesian coordinates
    x_ergo = r_ergo * np.sin(theta) * np.cos(phi)
    y_ergo = r_ergo * np.sin(theta) * np.sin(phi)
    
    x_bh = r_bh * np.sin(theta) * np.cos(phi)
    y_bh = r_bh * np.sin(theta) * np.sin(phi)
    
    plt.plot(x1_list, x2_list, color = 'blue', label = 'Time-like geodesic')
    plt.fill(x_ergo, y_ergo, color = 'gray', alpha = 0.5)
    plt.fill(x_bh, y_bh, color = 'black', alpha = 0.5)
    plt.scatter([], [], color = 'gray', label = 'Ergosphere')
    plt.scatter([], [], color = 'black', label = 'Outer horizon')
    plt.legend(loc= 'upper right', borderaxespad=0.)
    plt.title(f"2D plot of time-like geodisic, Kerr balck hole a = {a}")
    plt.axis('equal')
    plt.show()
    

#%%
#plotting the result
kerr_orbit_3D(a) # 3D plot
kerr_orbit_2D(a) # 2D plot
#%%trajectory = geod.trajectory[1]
step1 = 7900
x1_list = []
x2_list = []
x3_list = []
iterations = []
for i in range(0,step1):
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
# determining the valuue of r
r_list = []
for i in range(0, step1):
    r = np.sqrt(x1_list[i]**2 + x2_list[i]**2 + x3_list[i]**2)
    r_list.append(r)

plt.plot(iterations, r_list)
plt.title(r'r against the affine parameter $\lambda$')
plt.xlabel(r'Affine parameter $\lambda$')
plt.ylabel('r')
plt.show()
#%%
# Compare with default plotting from EinsteinPy
gpl = GeodesicPlotter(ax=None, bh_colors=('#000', '#FFC'), draw_ergosphere=True)
#matplotlib nbagg
gpl.plot(geod, color = 'blue')
gpl.show()

#%%
# Animation plot
sgpl = StaticGeodesicPlotter()
sgpl.animate(geod, interval=1)
sgpl.show()
#%%
gpl.parametric_plot(geod)
gpl.show()
#%%
gpl.plot2D(geod, coordinates=(1,2))
gpl.show()

#%%
# =============================================================================
# Lightlike
# =============================================================================

import numpy as np
import time

from einsteinpy.geodesic import Nulllike
from einsteinpy.plotting import StaticGeodesicPlotter, GeodesicPlotter

start = time.time()

# Constant Radius Orbit
position = [3, np.pi/ 2, 0.] # [4, np.pi / 3, 0.] original
momentum = [0, 0, -2.] # [0., 0.767851, 2.]
a = 0 # 0.99
steps = 150000 # set to 10000 for pi/4
delta = 0.0001 #set to 0.5 by default
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
end = time.time()
ellapse = end - start
print('Ellapse is', ellapse, 'seconds' )

#%%
step = steps
alpha = 0.3
# Using my own plot
import matplotlib.pyplot as plt
trajectory = geod.trajectory[1]
x1_list = []
x2_list = []
x3_list = []
iterations = []
for i in range(0,step):
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

def kerr_orbit_3D(a):
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

    # Plot event horizon
    ax.plot_surface(x_bh, y_bh, z_bh, color='black', alpha = alpha)
    ax.plot([], [], [], 'ko', label='Outer horizon', alpha = 0.5)
    
    if a>0:
        ax.plot_surface(x_ergo, y_ergo, z_ergo, color='gray', alpha=alpha)
        # Plot ergosphere
        ax.plot([], [], [], 'o',color = 'gray', label='Ergosphere', alpha = 0.5)
        ax.set_title(f"3D plot of light-like geodisic, Kerr balck hole (a = {a})")
    if a==0:
        ax.set_title("3D plot of light-like geodisic, Schwarzschild balck hole")
        
    #ax1 = plt.axes(projection = '3d')
    ax.grid()
    ax.plot3D(x1_list, x2_list, x3_list, color = 'darkorange', label = 'Light-like geodesic')
    
    ax.set_xlabel(r'$\frac{X}{R_s}$', labelpad=20)
    ax.set_ylabel(r'$\frac{Y}{R_s}$', labelpad=20)
    ax.set_zlabel(r'$\frac{Z}{R_s}$', labelpad=20)
    b1 = 3
    ax.legend()
    limx = b1
    limy = b1
    limz = b1
    
    ax.set_xlim3d(-limx, limx)
    ax.set_ylim3d(-limy, limy)
    ax.set_zlim3d(-limz, limz)
    
    plt.show()

def kerr_orbit_2D(a):
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    r_bh = horizon(Rs, a)
    r_ergo = ergosphere(Rs, a, phi)
    
    # Convert spherical coordinates to Cartesian coordinates
    x_ergo = r_ergo * np.sin(theta) * np.cos(phi)
    y_ergo = r_ergo * np.sin(theta) * np.sin(phi)
    
    x_bh = r_bh * np.sin(theta) * np.cos(phi)
    y_bh = r_bh * np.sin(theta) * np.sin(phi)
    
    
    plt.plot(x1_list, x2_list, color = 'darkorange', label = 'Light-like geodesic')
    if a>0:
        plt.title(f"2D plot of light-like geodisic, Kerr balck hole a = {a}")
        plt.fill(x_ergo, y_ergo, color = 'gray', alpha = 0.5)
    if a==0:
        plt.title("2D plot of light-like geodisic, Schwarzschild balck hole")
    plt.fill(x_bh, y_bh, color = 'black', alpha = 0.5)
    plt.scatter([], [], color = 'gray', label = 'Ergosphere')
    plt.scatter([], [], color = 'black', label = 'Outer horizon')
    plt.legend(loc= 'upper right', borderaxespad=0.)

    plt.axis('equal')
    plt.show()
    

#%%
#plotting the result
kerr_orbit_3D(a) # 3D plot
kerr_orbit_2D(a) # 2D plot
#%%trajectory = geod.trajectory[1]
step1 = steps
x1_list = []
x2_list = []
x3_list = []
iterations = []
for i in range(0,step1):
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
# determining the valuue of r
r_list = []
for i in range(0, step1):
    r = np.sqrt(x1_list[i]**2 + x2_list[i]**2 + x3_list[i]**2)
    r_list.append(r)

plt.plot(iterations, r_list)
plt.title(r'r against the affine parameter $\lambda$')
plt.xlabel(r'Affine parameter $\lambda$')
plt.ylabel('r')
plt.show()


