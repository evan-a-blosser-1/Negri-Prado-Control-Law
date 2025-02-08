import os
import sys
import numpy as np
from numba import jit,njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import trimesh
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
sys.dont_write_bytecode = True
import ASTER_Pack as AP
import constants as C
const    = C.constants()
target = C.DA1950()
#####################################
# Asteroid Name
asteroid = '1950DA_Prograde'
#####################################
# Asteroid Rotation Period in seconds
Spin_Rate = 2.1216
omega = ((2*np.pi)/Spin_Rate) *(1/3600)
Gamma =  target.gamma
Esc_Lim = 72.32565 # km 
###########################################################
################################################ Load files
OBJ_F = asteroid + '.obj'
# MASCONs (tetrahedron center of masses)
CM = np.loadtxt(asteroid + '_CM.in' , delimiter=' ',dtype=float)
Terta_Count = len(CM)
########################################
# Gravitational Parameter of each MASCON
mu_I = np.loadtxt(asteroid + '_mu.in', delimiter=' ')
mu = np.sum(mu_I)
# Polyhedron Center of Mass
Poly_CM = [target.Poly_x,target.Poly_y,target.Poly_z]
#################################
#######################################
############################## Settings
Data_PATH = 'Databank/'
isExist = os.path.exists(Data_PATH)
if not isExist:
    os.mkdir(Data_PATH)
    print(f"| Data Stored in: {Data_PATH} ")
#######################################
#######################################.
# Initial Position (km)
y = 3.0
########  0.13395027e-6
# Hamiltonian Energy (km^2/s^2)
Ham = 3.0e-6
#################################################################
##################### Begin Negri Prado Law #####################
#################################################################
################################################
############################################ EOM
def EOM_MASCON(a):
    x,y,z,vx,vy,vz, arb = a
    dxdt = vx
    dydt = vy
    dzdt = vz
    ########
    points = 1
    Ux = np.zeros(points, dtype="float64")
    Uy = np.zeros(points, dtype="float64")
    Uz = np.zeros(points, dtype="float64")
    for it in range(Terta_Count):
        R_x = CM[it,0] - Poly_CM[0]
        R_y = CM[it,1] - Poly_CM[1]
        R_z = CM[it,2] - Poly_CM[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_x = a[0] - R_x
        r_y = a[1] - R_y
        r_z = a[2] - R_z
        # Magnitude
        vector = np.array([r_x, r_y, r_z])
        # r_i   = np.sqrt( r_x**2 + r_y**2 + r_z**2 )
        r_i = np.linalg.norm(vector)
        # Potential in the x,y,z-direction
        Ux += - (mu_I[it]*r_x)/(r_i**3)
        Uy += - (mu_I[it]*r_y)/(r_i**3)
        Uz += - (mu_I[it]*r_z)/(r_i**3)
    #### EXIT LOOP ##################
    #################################
    A = np.array([[0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [omega**2, 0, 0, 0, 2*omega, 0, Ux[0]],
                  [0, omega**2, 0, -2*omega, 0, 0, Uy[0]],
                  [0, 0, 0, 0, 0, 0, Uz[0]],
                  [0, 0, 0, 0, 0, 0, 0] # Dummy state
                  ])
    x_in = np.array([a[0], a[1], a[2], a[3], a[4], a[5], 1])
    #######################################################
    B = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]  # Dummy state
    ])
    #################################################
    #################################################
    return A , B

################################################
################################################
# page 7 paragraph 2
# " We distributed our initial conditions in the y-axis, with x0 = z0 = y_dot_0 = z_dot_0 =
#  0 and x_dot_ 0 was computed according to Eq. 1."
def v_calc(Ham,omega,mu_I,CM,yp):
    U = np.zeros(1, dtype="float64")
    for it in range(len(CM)):
        x = 0.0  - CM[it,0]
        y = yp   - CM[it,1]
        z = 0.0  - CM[it,2]
        r = np.sqrt(x**2 + y**2 + z**2)
        U += mu_I[it]/r
    #########################
    psu = U[0]
    cori = (omega**2)*(x**2 + y**2)
    print(f"|   Ham:      {Ham} (km^2/s^2) ")
    print(f"|   Psuedo:   {psu} (km^2/s^2) ")
    print(f"|   Coriolis: {cori} (km^2/s^2) ")
    arg =  -2*Ham + cori + 2*psu
    if arg > 0:
        V = np.sqrt(arg)
    return V
############################################
################################################
################################################
####################################################
####################################################
########################################### Main
### Solve Hamiltonian for initial velocity
x_dot = v_calc(Ham,omega,mu_I,CM,y)
print(f"|  y: {y} x_dot: {x_dot}")
#
# Define initial conditions for this iteration
#      x0  y0      x_dot  y_dot
a0 = [ 0.0, y, 0.0, x_dot, 0.0,  0.0, 0.0]
##################################################
##################################################
###################################################################
###################################################################
A_stat, B_ctrl = EOM_MASCON(a0)
###################################################################
Controllability = np.array([B_ctrl,
                            A_stat @ B_ctrl, 
                            A_stat**2 @ B_ctrl,
                            A_stat**3 @ B_ctrl,
                            A_stat**4 @ B_ctrl,
                            A_stat**5 @ B_ctrl,
                            A_stat**6 @ B_ctrl])
# print(f"| Controllability: \n {np.linalg.matrix_rank(Controllability)}")
print(f"| IFF = n = 7, then the system is controllable")

print(f"| Controllability Matrix: \n {Controllability}")    
######
#
#
# Controllability matrix
n = A_stat.shape[0]
controllability_matrix = B_ctrl
for i in range(1, n):
    controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A_stat, i) @ B_ctrl))
print(controllability_matrix)
rank = np.linalg.matrix_rank(controllability_matrix)
print(f"Rank of controllability matrix: {rank} --> n = {n}")


