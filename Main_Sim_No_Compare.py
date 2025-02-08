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
##
# z hieght override
z = 0.0
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
###################
###################
# Period of asteroid rotation
T = 2 * np.pi * np.sqrt(y**3/mu)
################################
# 1 seconds
dt_min = T/T
##################
# freezing at 0.185763888888 days 
days       = 0.5
Start_Time = 0.0
End_Time   = days*const.day
# sN =  dt/step
dN = round((End_Time - Start_Time)/dt_min)
##########################################
Time = np.linspace(start=Start_Time,
                   stop=End_Time,
                   num=dN
                   )
#################################################################
##################### Begin Negri Prado Law #####################
#################################################################
# Control Law switch (I/O) design of on/off
#
# Set to on as Negri Prado is first sim
CTRL_F = 1
# Orbital elements (Inclination, 
#           Longitude of the Ascending Node, Argument of Perigee)
# ALl orbital elements set to zero
id = np.radians(0)
Omd = np.radians(0)
argP = np.radians(0)
##############################################################
####################################### Controller Design 
# Parameters design to determine the asymptotic convergence
#  to the sliding surface.
##########################
# Radial
LambR = 0.53
# Normal 
LambN = 0.53
#################
# set K > D_RTN
# This is a very sensitive parameter
#   Setting this too high will result in
#    a large correction and instability
#
k_11_p = 1e-11
k_22_p = 1e-15
k_33_p = 1e-13
# size of the boundary about the sliding surface
# used for creating the Phi matrix from the K matrix
n_phi = 10.0
###########
# Orbit eccentricity
# e = 0.005 approx to zero
# times to desired vector
ecc = 0.005
#####################################
# Disturbance matrix, for designing 
# the K-matrix
D = np.array([1e-14, 1e-14, 1e-14])
######################################
Initialize = f"""
{"-"*42}
|   Negri Prado Control Law
|
{"-"*42}
|   Asteroid:    {asteroid}
|   tetra count: {Terta_Count}
|   mu:           {mu:.5e} km^3/s^2
|   Omega:       {omega}  rad/s
|
|   Asteroid Rotational 
|    Period:     {T} s
{'-'*42}
| ---------- Integration Settings ----------
{'-'*42}
|   Start Time:    {Start_Time} sec
|   End Time:      {End_Time} sec
|   Integration:   {dN} steps
{'-'*42}
|   Integrating every: {dt_min} sec.
|   Total Time: {End_Time} sec.
|   Days: {days}
{"-"*42}
"""
print(Initialize)
Accel_Correct, Integration_time = [], []
################################################
############################################ EOM
def EOM_MASCON(t,a):
    #print(f"| Time: {t} s")
    AP.print_progress(t,End_Time)
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
    r = np.array([x,y,z])
    r_dot = np.array([vx,vy,vz])
    h = np.cross(r, r_dot)
    ######################
    # RTN unit vectors 
    r_hat = r / np.linalg.norm(r)
    h_hat = h / np.linalg.norm(h)
    Theta_til =  np.cross(h_hat, r_hat)
    #
    # eccentricity vector
    e = (1 / mu) * (np.cross(r_dot, h) - mu * r_hat)
    #
    # sliding surface
    s = np.array([
        (e - ed).dot(LambR * r_hat + Theta_til),
        np.linalg.norm(h) - np.linalg.norm(hd),
        h_hat.dot(LambN * r_hat + Theta_til)
    ])
    #
    f_11 = -np.linalg.norm(h)**2
    f_12 = (2 * LambR *np.linalg.norm(h) - np.dot(r_dot, r_hat) * np.linalg.norm(r))*np.linalg.norm(h)
    f_13 = -mu * np.linalg.norm(r)* np.dot(ed, h_hat)
    f_22 = mu * np.linalg.norm(r)*np.linalg.norm(h)
    f_33 = mu * np.linalg.norm(r)* np.dot(hd, h_hat)
    #
    F = (1/(np.linalg.norm(h)*mu))*np.array([
        [f_11, f_12, f_13 ],
        [0, f_22, 0 ],
        [0, 0, f_33]
    ])
    G = ((np.linalg.norm(h)/np.linalg.norm(r)**2))*np.array([
        (e - ed).dot(LambR * Theta_til - r_hat),
        0.0,
        hd.dot(LambN * Theta_til - r_hat)
    ])

    # Calculate the control input in RTN coordinates
    Term1 = G + K.dot(sat(s, PHI))
    uRTN = -np.linalg.inv(F).dot(Term1)

    # Calculate the final acceleration corrections in the frame of reference.
    #  This is similar to a rotation matrix?
    # 
    if CTRL_F == 1:
        
        u = uRTN[0] * r_hat + uRTN[1] * Theta_til + uRTN[2] * h_hat
        Accel_Correct.append((u[0], u[1], u[2],t))
        
    if CTRL_Non == 1:
        u = np.zeros(3, dtype="float64")

    ############################
    # Integrating in cartesian coordinates
    #
    # u_cart = AP.rtn_rot(a, u,choice="CART")
    
    # dvxdt = (omega**2)*x + 2*omega*vy + Ux[0] + u[0]
    # dvydt = (omega**2)*y - 2*omega*vx + Uy[0] + u[1]
    # dvzdt = Uz[0] + u[2]
    # State = [dxdt,dydt,dzdt,dvxdt,dvydt,dvzdt]
    
    ################
    # Integrating in RTN coordinates
    #
    # dvxdt = (omega**2)*x + 2*omega*vy + Ux[0] 
    # dvydt = (omega**2)*y - 2*omega*vx + Uy[0] 
    # dvzdt = Uz[0] 
    # dadt  = [dvxdt,dvydt,dvzdt]
    # dvdt  = [dxdt,dydt,dzdt]
    # accel_RTN = rtn_rot(a, dadt,choice="RTN")
    # vel_RTN = rtn_rot(a, dvdt,choice="RTN")
    # #
    # A_state = accel_RTN + u
    # State = [vel_RTN[0],vel_RTN[1],vel_RTN[2],A_state[0],A_state[1],A_state[2]]
    ################
    State  = A @ x_in  +  B @ u
    # ################

    return State

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
def Collision(t,a):
    global cond, critical
    # Load the mesh
    mesh = trimesh.load_mesh(OBJ_F)
    # Define the scale factor
    mesh.apply_scale(Gamma)
    # Define the point coordinates
    point_coords = np.array([a[0], a[1], a[2]])
    r_mag = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    # Check if the point is inside the mesh
    is_inside = mesh.contains([point_coords])[0]
    # If the point is not inside, check if it is on the surface
    if not is_inside:
        # Find the closest point on the mesh surface
        closest_point, distance, triangle_ids = mesh.nearest.on_surface([point_coords])

        # within 1 meter of the surface 0.001 km
        #
        #
        tolerance = 0.0002

        # Check if the point is on the surface
        is_on_surface = distance[0] < tolerance
    else:
        is_on_surface = False
    # Initialize
    cond = 0
    #######################################
    ################### Collision Detection
    if is_inside:
        print(f"The point {point_coords} is inside the mesh.")
        cond = 1
    elif is_on_surface:
        print(f"The point {point_coords} is on the surface of the mesh.")
        cond = 1
    #######################################
    if cond != 0:
        critical = 0
    else:
        critical = 1

    return critical
# Test -1 for escape as well?
Collision.direction = -1
Collision.terminal = True


def Escape(t,a):
    global cond, critical
    # Distance from the center of mass
    R_mag = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    #
    cond = 0
    ###############################################################
    ############################################## Escape Detection
    # Hill Sphere for the Asteroid 1950DA
    if R_mag >= Esc_Lim:
        # print(f"Escape from the system, R = {R_mag} km")
        cond = 9


    if cond != 0:
        critical = 0
    else:
        critical = 1

    return critical

Escape.direction = 1
Escape.terminal = True
####################################################
####################################################
########################################### Main
### Solve Hamiltonian for initial velocity
x_dot = v_calc(Ham,omega,mu_I,CM,y)
print(f"|  y: {y} x_dot: {x_dot}")
#
# Define initial conditions for this iteration
#      x0  y0      x_dot  y_dot
#
# set z=0.6km so that the scan is of the entire asteroid
a0 = [ 0.0, y, z, x_dot, 0.0,  0.0, 0.0]
##################################################
##################################################
###################################################################
###################################################################
# Define the desired eccentricity vector and angular momentum vector
#
# Negri Prado 2022 says it is times the eccentricity, 
# but this does show in Aljbaae 2021
# 
ed = ecc * np.array([
    np.cos(Omd) * np.cos(argP) - np.sin(Omd) * np.sin(argP) * np.cos(id),
    np.sin(Omd) * np.cos(argP) - np.cos(Omd) * np.sin(argP) * np.cos(id),
    np.sin(argP) * np.sin(id)
])
#
hd = np.array([
    np.sin(id) * np.sin(Omd),
    -np.sin(id) * np.cos(Omd),
    np.cos(id)
])
###################################
## D_RTN (rad, Trans, Norm) km/s^2  
# |d|_RTN < D_RTN where
#  d_RTN is our unkwowns from the rotating frame 
# km/s^2 
# d_BF = np.array([ 2 * omega * a0[4] + omega**2 * a0[0] , 
#                  - 2 * omega * a0[3] + omega**2 * a0[1],
#                  0.0])
#############################################
# Ux0, Uy0, Uz0 = AP.MASCON_SUM(a0, CM, mu_I, Poly_CM)
# G_poly = np.array([Ux0, Uy0, Uz0])
# R_z = np.array([
#         [np.cos(omega), -np.sin(omega), 0],
#         [np.sin(omega), np.cos(omega), 0],
#         [0, 0, 1]])
# # ??????
# d_I_poly = R_z @ G_poly +  (mu/(np.linalg.norm(r0)**3)) * r0 
#D = d_BF 
# print((np.linalg.norm(h)/mu) * D[0])
# print(abs((2*LambR*np.linalg.norm(h) - np.dot(r_dot,r))/(np.linalg.norm(h))) * D[1])
# print(np.linalg.norm(r)*(abs(ed[2])/np.linalg.norm(h)) * D[2])
####################
# Initial Definition of RTN frame
r0 = np.array([0.0, y, 0.0])
r_dot0 = np.array([x_dot, 0.0, 0.0])
h0 = np.cross(r0, r_dot0)
r_hat0 = r0 / np.linalg.norm(r0)
h_hat0 = h0 / np.linalg.norm(h0)
#
####### 
# Define the saturation function
def sat(alpha, beta):
    """ Saturation function

    This is for the Negri Prado Control Law
    to deal with discontinuities in the control
    of the sliding mode
        
    Args:
        alpha (3x1 vector): 
        beta  (3x1 vector): 

    Returns:
        3x1 vector: ?
    """
    result = np.zeros_like(alpha)
    for i in range(len(alpha)):
        if alpha[i] > beta[i]:
            result[i] = 1
        elif alpha[i] < -beta[i]:
            result[i] = -1
        else:
            result[i] = abs(alpha[i] / beta[i])
    return result
#
############################################
# Find lower bounds for the gains
k_11 = (np.linalg.norm(h0)/mu) * D[0] +\
        abs((2*LambR*np.linalg.norm(h0) - np.dot(r_dot0,r0))/(np.linalg.norm(h0))) * D[1] +\
        np.linalg.norm(r0)*(abs(ed[2])/np.linalg.norm(h0)) * D[2]
#
k_22 = np.linalg.norm(r0) * D[1]
k_33 = np.linalg.norm(r0) *( hd[2]/np.linalg.norm(h0)) * D[2]

#################
# Build the gain matrix
K = np.array([
    [k_11 + k_11_p, 0.0, 0.0],
    [0.000, k_22 + k_22_p, 0.0],
    [0.000, 0.0, k_33 + k_33_p]
])
####
# override
# K = np.array([
#     [0.5, 0.0, 0.0],
#     [0.000, 0.50, 0.0],
#     [0.000, 0.0, 0.50]
# ])
###########
# Design parameter for the sliding surface
#  such that 
PHI = n_phi*np.diag(K)
###
# override
# PHI = np.array([0.5, 0.5,0.5])
Ctrl_Sett_Out = f"""
{"-"*42}
|{"-"*13} Design Param {"-"*13}|
{"-"*42}
|  Eccentricity: {np.linalg.norm(ed)} 
{"-"*42}
|{"-"*15} D Matrix {"-"*15}|
{D}
| D = {np.linalg.norm(D)} km/s^2 
{"-"*42}
|{"-"*14} PHI Matrix {"-"*14}|
{PHI}
{"-"*42}
|{"-"*13} Gain kMatrix {"-"*13} 
{K}
{"-"*42}
"""
print(Ctrl_Sett_Out)


################################
# Control Law switch (I/O) design of on/off
CTRL_F = 1
CTRL_Non = 0
print("\n")
print(f"| Control Law: Negri Prado (2020,2021)")
CTRL_EXTi = time.time()
sol = solve_ivp(
        fun=EOM_MASCON,
        t_span=[Time[0], Time[-1]],
        y0=a0,
        events=[Collision,Escape],
        method='DOP853',
        first_step=dt_min,
        rtol=1e-10,
        atol=1e-12,
        t_eval=Time,
)

state = sol.y
t = sol.t
# print(sol)
CTRL_EXTf = time.time()
#####################
#####################################
# unpack the acceleration corrections
# i.e. the controls in RTN coordinates 
Accel_Correct = np.array(Accel_Correct)
#####################################
# Error in position from perfect circle
#  to the actual orbit
Err_Rad = np.zeros(state.shape[1], dtype="float64")
for it_E in range(state.shape[1]):
    Radius = np.sqrt(state[0, it_E]**2 + state[1, it_E]**2 + state[2, it_E]**2) 
    Err_Rad[it_E] = (abs(Radius) - y)/y
###########################
###########################
U_Con = np.linalg.norm(Accel_Correct[:, :3], axis=1)
delv_use = np.sum(U_Con)
print('\n')
#################################
# Execution Time for Out, 
# NO other time calcs after THIS!!!
Time_Tag = "sec"
if t[-1] > 120.0:
    t = t/60
    Time_Tag = "min"
    print(f"| Sim Duration Time: {t[-1]} minutes")
elif t[-1] > 3600.0:
    t = t/3600
    Time_Tag = "hr"
    print(f"| Sim Duration Time: {t[-1]} hours")
elif t[-1] > const.day:
    t = t/const.day
    Time_Tag = "day"
    print(f"| Sim Duration Time: {t[-1]} days")
#################################
Sim_Out = f"""
| Controlled Sim execution time: 
|   {CTRL_EXTf - CTRL_EXTi:.2f} sec
{'-'*42}
| Negri Prado Control Law
|
| Total Delta V: {delv_use} km/s"
|
| Total Orbit Time: {t[-1]} {Time_Tag}
|
| Plotting...
"""
print(Sim_Out)
###########################
###########################
# Plot State and Control Input
OBJ_Data = np.loadtxt(OBJ_F, delimiter=' ', dtype=str)
# Extract vertex and face data
vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
# Scale vertices
vertices = vertices * Gamma
# Set faces to start at index 0 instead of 1
faces = faces - 1
# Create mesh
mesh = Poly3DCollection([vertices[ii] for ii in faces], 
                        edgecolor='black',
                        facecolors="white",
                        linewidth=0.5,
                        alpha=0.0)
X = state[0,:]
Y = state[1,:]  
Z = state[2,:]
#######################################################
#######################################################
# Control Input 
figure3D = plt.figure()
ax3D = figure3D.add_subplot(111, projection='3d')
ax3D.add_collection3d(mesh)
ax3D.plot(X, Y, Z, label='Trajectory', color='blue')
# ax3D.plot(X_ESC, Y_ESC, Z_ESC, label='Uncontrolled', color='red')
ax3D.set_title(r'$y_0$:'f'{state[1,0]:.2f}' r'$(km)$'f'  'f'Ham:{Ham:.2e}' r'$(km^2/s^2)$'f'  'f'days:{days}'
               , fontsize=16, fontweight='bold')
ax3D.set_xlabel('X (km)', fontsize=12, fontweight='bold')
ax3D.set_ylabel('Y (km)', fontsize=12, fontweight='bold')
ax3D.set_zlabel('Z (km)', fontsize=12, fontweight='bold')
ax3D.tick_params(axis='x', labelsize=12)
ax3D.tick_params(axis='y', labelsize=12)
ax3D.tick_params(axis='z', labelsize=12)
ax3D.set_aspect('equal', 'box') 

#######################################################
#######################################################
# Calculate Hamiltonian Energy and plot it wiht the velocity
Upsu = np.zeros(state.shape[1], dtype="float64")
for it in range(len(CM)):
    x = state[0,:] - CM[it,0]
    y = state[1,:] - CM[it,1]
    z = state[2,:] - CM[it,2]
    r = np.sqrt(x**2 + y**2 + z**2) 
    Upsu += mu_I[it]/r
############################
X = state[0,:]
Y = state[1,:]
Z = state[2,:]
VX = state[3,:]
VY = state[4,:]
VZ = state[5,:]
V_mag = np.sqrt(VX**2 + VY**2 + VZ**2)
Energy = 0.5*(VX**2 + VY**2 + VZ**2) - 0.5*omega**2* (X**2 + Y**2 + Z**2) - Upsu[0]
H_Max = np.max(Energy)*np.ones(len(Time))
H_Min = np.min(Energy)*np.ones(len(Time))
V_Max = np.max(V_mag)*np.ones(len(Time))
V_Min = np.min(V_mag)*np.ones(len(Time))
##################
# Plot the Hamiltonian and Velocity
fig, (ax1, ax2, ax3,axE) = plt.subplots(4, 1, figsize=(12, 6))
ax1.plot(V_mag)
ax1.plot(V_Max, label='Perigee',color='green') 
ax1.plot(V_Min, label='Apogee', color='r') 
ax1.set_title('Velocity Magnitude', fontsize=16, fontweight='bold')
ax1.set_xlabel(f'Time (s)', fontsize=12, fontweight='bold')
ax1.set_ylabel(r'Velocity $\frac{km}{s}$', fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.legend()
ax2.plot(Energy)
ax2.set_title('Hamiltonian', fontsize=16, fontweight='bold')
ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax2.set_ylabel(r'Energy $\frac{km^2}{s^2}$', fontsize=12, fontweight='bold')
ax2.plot(H_Max, label='Apogee', color='r')
ax2.plot(H_Min, label='Perigee',color='green')
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax2.legend()
# ax2.plot(H_Max) 
# ax2.plot(H_Min) 
ax3.scatter(Accel_Correct[:, 3], U_Con[:], label=f'Control Input', color='blue', s=5, marker='x')
ax3.scatter(Accel_Correct[:, 3],Accel_Correct[:, 0], label=f'Component R', color='green', s=5, marker='|')
ax3.scatter(Accel_Correct[:, 3],Accel_Correct[:, 1], label=f'Component T', color='purple', s=5, marker='_')
ax3.scatter(Accel_Correct[:, 3],Accel_Correct[:, 2], label=f'Component N', color='red', s=5, marker='o')
ax3.set_ylabel(r'Control Input $\frac{km^2}{s^2}$', fontsize=12, fontweight='bold')
ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax3.set_title('Control Input vs. Time', fontsize=16, fontweight='bold')
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=12)
ax3.legend()
#
axE.plot(Err_Rad, label='Radial Error (%)', color='red')
axE.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
axE.set_title('Error in Position', fontsize=16, fontweight='bold')
axE.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
axE.tick_params(axis='x', labelsize=12)
axE.tick_params(axis='y', labelsize=12)
axE.legend()
#
plt.subplots_adjust(right=0.85)  # Make room for legends
plt.tight_layout(rect=[0, 0, 0.85, 1]) 
plt.tight_layout()
################## 
##################
##################
# Show All Plots #
plt.show()
############




# Storage

######
# This system is not controllable in this 
#  method i believe 
#
# from scipy.linalg import eigvals

# # Controllability matrix
# def is_controllable(A, B):
#     n = A.shape[0]
#     controllability_matrix = B
#     for i in range(1, n):
#         controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A, i) @ B))
#     rank = np.linalg.matrix_rank(controllability_matrix)
#     return rank == n

# if not is_controllable(A, B):
#     raise ValueError("The system is not controllable")

# from scipy.signal import place_poles

# # desired_poles = []  
# # result = place_poles(A, B, desired_poles)
# # K = result.gain_matrix