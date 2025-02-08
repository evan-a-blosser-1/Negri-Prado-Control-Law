""" Asteroid Package 

    This handles backend stuff 

"""
import time
import sys
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 

#%% MASCON_SUM
def MASCON_SUM(a, CM, mu_I, Poly_CM):
    points = 1
    Ux = np.zeros(points, dtype="float64")
    Uy = np.zeros(points, dtype="float64")
    Uz = np.zeros(points, dtype="float64")
    for it in range(len(CM)):
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
    return Ux, Uy, Uz
            

def OBJ_Read(OBJ_File):
    ##############################################################
    OBJ_Data = np.loadtxt(OBJ_File, delimiter=' ', dtype=str)
    
    # Extract vertex and face data
    vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
    faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
    
    return vertices, faces

def OBJ_2_volInt(OBJ_File):
    """Reads an OBJ file and converts the vertex and face data into arrays.
    
    Args:
        OBJ_File (file): .obj file
    
    Returns:
        tuple: Two numpy arrays containing vertex and face data
    """
    # Read the OBJ file data
    OBJ_Data = np.loadtxt(OBJ_File, delimiter=' ', dtype=str)
    
    # Extract vertex data and prepend an empty row of zeros
    vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
    vertices = np.vstack(([0, 0, 0], vertices))
    
    # Extract face data and prepend a 3 to each row
    faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
    faces = np.hstack((np.full((faces.shape[0], 1), 3), faces))
    
    return vertices, faces

#%% Plotting

def Plot_State(state,OBJ_File,gamma,Mesh_color,M_line=0.5,M_alpha=0.05):
    ##############################################################
    OBJ_Data = np.loadtxt(OBJ_File, delimiter=' ', dtype=str)
    
    # Extract vertex and face data
    vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
    faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
    # Scale vertices
    vertices = vertices * gamma
    # Set faces to start at index 0 instead of 1
    faces = faces - 1
    # Create mesh
    mesh = Poly3DCollection([vertices[ii] for ii in faces], 
                            edgecolor=Mesh_color,
                            facecolors="white",
                            linewidth=M_line,
                            alpha=M_alpha)
    X = state[0,:]
    Y = state[1,:]  
    Z = state[2,:]
    #######################################################
    #######################################################
    figure3D = plt.figure()
    ax = figure3D.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)
    ax.plot(X, Y, Z, label='Trajectory')
    ax.set_title(f'y0:{state[1,0]:.2f} x_dot:{state[3,0]:.2e}')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_aspect('equal', 'box') 
    return 


def Plot_Ham(state,omega,mu_I,CM):
    U = np.zeros(state.shape[1], dtype="float64")
    for it in range(len(CM)):
        x = state[0,:] - CM[it,0]
        y = state[1,:] - CM[it,1]
        z = state[2,:] - CM[it,2]
        r = np.sqrt(x**2 + y**2 + z**2) 
        U += mu_I[it]/r
    ############################
    X = state[0,:]
    Y = state[1,:]
    Z = state[2,:]
    VX = state[3,:]
    VY = state[4,:]
    VZ = state[5,:]
    V_mag = np.sqrt(VX**2 + VY**2 + VZ**2)
    Energy = 0.5*(VX**2 + VY**2 + VZ**2) - 0.5*omega**2* (X**2 + Y**2 + Z**2) - U[0]
    ref_Max = np.max(Energy)*np.ones(len(Energy))
    ref_Min = np.min(Energy)*np.ones(len(Energy))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(V_mag)
    ax1.set_title('Velocity Magnitude')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'Velocity $\frac{km}{s}$')
    ax2.plot(Energy)
    ax2.plot(ref_Max, 'r-', label='Perigee') 
    ax2.plot(ref_Min, 'r-', label='Apogee') 
    ax2.set_title('Hamiltonian')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'Energy $\frac{km^2}{s^2}$')
    plt.tight_layout()
    plt.legend()
    return


def ani_3D_Comp(data1, data2,OBJ_File,gamma, labels=['Dataset 1', 'Dataset 2'],
                    Mesh_color='black',M_line=0.5,M_alpha=0.05,F_T=1000):
    """_summary_

    Args:
        data1 (_type_): _description_
        data2 (_type_): _description_
        labels (list, optional): _description_. Defaults to ['Dataset 1', 'Dataset 2'].

    Returns:
        _type_: _description_
        
        
    Example: AP.animate_3d_data(pos_data[:, 3:], target_data[:, 3:], ['Actual', 'Target'])
    """
    ##############################################################
    OBJ_Data = np.loadtxt(OBJ_File, delimiter=' ', dtype=str)
    
    # Extract vertex and face data
    vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
    faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
    # Scale vertices
    vertices = vertices * gamma
    # Set faces to start at index 0 instead of 1
    faces = faces - 1
    # Create mesh
    mesh = Poly3DCollection([vertices[ii] for ii in faces], 
                            edgecolor=Mesh_color,
                            facecolors="white",
                            linewidth=M_line,
                            alpha=M_alpha)
    
    # Setup the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)
    # Initialize lines with dotted style and markers
    line1, = ax.plot([], [], [], 'b:', label=labels[0], markevery=[-1], marker='o', markersize=8)
    line2, = ax.plot([], [], [], 'r:', label=labels[1], markevery=[-1], marker='o', markersize=8)
    # Calculate bounds with 20% padding
    max_range = np.max([
        np.max(data1[0]) - np.min(data1[0]),
        np.max(data1[1]) - np.min(data1[1]),
        np.max(data1[2]) - np.min(data1[2])
    ])

    # Find center point
    mid_x = (np.max(data1[0]) + np.min(data1[0])) * 0.5
    mid_y = (np.max(data1[1]) + np.min(data1[1])) * 0.5
    mid_z = (np.max(data1[2]) + np.min(data1[2])) * 0.5
    
    # Set equal limits with padding
    padding = max_range * 0.6
    ax.set_xlim(mid_x - padding, mid_x + padding)
    ax.set_ylim(mid_y - padding, mid_y + padding)
    ax.set_zlim(mid_z - padding, mid_z + padding)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal', 'box')
    ###################
    # Space Theme it for the kiddos 
    ax.grid(False)
    ax.set_axis_off()
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black') 
    # Add frame skipping - only animate every nth frame
    frame_skip = 2  # Adjust this value to skip more frames
    
    # Create frames array with skipped indices
    frames = range(0, len(data1[0]), frame_skip)

    def update(frame):
        # Update data for both lines
        line1.set_data(data1[0][:frame], data1[1][:frame])
        line1.set_3d_properties(data1[2][:frame])
        
        line2.set_data(data2[0][:frame], data2[1][:frame])
        line2.set_3d_properties(data2[2][:frame])
        # Ensure markers stay at end points
        line1.set_markevery([-1])  # Show marker only at latest point
        line2.set_markevery([-1])
            
        # ax.view_init(elev=30, azim=frame % 360)
        return line1, line2
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=frames, 
        interval=F_T,
        blit=True,               # Enable blitting
        cache_frame_data=False  # Disable frame caching
    )
    
    
    plt.legend()
    plt.show()





def ani_3D_data(data1, OBJ_File, gamma, labels=['Orbit'],
                Mesh_color='cyan', M_line=0.5, M_alpha=0.05, F_T=0.1):
    """Animate 3D trajectory data with asteroid mesh.

    Args:
        data1: Trajectory data array (3xN)
        OBJ_File: Path to asteroid obj file
        gamma: Scaling factor for asteroid mesh
        labels: Label for trajectory. Defaults to ['Dataset 1']
        Mesh_color: Color of mesh edges. Defaults to 'black'
        M_line: Mesh line width. Defaults to 0.5
        M_alpha: Mesh transparency. Defaults to 0.05
        F_T: Animation frame time in ms. Defaults to 1000

    Example: 
        AP.ani_3D_data(pos_data[:, 3:], 'asteroid.obj', 1.0, ['Trajectory'])
    """
    # Load and process mesh
    OBJ_Data = np.loadtxt(OBJ_File, delimiter=' ', dtype=str)
    vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
    faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
    vertices = vertices * gamma
    faces = faces - 1
    mesh = Poly3DCollection([vertices[ii] for ii in faces], 
                          edgecolor=Mesh_color,
                          facecolors="white",
                          linewidth=M_line,
                          alpha=M_alpha)
    
    # Setup figure and axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)
    line1, = ax.plot([], [], [], 'b:', label=labels[0], markevery=[-1], marker='o', markersize=2)

    # Calculate view bounds
    max_range = np.max([
        np.max(data1[0]) - np.min(data1[0]),
        np.max(data1[1]) - np.min(data1[1]),
        np.max(data1[2]) - np.min(data1[2])
    ])
    mid_x = (np.max(data1[0]) + np.min(data1[0])) * 0.5
    mid_y = (np.max(data1[1]) + np.min(data1[1])) * 0.5
    mid_z = (np.max(data1[2]) + np.min(data1[2])) * 0.5
    
    # Set view limits
    padding = max_range * 0.6
    ax.set_xlim(mid_x - padding, mid_x + padding)
    ax.set_ylim(mid_y - padding, mid_y + padding)
    ax.set_zlim(mid_z - padding, mid_z + padding)
    
    # Style settings
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    ax.set_axis_off()
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Animation settings
    frame_skip = 2
    frames = range(0, len(data1[0]), frame_skip)

    def update(frame):
        line1.set_data(data1[0][:frame], data1[1][:frame])
        line1.set_3d_properties(data1[2][:frame])
        line1.set_markevery([-1])
        return (line1,)
    
    anim = animation.FuncAnimation(
        fig, update, frames=frames,
        interval=F_T, blit=True,
        cache_frame_data=False
    )
    
    plt.legend()
    plt.show()


def ani_3D_NOrbs(OBJ_File, gamma, *trajectories, labels=None, 
                Mesh_color='cyan', M_line=0.5, M_alpha=0.05, F_T=1000):
    """Animate multiple 3D trajectories with asteroid mesh.
    Expects trajectories in Nx3 format (no transpose needed).
    """
    if labels is None:
        labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))
    
    # Mesh setup remains unchanged
    OBJ_Data = np.loadtxt(OBJ_File, delimiter=' ', dtype=str)
    vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
    faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
    vertices = vertices * gamma
    faces = faces - 1
    mesh = Poly3DCollection([vertices[ii] for ii in faces], 
                          edgecolor=Mesh_color,
                          facecolors="white",
                          linewidth=M_line,
                          alpha=M_alpha)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)
    
    lines = []
    for i, color in enumerate(colors):
        line, = ax.plot([], [], [], ':', label=labels[i], 
                       markevery=[-1], marker='o', markersize=8,
                       color=color)
        lines.append(line)
    
    # Calculate bounds using first trajectory (now Nx3)
    first_traj = trajectories[0]
    max_range = np.max([
        np.max(first_traj[:, 0]) - np.min(first_traj[:, 0]),
        np.max(first_traj[:, 1]) - np.min(first_traj[:, 1]),
        np.max(first_traj[:, 2]) - np.min(first_traj[:, 2])
    ])
    mid_x = (np.max(first_traj[:, 0]) + np.min(first_traj[:, 0])) * 0.5
    mid_y = (np.max(first_traj[:, 1]) + np.min(first_traj[:, 1])) * 0.5
    mid_z = (np.max(first_traj[:, 2]) + np.min(first_traj[:, 2])) * 0.5
    
    padding = max_range * 0.6
    ax.set_xlim(mid_x - padding, mid_x + padding)
    ax.set_ylim(mid_y - padding, mid_y + padding)
    ax.set_zlim(mid_z - padding, mid_z + padding)
    
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    ax.set_axis_off()
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    frame_skip = 2
    max_frames = max(len(traj) for traj in trajectories)
    frames = range(0, max_frames, frame_skip)

    def update(frame):
        for line, traj in zip(lines, trajectories):
            if frame < len(traj):
                line.set_data(traj[:frame, 0], traj[:frame, 1])
                line.set_3d_properties(traj[:frame, 2])
                line.set_markevery([-1])
        return tuple(lines)
    
    anim = animation.FuncAnimation(
        fig, update, frames=frames,
        interval=F_T, blit=True,
        cache_frame_data=False
    )
    
    plt.legend()
    plt.show()

#%% Tetra_Volume 


def Tetra_Volume(Verts,Faces,MASCON_Div,total_vol=0):
    """Tetrahedron Volume Calculation 
            - for polyhedron shape models. 
    Args:
        Verts (array): Polyhedron Vertices
        Faces (array): Polyhedron Face Data
        MASCON_Div (fracton): Divides the tetrahedron, set = 1 for total
        total_vol (bool): 0/1 for total volume calculations, or  just tetra calculations
    """
    tetra_count=np.shape(Faces)[0]
    ###################    
    # Make list
    Volume_Tetra_Array  = np.zeros(tetra_count, dtype="float64")

    Tetra_U = np.zeros((tetra_count, 3), dtype="float64")
    Tetra_V = np.zeros((tetra_count, 3), dtype="float64")
    Tetra_W = np.zeros((tetra_count, 3), dtype="float64")
    ############################### Analytical Method:
    for it in range(0,tetra_count):
    ##### Center of mass to vertex vectors
        U_vec = np.array([Verts[Faces[it,0],0]*MASCON_Div,
                          Verts[Faces[it,0],1]*MASCON_Div,
                          Verts[Faces[it,0],2]*MASCON_Div
                        ]) 
        V_vec = np.array([Verts[Faces[it,1],0]*MASCON_Div,
                          Verts[Faces[it,1],1]*MASCON_Div,
                          Verts[Faces[it,1],2]*MASCON_Div
                        ]) 
        W_vec = np.array([Verts[Faces[it,2],0]*MASCON_Div,
                          Verts[Faces[it,2],1]*MASCON_Div,
                          Verts[Faces[it,2],2]*MASCON_Div
                        ])
        Tetra_U[it] = U_vec
        Tetra_V[it] = V_vec 
        Tetra_W[it] = W_vec 
        ######################################################
        ############### Triple Scalar Product ################
        Vol_Full_Sum =  U_vec[0]*(V_vec[1]*W_vec[2] - V_vec[2]*W_vec[1]) -\
                        U_vec[1]*(V_vec[0]*W_vec[2] - V_vec[2]*W_vec[0]) +\
                        U_vec[2]*(V_vec[0]*W_vec[1] - V_vec[1]*W_vec[0]) 
        #
        Vol_tetra_full = (1/6) * abs((Vol_Full_Sum))
        Volume_Tetra_Array[it] = Vol_tetra_full
    ###############################################
    ######### Sum Tetra Volumes for Check #########
    Total_Volume_Out = np.sum(Volume_Tetra_Array)
    #######################################################
    ######### Output ######################################
    if total_vol == 0:
        Volume_Tetra_Array_Message = f"""
{'-'*42}
|
|  Total Volume Calculated as:
|    V = {Total_Volume_Out}
|
{'-'*42}
"""
        # print(Volume_Tetra_Array_Message )
        return Volume_Tetra_Array,Total_Volume_Out
    elif total_vol == 1:
        return Volume_Tetra_Array



#%% solve_ivp progress bar 



def print_progress(current_time, total_time, bar_length=40):
    fraction = current_time / total_time
    block = int(round(bar_length * fraction))
    bar = '\u25A1' * block + '-' * (bar_length - block)
    # Added current_time in seconds after the percentage
    sys.stdout.write(f'\r[{bar}] {fraction:.2%} (Int Time: {current_time:.2e}s)')
    sys.stdout.flush()



#%% Rotations 


def rtn_rot(a, vector,choice="RTN"):
    """
    Translates a vector from RTN coordinates back to Cartesian coordinates.
    Parameters:
    a (array-like): The state vector [x, y, z, vx, vy, vz].
    rtn_vector (array-like): The vector in RTN coordinates to be translated.
    Returns:
    cartesian_vector (array-like): The translated vector in Cartesian coordinates.
    """
    r = np.array([a[0], a[1], a[2]])
    r_dot = np.array([a[3], a[4], a[5]])
    
    # Normalize the position vector to get the radial direction
    r_hat = r / np.linalg.norm(r)
    
    # Compute the transverse direction
    h = np.cross(r, r_dot)
    h_hat = h / np.linalg.norm(h)
    t_hat = np.cross(h_hat, r_hat)
    
    # Construct the rotation matrix
    rotation_matrix = np.vstack((r_hat, t_hat, h_hat)).T
    
    if choice == "CART":
        # Apply the inverse rotation matrix to the RTN vector
        cartesian_vector = rotation_matrix.T @ vector
        
        return cartesian_vector
    
    elif choice == "RTN":
        # Apply the rotation matrix to the Cartesian vector
        rtn_vector = rotation_matrix @ vector
        return rtn_vector
#%% Velocity from hamiltonian
################################################################################
########################################### Velocity from hamiltonian 
def v_calc(Ham,omega,mu_I,CM,P,Loc='y'):
    if Loc == 'y':
        Px = 0.0
        Py = P
        Pz = 0.0
    elif Loc == 'x':
        Px = P
        Py = 0.0
        Pz = 0.0
    elif Loc == 'z':
        Px = 0.0
        Py = 0.0
        Pz = P
    U = np.zeros(1, dtype="float64")
    for it in range(len(CM)):
        x = Px  - CM[it,0]
        y = Py  - CM[it,1]
        z = Pz  - CM[it,2]
        r = np.sqrt(x**2 + y**2 + z**2) 
        U += mu_I[it]/r
    #########################
    psu = U[0]
    cori = (omega**2)*(x**2 + y**2)
    print(f"Omega:    {omega} rad/s")
    print(f"Ham:      {Ham} (km^2/s^2) ")
    print(f"Psuedo:   {psu} (km^2/s^2) ")
    print(f"Coriolis: {cori} (km^2/s^2) ")
    arg =  2*Ham + cori + 2*psu
    if arg > 0:
        V = np.sqrt(arg)
    return V


#%% Solar Radiation Pressure
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Function to calculate the acceleration due to the direct solar radiation
# pressure (SRP) whith shadow effect.
#
# Author:  Diogo Merguizo Sanchez
#          The University of Oklhoma
#          dmsanchez@ou.edu
#          2023
# 
# Input data: - satellite's position vector (km)
#             - Sun's position vector (km)
#             - satellite's area-to-mass ratio (m^2/kg)
#             - satellite's reflectivity coefficient (1.0 <= ref_co <= 2.0)
#             - Reflectivity coefficient (n.d.)
#             - Central body's equatorial or effecive radius
# Output data: - SRP acceleration vector (km/s^2)
#
# References: Beutler, 2004; Alsemo & Pardini, 2007; Delsate & Compère, 2012
#             Montenbruck & Gill, 2001

def SRPacc(xsat, xsun, AtoM, Cr, Re):

  PI = np.pi

  AU = 149597870.700 # km (From the 2023 Astronomical Almanac)
  Psun  = 4.56316e-6 # N/m^2 # Solar radiation pressure constant (at 1 au)
  reqsun = 695700.0 # km # Sun's volumetric mean radius - JPL Horizons rev. 2013

  SRP0 = Psun * AtoM * Cr * 1e-3 # km/s^2 (be sure to have the right units)

  Rsat = np.sqrt(xsat[0]**2 + xsat[1]**2 + xsat[2]**2)

  # Relative position spacecraft ---> Sun
  xsatsun = xsun - xsat
  Rsatsun = np.sqrt(xsatsun[0]**2 + xsatsun[1]**2 + xsatsun[2]**2)

  # Shadow effect coefficient calculation (Montenbruck & Gill, 2001)

  # Apparent radius if the Sun
  a = np.arcsin(reqsun/Rsatsun)

  # Apparent radius of the planet
  b = np.arcsin(Re/Rsat)

  # Apparent separation of the centers of both bodies
  dotprod = - np.dot(xsat, xsatsun)
  c = np.arccos(dotprod/(Rsat*Rsatsun))

  if (a + b) <= c:
    nu = 1.0
  elif (c < (a - b)) or (c < (b - a)):
    nu = 0.0
  else:
    x = (c**2 + a**2 - b**2)/(2.0*c)
    y = np.sqrt(a**2 - x**2)
    # Occulted area
    Area = a**2 * np.arccos(x/a) + b**2 * np.arccos((c - x)/b) - c*y

    nu = 1.0 - Area/(PI*a**2)
  ###########
  # OVerride for error 
  # around asteorid
  if np.isnan(nu):
    nu = 0.0
  # Acceleration due to the SRP

  factor = (AU/Rsatsun)**2

  if np.isnan(nu):
    print("PROBLEM in SRP: nu is NaN.")
    exit()
  
  SRP = np.zeros(3)

  SRP =  - nu * SRP0 * factor * xsatsun/Rsatsun

  return SRP

#%%



def rotation_matrix(angle, axis='z'):
    """Create 3D rotation matrix"""
    c = np.cos(angle)
    s = np.sin(angle)
    if axis.lower() == 'z':
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
    elif axis.lower() == 'y':
        return np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
    elif axis.lower() == 'x':
        return np.array([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
        
        
        
        