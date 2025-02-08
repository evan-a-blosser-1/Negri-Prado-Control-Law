
import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import sys
import trimesh
########################################################
#################################### Personal Packages #    
sys.dont_write_bytecode = True
import ASTER_Pack as AP
import constants as C
#################################################
####### Enter Asteroid Name & Update Here #######
Asteroid_Name = '1950DA_Prograde'
##################################################
########### Load File ############################
Aster_File_CM   = Asteroid_Name + "_CM.in"
Aster_File_OBJ  = Asteroid_Name + ".obj"
##################################################
Aster_CM  = np.loadtxt(Aster_File_CM, delimiter=' ')
mu_I = np.loadtxt(Asteroid_Name + '_mu.in', delimiter=' ')
##################################################
print(f'| Asteroid File: {Aster_File_OBJ} |')
vertices, faces = AP.OBJ_Read(Aster_File_OBJ)
target = C.DA1950()
vertices = vertices * target.gamma  # km 
print(f'| Vertices: {vertices} |')
faces = faces - 1
print(f'| Faces: {faces} |')
########################################
mesh = trimesh.load_mesh(Aster_File_OBJ)
# Scale the mesh
mesh.apply_scale(target.gamma)
# Calculate the volume of the polyhedron
polyhedron_volume = mesh.volume
R_eff = (3 * polyhedron_volume / (4 * np.pi)) ** (1/3)
print(f"Volume: {polyhedron_volume} km^3")
print(f"Effective Radius: {R_eff} km")
############################
mesh_plt = Poly3DCollection([vertices[ii] for ii in faces], 
                        edgecolor='black',
                        facecolors="white",
                        linewidth=0.5,
                        alpha=0.0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Aster_CM[:,0], Aster_CM[:,1], Aster_CM[:,2], 'o',color='red')
ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], 'o')
ax.add_collection3d(mesh_plt)
ax.set_title(f'{Asteroid_Name} - MASCON Model' , fontsize=16, fontweight='bold')
ax.set_xlabel('X (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (km)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (km)', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='z', labelsize=12)
plt.show()
########################################
