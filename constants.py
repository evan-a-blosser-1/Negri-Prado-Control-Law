# This file contains parameters and constants used in the simulation.
class constants:
  def __init__(self):
    self.au = 149597870.7 # km (Astronomical Almanac 2024)
    self.G = 6.67428e-20 # km^3/kg/s^2 (Astronomical Almanac 2024)
    self.SRP0 = 4.56e-6 # N/m^2 Solar radiation pressure at 1 au
    self.day = 86400 # seconds

const = constants()
G = const.G # km^3/kg/s^2

# Asteroid physical parameters
class apophis:
  def __init__(self):
    self.mass = 5.31e10               # kg (Diogo 2021)
    self.GM = G*self.mass
    self.Re = 0.1935                  # km (volumetric mean radius - Diogo 2021)
    self.spin =  0.05305800926062762  # rad/s (30.4h per rotation Diogo 2021)
    self.Poly_x = -0.002150           # volInt.c output
    self.Poly_y = -0.001070           # Polyhedron Center of Mass (CM)
    self.Poly_z = -0.000308           # -x,y,z (km)
    self.gamma  = 0.2848196900000026  # Asteroid Scale to Km
    
    
    
class DA1950:
  def __init__(self):
    self.mass = 1.0e10                # kg
    self.GM = G*self.mass
    self.Re = 0.58                    # km (volumetric mean radius)
    self.spin =  0.0037028905410311694  # rad/s (2.1216 rot/hr)
    self.Poly_x = 0.016325            # volInt.c output
    self.Poly_y = 0.026771            # Polyhedron Center of Mass (CM)
    self.Poly_z = 0.016897            # -x,y,z (km)
    self.gamma  = 0.9913709055500112 # Asteroid Scale to Km
    
      