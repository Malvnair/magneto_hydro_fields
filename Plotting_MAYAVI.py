## In this file, the vector functions will be plotted in 3D
# Importing the libraries and modules required
from mayavi import mlab
import pandas as pd
import numpy as np
from pathlib import Path

# The initial conditions:

AU_TO_CM = 1.496e13 # This is the factor to transfer AU to cm
Resolution = 11 # The resolution of the file.

# Loading the files:
BASE_DIR = Path(__file__).resolve().parent   # folder where the script lives
### The files
file_name_V = "Cartesian_ALL_and_net_forces_cgs_no_boundaries_L_"+ str(Resolution)+".xlsx" # Net force file
# file_name_V = "Gravitational_force_cgs_no_boundaries_L_11" # Net force file
# file_name_V = 'New_Cartesian_Lorentz_forces_cgs_no_boundaries_L_'+str(Resolution)+'.xlsx' # Lorentz forces files
# file_name_V = 'New_Cartesian_Pressure_gradient_cgs_no_boundaries_L_' + str(11) + '.xlsx' # Thermal Pressure gradient file
# file_name_V = 'Velocity_filddS_no_bounadaries_s_' + str(11) + '.xlsx' # Velocity filed file
# file_name_V_vel = 'New_Cartesian_rho_vel_cgs_no_boundaries_L_'+str(11)+'.xlsx' # Advective acceleration term file
file_name_V_vel = 'Magnetic_Fields_no_boundaries_s'+str(Resolution)+'.xlsx' # Magnetic field file

# Reshaping the numpy arrays into a grid
N = 62  # The size of the grid cell
shape = (N, N, N)

# reading the path
file_path = BASE_DIR / file_name_V
V_fielddd = pd.read_excel(file_path)

## Reshaping the arrays
# The position arrays
x = V_fielddd['X'].to_numpy().reshape(shape)
y = V_fielddd['Y'].to_numpy().reshape(shape)
z = V_fielddd['Z'].to_numpy().reshape(shape)

# The vector field arrays (This should correspond to the file chosen. For example, if the force file is chosen, then ['F_net_x'],
# ['F_net_y'], ..., if the magnetic file is chosen, then ['B_x'], ['B_y'],... and so on
# This is going to be in MAYAVI window 1
vx = V_fielddd['F_net_x'].to_numpy().reshape(shape)
vy = V_fielddd['F_net_y'].to_numpy().reshape(shape)
vz = V_fielddd['F_net_z'].to_numpy().reshape(shape)

## Reading the other file
file_path_vel = BASE_DIR / file_name_V_vel
V_field_vel = pd.read_excel(file_path_vel)

# Extracting and reshaping the arrays
# This is going to be in MAYAVI window 2
vx2 = V_field_vel['B_x'].to_numpy().reshape(shape)
vy2 = V_field_vel['B_y'].to_numpy().reshape(shape)
vz2 = V_field_vel['B_z'].to_numpy().reshape(shape)

# --- Full-domain ranges in AU (for consistent axes on all figs) ---
x_min_AU, x_max_AU = x.min()/AU_TO_CM, x.max()/AU_TO_CM
y_min_AU, y_max_AU = y.min()/AU_TO_CM, y.max()/AU_TO_CM
z_min_AU, z_max_AU = z.min()/AU_TO_CM, z.max()/AU_TO_CM
full_range_vals = [x_min_AU, x_max_AU, y_min_AU, y_max_AU, z_min_AU, z_max_AU]

# the space between the vectors
step = 3 # The higher the number, the fewer vectors plotted (the more space between the vectors).
x_d = x[::step, ::step, ::step]
y_d = y[::step, ::step, ::step]
z_d = z[::step, ::step, ::step]
vx_d = vx[::step, ::step, ::step]
vy_d = vy[::step, ::step, ::step]
vz_d = vz[::step, ::step, ::step]
vx2_d = vx2[::step, ::step, ::step]
vy2_d = vy2[::step, ::step, ::step]
vz2_d = vz2[::step, ::step, ::step]

## Flatten the arrays
x_flat = x_d.flatten()
y_flat = y_d.flatten()
z_flat = z_d.flatten()
vx_flat = vx_d.flatten()
vy_flat = vy_d.flatten()
vz_flat = vz_d.flatten()
vx2_flat = vx2_d.flatten()
vy2_flat = vy2_d.flatten()
vz2_flat = vz2_d.flatten()

# Transforming the cm units into AU
x_flat /= AU_TO_CM
y_flat /= AU_TO_CM
z_flat /= AU_TO_CM

# === Mask on flattened arrays
out = 200 # The maximum x and y position values that the vectors will reach
inn = 0 # The minimum x and y position values that the vectors will reach
z_max = 150 # The maximum z position value that the vectors will reach
z_min = -150 # The minimum z position value that the vectors will reach
mask = (
    (x_flat > out*-1) & (x_flat < out) &
    (y_flat > out*-1) & (y_flat < out) &
    (z_flat > z_min) & (z_flat < z_max) &
    ~((x_flat > inn*-1) & (x_flat < inn) & (y_flat > inn*-1) & (y_flat < inn))
)

# Applying the mask/threshold
x_d = x_flat[mask]
y_d = y_flat[mask]
z_d = z_flat[mask]
vx_d = vx_flat[mask]
vy_d = vy_flat[mask]
vz_d = vz_flat[mask]
vx2_d = vx2_flat[mask]
vy2_d = vy2_flat[mask]
vz2_d = vz2_flat[mask]

# Normalizing vectors
v_mag = np.sqrt(vx_d**2 + vy_d**2 + vz_d**2) # Normalizing the vectors of the first file
v_mag[v_mag == 0] = 1.0
vx_d /= v_mag
vy_d /= v_mag
vz_d /= v_mag

v_mag2 = np.sqrt(vx2_d**2 + vy2_d**2 + vz2_d**2) # Normalizing the vectors of the second file
v_mag2[v_mag2 == 0] = 1.0
vx2_d /= v_mag2
vy2_d /= v_mag2
vz2_d /= v_mag2

# Plotting the 3D vectors in two different MAYAVI windows
scale = 7
range_vals = [x_d.min(), x_d.max(), y_d.min(), y_d.max(), z_d.min(), z_d.max()]

# === First figure: Force or Magnetic Field ===
fig1 = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 800))
mlab.figure(fig1)  # Explicitly set focus
quiver1 = mlab.quiver3d(x_d, y_d, z_d, vx_d, vy_d, vz_d,
                        scale_factor=scale, line_width=1.0,
                        opacity=0.7, color=(0.2, 0.3, 0.9), mode='arrow')

axes1 = mlab.axes(quiver1,
                  xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]',
                  nb_labels=5, ranges=range_vals,
                  color=(0, 0, 0))
axes1.label_text_property.font_size = 8
axes1.label_text_property.color = (0, 0, 0)
axes1.title_text_property.color = (0, 0, 0)

mlab.outline()
mlab.view(azimuth=45, elevation=60, distance='auto')

# === Second figure: Velocity Field ===
fig2 = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 800))
mlab.figure(fig2)  # Switch context to second figure
quiver2 = mlab.quiver3d(x_d, y_d, z_d, vx2_d, vy2_d, vz2_d,
                        scale_factor=scale, line_width=1.0,
                        opacity=0.7, color=(0.9, 0.2, 0.2), mode='arrow')

axes2 = mlab.axes(quiver2,
                  xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]',
                  nb_labels=5, ranges=range_vals,
                  color=(0, 0, 0))
axes2.label_text_property.font_size = 8
axes2.label_text_property.color = (0, 0, 0)
axes2.title_text_property.color = (0, 0, 0)

mlab.outline()
mlab.view(azimuth=45, elevation=60, distance='auto')
fig1.scene.background = (1, 1, 1)  # white background
fig2.scene.background = (1, 1, 1)

# Plotting the two vector fields in the same window (overlaying vector fields)
mlab.figure(bgcolor=(1, 1, 1), size=(1000, 800))
scale = 7
# Magnetic field (blue)
quiver = mlab.quiver3d(  # assign this one to 'quiver'
    x_d , y_d , z_d ,
    vx_d, vy_d, vz_d,
    scale_factor=scale,
    line_width=1.0,
    opacity=0.7,
    color=(0.2, 0.3, 0.9),
    mode='arrow'
)

# Velocity field (red)
mlab.quiver3d(
    x_d, y_d, z_d,
    vx2_d, vy2_d, vz2_d,
    scale_factor=scale,
    line_width=1.0,
    opacity=0.7,
    color=(0.9, 0.2, 0.2),  # red
    mode='arrow'
)

# Axes based on 'quiver' (can be either one)
axes = mlab.axes(quiver,
                 xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]',
                 nb_labels=5,
                 ranges=full_range_vals,   # use FULL domain for consistent box
                 color=(0, 0, 0))
axes.label_text_property.font_size = 8     # Reduce axis number size
axes.label_text_property.color = (0, 0, 0)
axes.title_text_property.color = (0, 0, 0)

mlab.outline()
mlab.view(azimuth=45, elevation=60, distance='auto')
mlab.roll()

# === Fourth figure: 2D-style streamlines on a plane (clean flow view)
#     (seed plane spans the whole XY box at z=0; no random glyph dots)
fig4 = mlab.figure(bgcolor=(1, 1, 1), size=(1200, 900))

# Create a vector field source from the RAW magnetic field (no normalization)
src = mlab.pipeline.vector_field(vx2, vy2, vz2)
mag = mlab.pipeline.extract_vector_norm(src)  # enables coloring by |B|

# Streamlines seeded from a plane spanning the FULL domain at z = 0
flow = mlab.pipeline.streamline(
    mag,
    seedtype='plane',
    seed_visible=False,
    seed_resolution=32,          # higher = denser coverage
    integration_direction='both',
    linetype='line'              # clean 2D-style lines
)

# Configure the seed plane to cover the whole XY box at z=0 (AU space)
sp = flow.seed.widget
sp.normal_to_z_axis = True
cx = 0.5*(x_min_AU + x_max_AU)  # center in X
cy = 0.5*(y_min_AU + y_max_AU)  # center in Y
sp.origin = (cx, cy, 0.0)
sp.point1 = (x_max_AU, cy, 0.0)   # +X edge
sp.point2 = (cx, y_max_AU, 0.0)   # +Y edge

# Integrator settings: allow streamlines to traverse the full domain
st = flow.stream_tracer
st.initial_integration_step = 0.5       # tune 0.2–1.0 for smoothness
st.maximum_propagation = 2000.0         # large so they fill the box
st.integrator_type = 'runge_kutta4'     # RK4 integration

# Axes & camera: full ranges, top-down, and parallel projection for 2D look
ax4 = mlab.axes(flow,
                xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]',
                nb_labels=5, ranges=full_range_vals,
                color=(0, 0, 0))
ax4.label_text_property.font_size = 8
ax4.label_text_property.color = (0, 0, 0)
ax4.title_text_property.color = (0, 0, 0)

mlab.outline()                    # full-domain outline
fig4.scene.z_plus_view()          # top-down
fig4.scene.parallel_projection = True
mlab.view(distance='auto')

# Showing both
mlab.show()
