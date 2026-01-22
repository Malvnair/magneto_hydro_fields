import os
import numpy as np
import pandas as pd
from tvtk.api import tvtk
from mayavi import mlab

# =============================
# CONFIG
# =============================
AU_TO_CM = 1.496e13
N = 62
shape = (N, N, N)

file_directory = "/Users/malavikanair/Documents/GitHub/magneto_hydro_fields/"
file_name_vel = "Velocity_filddS_no_bounadaries_s_11.xlsx"

BLACK_BACKGROUND = True
DATA_SCALING_OFF = True
SHOW_ORIENTATION_AXES = True

VEL_VIS_SCALE = 1e-8

# Random split controls
RANDOM_SEED = 7
SPLIT = 2/3          # Group 1 gets ~66.7% of points, Group 2 gets the rest

# Vector density controls (smaller = more arrows)
VEC_MASK_POINTS_1 = 50   # denser group
VEC_MASK_POINTS_2 = 35   # sparser group

# Arrow size
VEC_SCALE_1 = 8.0
VEC_SCALE_2 = 8.0

# =============================
# LOAD DATA
# =============================
df = pd.read_excel(os.path.join(file_directory, file_name_vel))

x = df["X"].to_numpy().reshape(shape)
y = df["Y"].to_numpy().reshape(shape)
z = df["Z"].to_numpy().reshape(shape)
vx = df["V_x"].to_numpy().reshape(shape)
vy = df["V_y"].to_numpy().reshape(shape)
vz = df["V_z"].to_numpy().reshape(shape)

# =============================
# CONVERT TO AU + VISUAL SCALE
# =============================
X_AU = (x / AU_TO_CM).ravel()
Y_AU = (y / AU_TO_CM).ravel()
Z_AU = (z / AU_TO_CM).ravel()

vx_v = (vx * VEL_VIS_SCALE).ravel()
vy_v = (vy * VEL_VIS_SCALE).ravel()
vz_v = (vz * VEL_VIS_SCALE).ravel()

speed = np.sqrt(vx_v**2 + vy_v**2 + vz_v**2)

# Axis ranges (AU)
x_min, x_max = float(X_AU.min()), float(X_AU.max())
y_min, y_max = float(Y_AU.min()), float(Y_AU.max())
z_min, z_max = float(Z_AU.min()), float(Z_AU.max())
full_range_vals = [x_min, x_max, y_min, y_max, z_min, z_max]

# =============================
# RANDOM SPLIT (2/3 vs 1/3)
# =============================
rng = np.random.default_rng(RANDOM_SEED)
group = rng.random(X_AU.size)   # in [0,1)

m1 = group < SPLIT
m2 = ~m1

# =============================
# BUILD POINT CLOUD DATASET (PolyData)
# =============================
pts = np.c_[X_AU, Y_AU, Z_AU]
vecs = np.c_[vx_v, vy_v, vz_v]

poly = tvtk.PolyData(points=pts)
poly.point_data.vectors = vecs
poly.point_data.vectors.name = "velocity"

poly.point_data.scalars = speed
poly.point_data.scalars.name = "speed"

# =============================
# MAYAVI SCENE
# =============================
mlab.close(all=True)

fig = mlab.figure(size=(900, 1200),
                  bgcolor=(0, 0, 0) if BLACK_BACKGROUND else (0.5, 0.5, 0.5))
fig.scene.background = (0, 0, 0) if BLACK_BACKGROUND else (0.5, 0.5, 0.5)
fig.scene.foreground = (1, 1, 1)
fig.scene.parallel_projection = True
fig.scene.show_axes = bool(SHOW_ORIENTATION_AXES)

src = mlab.pipeline.add_dataset(poly)

# =============================
# TWO VECTOR SETS (DISJOINT)
# =============================
def make_vec_layer(mask, mask_points, scale_factor):
    p = tvtk.PolyData(points=np.c_[X_AU[mask], Y_AU[mask], Z_AU[mask]])
    p.point_data.vectors = np.c_[vx_v[mask], vy_v[mask], vz_v[mask]]
    p.point_data.vectors.name = "velocity"
    p.point_data.scalars = speed[mask]
    p.point_data.scalars.name = "speed"

    s = mlab.pipeline.add_dataset(p)

    v = mlab.pipeline.vectors(
        s,
        mask_points=mask_points,
        scale_factor=scale_factor,
        line_width=1.0,
    )

    # Force arrow glyph (guarded)
    try:
        v.glyph.glyph_source.glyph_source = v.glyph.glyph_source.glyph_list[1]
    except Exception:
        pass

    # Scaling behavior
    if DATA_SCALING_OFF:
        v.glyph.glyph.scale_mode = "data_scaling_off"
    else:
        v.glyph.glyph.scale_mode = "scale_by_vector"
    v.glyph.glyph.clamping = True

    # Color by speed
    v.glyph.color_mode = "color_by_scalar"
    v.module_manager.scalar_lut_manager.data_range = (float(speed.min()), float(speed.max()))
    return v

vec1 = make_vec_layer(m1, VEC_MASK_POINTS_1, VEC_SCALE_1)
vec2 = make_vec_layer(m2, VEC_MASK_POINTS_2, VEC_SCALE_2)

# =============================
# AXES / OUTLINE / CAMERA
# =============================
axes = mlab.axes(
    src,
    ranges=full_range_vals,
    xlabel="X (AU)",
    ylabel="Y (AU)",
    zlabel="Z (AU)",
    nb_labels=5,
    color=(1, 1, 1),
)
axes.label_text_property.font_size = 10
axes.title_text_property.font_size = 12

mlab.outline(color=(1, 1, 1))
mlab.view(azimuth=45, elevation=65, distance="auto")

mlab.show()
