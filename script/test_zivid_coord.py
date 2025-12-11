import zivid
import os
import numpy as np

print("Initializing Zivid...")
app = zivid.Application()
camera = app.connect_camera()

settings_path = "./zivid_settings.yml"
if not os.path.exists(settings_path):
    print(f"Warning: {settings_path} not found. Using default settings.")
    settings = zivid.Settings()
    settings.acquisitions.append(zivid.Settings.Acquisition())
else:
    settings = zivid.Settings.load(settings_path)

print("Capturing frame...")
with camera.capture_2d_3d(settings) as frame:
    point_cloud = frame.point_cloud()

    # Get XYZ data
    points = point_cloud.copy_data("xyz").reshape((-1, 3))

    # Filter NaNs
    flag = np.logical_not(np.any(np.isnan(points), axis=1))
    points = points[flag, :]

    # Find out the range of X, Y, Z
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    print(f"Point Cloud Range:")
    print(f"X: {x_min:.3f} to {x_max:.3f} m")
    print(f"Y: {y_min:.3f} to {y_max:.3f} m")
    print(f"Z: {z_min:.3f} to {z_max:.3f} m")