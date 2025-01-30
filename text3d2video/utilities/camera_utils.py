import math


def focal_length_to_fov(focal_length: float, sensor_width: float, degrees=True):
    fov = 2 * math.atan(sensor_width / (2 * focal_length))
    if degrees:
        fov = math.degrees(fov)
    return fov


def fov_to_focal_length(fov: float, sensor_width: float, degrees=True):
    if degrees:
        fov = math.radians(fov)
    return sensor_width / (2 * math.tan(fov / 2))
