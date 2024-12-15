OPENCV2PT3D = {
    "flip_x": True,
    "flip_y": True,
    "flip_z": False,
}

BLENDER2PT3D = {
    "flip_x": True,
    "flip_y": False,
    "flip_z": True,
}


def flip_rotation_axes(rot_mx, flip_x, flip_y, flip_z):
    flipped_mx = rot_mx.clone()

    if flip_x:
        flipped_mx[1:3, :] = -flipped_mx[1:3, :]

    if flip_y:
        flipped_mx[[0, 2], :] = -flipped_mx[[0, 2], :]

    if flip_z:
        flipped_mx[:, [0, 1]] = -flipped_mx[:, [0, 1]]

    return flipped_mx


def flip_translation_vector(trans_vec, flip_x=False, flip_y=False, flip_z=False):
    flipped_vec = trans_vec.clone()

    if flip_x:
        flipped_vec[0] = -flipped_vec[0]

    if flip_y:
        flipped_vec[1] = -flipped_vec[1]

    if flip_z:
        flipped_vec[2] = -flipped_vec[2]

    return flipped_vec
