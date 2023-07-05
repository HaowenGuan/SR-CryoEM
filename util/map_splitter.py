import numpy as np
import math


def split_map(data: np.ndarray, box_size=64, core_size=50, dtype=np.float64):
    """
    Creates a list of 'box_size'^3 matrices from the density map data, which
    can be fed to the U-Net independently

    When the output image is reconstructed only the middle('core_size'^3) region
    in the image is used. This helps eliminate the issue of boundary prediction
    issues.

    :param density_map: Full input density map
    :param box_size: Size of sub maps
    :param core_size: Size of core that is considered after prediction
    :param dtype: Data type of the output arrays
    :return: List of 'box_size'^3 sub maps
    """
    image_shape = np.shape(data)
    padded_image = np.zeros((image_shape[0] + 2 * box_size,
                             image_shape[1] + 2 * box_size,
                             image_shape[2] + 2 * box_size), dtype=dtype)
    padded_image[
        box_size:box_size + image_shape[0],
        box_size:box_size + image_shape[1],
        box_size:box_size + image_shape[2]
    ] = data

    manifest = list()

    start_point = box_size - int((box_size - core_size) / 2)
    cur_x = start_point
    cur_y = start_point
    cur_z = start_point
    while cur_z + (box_size - core_size) / 2 < image_shape[2] + box_size:
        next_chunk = padded_image[cur_x:cur_x + box_size,
                                  cur_y:cur_y + box_size,
                                  cur_z:cur_z + box_size]
        manifest.append(np.expand_dims(next_chunk, axis=0))
        cur_x += core_size
        if cur_x + (box_size - core_size) / 2 >= image_shape[0] + box_size:
            cur_y += core_size
            cur_x = start_point  # Reset
            if cur_y + (box_size - core_size) / 2 >= image_shape[1] + box_size:
                cur_z += core_size
                cur_y = start_point  # Reset
                cur_x = start_point  # Reset

    return manifest


def reconstruct_map(sub_maps, target_shape, box_size=64, core_size=50, dtype=np.float32):
    """
    Takes the output of the U-Net and reconstructs it to a matrix with the
    given target shape

    :param sub_maps: List of sub maps of same density map
    :param box_size: Size of sub maps
    :param core_size: Size of core that is considered after prediction
    :param target_shape: Target shape of density map data
    :param dtype: Data type of reconstructed map
    :return: Reconstructed density map
    """
    extract_start = int((box_size - core_size) / 2)
    extract_end = int((box_size - core_size) / 2) + core_size
    dimensions = [
        math.ceil(target_shape[0] / core_size) * core_size,
        math.ceil(target_shape[1] / core_size) * core_size,
        math.ceil(target_shape[2] / core_size) * core_size
    ]

    reconstruct_image = np.zeros((dimensions[0], dimensions[1], dimensions[2]), dtype=dtype)
    counter = 0
    for z_steps in range(int(dimensions[2] / core_size)):
        for y_steps in range(int(dimensions[1] / core_size)):
            for x_steps in range(int(dimensions[0] / core_size)):
                reconstruct_image[
                    x_steps * core_size:(x_steps + 1) * core_size,
                    y_steps * core_size:(y_steps + 1) * core_size,
                    z_steps * core_size:(z_steps + 1) * core_size
                ] = sub_maps[counter][extract_start:extract_end,
                                      extract_start:extract_end,
                                      extract_start:extract_end]
                counter += 1

    return reconstruct_image[:target_shape[0], :target_shape[1], :target_shape[2]]
