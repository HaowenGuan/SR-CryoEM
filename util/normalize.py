from util.density_map import DensityMap
import numpy as np


def normalize_map(density_map: DensityMap) -> DensityMap:
    """
    Normalizes the density values of the density map

    :param logger: Logger
    :param density_map: Input density map
    """
    density_map.data[density_map.data < 0] = 0
    percentile = np.percentile(density_map.data[np.nonzero(density_map.data)], 99.9)
    density_map.data /= percentile
    density_map.data[density_map.data > 1] = 1

    return density_map
