from util.density_map import DensityMap
from util.resample_mrc import resample_mrc
import tempfile
import shutil
import os


def resample(density_map: DensityMap, voxelSize=0.5,
             DeepTracer=True, dimension_spec=None) -> DensityMap:
    """
    Crops density map to area with high density values and reSamples map on grid
    with voxel size of 0.5.  Algorithm is based on triLinear interpolation, detail
    information can be found in https://en.wikipedia.org/wiki/Trilinear_interpolation

    :param logger: Logger
    :param density_map: Input density map
    :param (optional) voxelSize: ReSample voxel size (Note: 0.5 will only reSample the top 1% value's 3D space)
    :param (optional) DeepTracer: Using DeepTracer standard for 0.5 voxel size reSample task (default True)
    :param (optional) dimension_spec: A python list of box dimension in [x_ori, y_ori, z_ori, x_size, y_size, z_size]
    :return: Density map with updated voxel size
    """

    tmp_dir = tempfile.mkdtemp(prefix='reSampling_map')
    temp_map_dir = os.path.join(tmp_dir, 'reSampling_map.mrc')
    density_map.save(temp_map_dir)

    resample_mrc(temp_map_dir, temp_map_dir, voxelSize, DeepTracer=DeepTracer, dimension_spec=dimension_spec)

    density_map.update_from_file(temp_map_dir)
    shutil.rmtree(tmp_dir)

    return density_map
