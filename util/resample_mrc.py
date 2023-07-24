from copy import deepcopy
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular
import numpy as np
import math
import time
import mrcfile


def resample_mrc(input_filename: str, output_filename: str, voxelSize=0.5,
                 DeepTracer=True, dimension_spec=None):
    """
    Crops density map to area with high density values and reSamples map on grid
    with voxel size of 0.5.  Algorithm is based on triLinear interpolation, detail
    information can be found in https://en.wikipedia.org/wiki/Trilinear_interpolation
    (Note: This function directly works on/rewrite the (map, mrc) files if the input and output is same)

    :param input_filename: Input directory of map file
    :param output_filename: Output directory of map file
    :param (optional) voxelSize: ReSampling voxel size (Note: 0.5 will only reSample the top 1% value's 3D space)
    :param (optional) DeepTracer: Using DeepTracer standard for 0.5 voxel size reSample task (default True)
    :param (optional) dimension_spec: A python list of box dimension in [x_ori, y_ori, z_ori, x_size, y_size, z_size]
    """

    print('Re-sampling density map on new grid with voxel size %s' % voxelSize)
    with mrcfile.open(input_filename) as mrc:
        data = deepcopy(mrc.data)
        v = deepcopy(mrc.voxel_size)
        origin = deepcopy(mrc.header.origin)
        nxstart = deepcopy(mrc.header.nxstart)
        nystart = deepcopy(mrc.header.nystart)
        nzstart = deepcopy(mrc.header.nzstart)

    old_ox = origin.x + nxstart * v.x
    old_oy = origin.y + nystart * v.y
    old_oz = origin.z + nzstart * v.z

    if dimension_spec is None:
        # If using DeepTracer standard. Only keep area of map which contains the highest 1% of values
        if voxelSize == 0.5 and DeepTracer:
            temp_data = deepcopy(data)
            threshold = np.percentile(temp_data, 99.0)
            temp_data[temp_data < threshold] = 0
            k_list, j_list, i_list = [sorted(i) for i in np.nonzero(temp_data)]

            # New dimensions before reSampling
            width = (int((i_list[-1] - i_list[0]) * v.x) + 10) * 2
            height = (int((j_list[-1] - j_list[0]) * v.y) + 10) * 2
            depth = (int((k_list[-1] - k_list[0]) * v.z) + 10) * 2

            # New origin
            new_ox = old_ox + (i_list[0] * v.x) - 5
            new_oy = old_oy + (j_list[0] * v.y) - 5
            new_oz = old_oz + (k_list[0] * v.z) - 5

            # Delete variable to save memory
            del temp_data, i_list, j_list, k_list
        else:
            new_ox, new_oy, new_oz = old_ox, old_oy, old_oz
            width = math.floor((data.shape[2] * v.x) / voxelSize)
            height = math.floor((data.shape[1] * v.y) / voxelSize)
            depth = math.floor((data.shape[0] * v.z) / voxelSize)
    else:
        new_ox, new_oy, new_oz, width, height, depth = dimension_spec

    # Adjust the reSampling dimensions and origin if it is go beyond the original shape
    new_ox, width = adjust_dim(old_ox, new_ox, data.shape[2], width, v.x, voxelSize)
    new_oy, height = adjust_dim(old_oy, new_oy, data.shape[1], height, v.y, voxelSize)
    new_oz, depth = adjust_dim(old_oz, new_oz, data.shape[0], depth, v.z, voxelSize)

    #  Try to create an array to resample the map onto
    try:
        s = np.empty((depth, height, width, 8), dtype='float32')
    except MemoryError as e:
        raise MemoryError('Map is too large to allocate resampled array')


    # ReSampling method based on triLinear interpolation, detail information can be found in
    # https://en.wikipedia.org/wiki/Trilinear_interpolation
    start_time = time.time()
    print('Running resampling ...')
    # Initializing triLinear interpolation matrix. A small modified version of the original triLinear interpolation M
    triLinearInterpolationM = sparse.csr_matrix([[1, 0, 0, 0, 0, 0, 0, 0],
                                                 [1, 1, 0, 0, 0, 0, 0, 0],
                                                 [1, 0, 1, 0, 0, 0, 0, 0],
                                                 [1, 1, 1, 1, 0, 0, 0, 0],
                                                 [1, 0, 0, 0, 1, 0, 0, 0],
                                                 [1, 1, 0, 0, 1, 1, 0, 0],
                                                 [1, 0, 1, 0, 1, 0, 1, 0],
                                                 [1, 1, 1, 1, 1, 1, 1, 1]])
    # Pre-recording axis information
    # xx, yy, zz records the corresponding index of each voxel in the original density map
    # x_d, y_d, z_d records the remainder to each corresponding index (refer z_d, y_d, x_d in triLinear interpolation)
    xx, x_d = axis_info(old_ox, new_ox, width, v.x, voxelSize)
    yy, y_d = axis_info(old_oy, new_oy, height, v.y, voxelSize)
    zz, z_d = axis_info(old_oz, new_oz, depth, v.z, voxelSize)
    x_d = x_d.reshape((1, 1, width))
    y_d = y_d.reshape((1, height, 1))
    z_d = z_d.reshape((depth, 1, 1))

    # Using runtime optimized approach if the reSampling voxel size is less than two times of
    # the original voxel size for fast runtime. Essentially, both algorithm can do the same job.
    # Approach justification: If the original voxel size is > half of the target reSampling voxel size,
    # every value in the original density map will be used at least once, so using optimized method
    # is faster since it assumes every value will be used at least once, vice-versa.
    meanVoxelSize = (v.x + v.y + v.z) / 3
    if meanVoxelSize > voxelSize / 2:
        # Pre-calculate z direction index tuple used for array querying
        data = data[zz[0]:zz[-1] + 2, yy[0]:yy[-1] + 2, xx[0]:xx[-1] + 2]
        zz, yy, xx = zz - zz[0], yy - yy[0], xx - xx[0]
        zRange, yRange, xRange = zz[-1] + 1, yy[-1] + 1, xx[-1] + 1
        data = data.repeat(2, axis=0)
        data = data[1:(2 * zRange) + 1]
        data = data.reshape((zRange, 2, yRange + 1, xRange + 1))
        # Pre-recording information for each cube
        records = np.empty((zRange, yRange, xRange, 2, 2, 2))
        for j in range(yRange):
            for i in range(xRange):
                records[:, j, i] = data[:, :, j:j + 2, i:i + 2]
        del data
        print("Data retrieving finished. Time:  %s seconds" % (time.time() - start_time))

        # Data format processing, reshape dimension to (8, x * y * z)
        records = np.reshape(records, (zRange * yRange * xRange, 8), order='F')
        records = np.transpose(records)

        # Using spsolve_triangular to solve matrix (Fastest matrix solving method for triangular matrix)
        records = spsolve_triangular(triLinearInterpolationM, records)
        print("TriLinear interpolation matrix solved. Time:  %s seconds" % (time.time() - start_time))

        # Data format processing, reshape dimension to (x, y, z, 8)
        records = np.transpose(records)
        records = np.reshape(records, (zRange, yRange, xRange, 8), order='F')

        # Pre-calculate x and y axis duplicate index range for fast coefficients' allocation
        zIndex = index_range(zz)
        yIndex = index_range(yy)
        # Allocate coefficients for each voxel according to its corresponding cube (s represent matrix's solution)
        for i in range(0, zRange):
            for j in range(0, yRange):
                s[zIndex[i, 0]:zIndex[i, 1], yIndex[j, 0]:yIndex[j, 1], :] = records[i, j, xx]
        del records
        print("Data assigning finished. Time:  %s seconds" % (time.time() - start_time))
    else:
        # If original voxel size is < half of the target reSampling voxel size, use ordinary data query method.
        b = np.array([[[data[i:i + 2, j:j + 2, k:k + 2] for k in xx] for j in yy] for i in zz])
        b = np.transpose(np.reshape(b, (depth * height * width, 8), order='F'))
        print("Data retrieving finished. Time:  %s seconds" % (time.time() - start_time))
        s = spsolve_triangular(triLinearInterpolationM, b)
        s = np.reshape(np.transpose(s), (depth, height, width, 8), order='F')
        print("TriLinear interpolation matrix solved. Time:  %s seconds" % (time.time() - start_time))

    # Calculate final reSampled data use coefficients from previous steps matrix solution
    resampledData = np.empty((depth, height, width), dtype='float32')
    resampledData[:, :, :] = s[:, :, :, 0] + s[:, :, :, 1] * z_d + (s[:, :, :, 2] + s[:, :, :, 3] * z_d) * y_d
    resampledData[:, :, :] += (s[:, :, :, 4] + s[:, :, :, 5] * z_d + (s[:, :, :, 6] + s[:, :, :, 7] * z_d) * y_d) * x_d
    myAlgorithmTime = time.time() - start_time
    print("Resampling finished. Runtime: %s seconds" % myAlgorithmTime)

    # Save the reSampled file into the output directory
    with mrcfile.new(output_filename, overwrite=True) as mrc:
        mrc.set_data(resampledData)
        mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart = 0, 0, 0
        mrc.header.origin = np.array((new_ox, new_oy, new_oz),
                                     dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')]).view(np.recarray)
        mrc.voxel_size = np.array((voxelSize, voxelSize, voxelSize),
                                  dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')]).view(np.recarray)
        print('Size of new map is (%d, %d, %d)' % resampledData.shape)
        mrc.update_header_stats()


def adjust_dim(old_ori: float, new_ori: float, old_dim: int, new_dim: int, old_vox: float, new_vox: float):
    """
    Auto-adjust the reSampling dimensions and origin back inside the original density map if it is go beyond the
    original shape
    :param old_ori: Old origin
    :param new_ori: New origin
    :param old_dim: Old data dimension
    :param new_dim: New data dimension
    :param old_vox: Old voxel size
    :param new_vox: New voxel size
    :return Adjusted origin and dimension
    """
    if new_ori < old_ori:
        new_dim -= math.ceil((old_ori - new_ori) / new_vox)
        new_ori += math.ceil((old_ori - new_ori) / new_vox) * new_vox
    if new_ori + (new_dim - 1) * new_vox > old_ori + (old_dim - 1) * old_vox:
        new_dim -= math.ceil((new_ori + (new_dim - 1) * new_vox - (old_ori + (old_dim - 1) * old_vox)) / new_vox)
    if new_ori + (new_dim - 1) * new_vox == old_ori + (old_dim - 1) * old_vox:
        new_dim -= 1
    return new_ori, new_dim


def axis_info(old_ori: float, new_ori: float, new_dim: int, old_vox: float, new_vox: float):
    """
    Pre-recording axis information
    Output attributes:
    "indexes" records the corresponding index of each voxel in the original density map
    "remainder" records the remainder to each corresponding index (refer z_d, y_d, x_d in triLinear interpolation)
    :param old_ori: Old origin
    :param new_ori: New origin
    :param new_dim: New data dimension
    :param old_vox: Old voxel size
    :param new_vox: New voxel size
    :return The list of index and the list remainder of each index
    """
    indexes, remainder = np.empty(new_dim, dtype='int'), np.empty(new_dim, dtype='float32')
    for i in range(new_dim):
        temp = (new_ori + i * new_vox - old_ori) / old_vox
        remainder[i] = temp % 1
        indexes[i] = math.floor(temp)
    return indexes, remainder


def index_range(array: np.array) -> np.array:
    """
    For a ordered array, finding the range of index of each number located, and return a array that record each
    number's starting index and ending index.
    :param array: 1D numpy ordered numpy array, array value must start from 0
    """
    index = np.empty((array[-1] + 1, 2), dtype='int')
    for i in range(array[-1] + 1):
        temp = np.where(array == i)[0]
        try:
            index[i, 0], index[i, 1] = temp[0], temp[-1] + 1
        except IndexError:
            index[i, 0], index[i, 1] = 0, 0
    return index
