from copy import deepcopy
from typing import Tuple
import mrcfile
import numpy as np
import os


class DensityMap:
    """
    Class represents a density map and realizes a wrapper around MRC files for
    simplified access to data and header elements

    For more information about the MRC file format see
    http://www.ccpem.ac.uk/mrc_format/mrc2014.php

    :param name: Name of the density map
    :param filename: If the density map was read from a file, this is the file
    name
    :param data: Three dimensional numpy array representing the data of the
    density map
    :param origin: Origin (offset) of the density data
    :param voxel_size: Size of each voxel (entry in the data array)
    :param nxstart: Location of first column in unit cell
    :param nystart: Location of first row in unit cell
    :param nzstart: Location of first section in unit cell
    """

    def __init__(self, name: str, filename: str = None, data=None, shape=None, origin=None,
                 voxel_size=None, nxstart=None, nystart=None, nzstart=None):
        self.name = name
        self.filename = filename
        self.data = data
        self.shape = shape
        self.origin = origin
        self.voxel_size = voxel_size
        self.nxstart = nxstart
        self.nystart = nystart
        self.nzstart = nzstart

    def __str__(self):
        return '\'%s\' Density Map' % self.name

    @staticmethod
    def open(filename: str, name: str = None) -> 'DensityMap':
        """
        Opens MRC file and creates a density map from it

        :param filename: Path to MRC file
        :param name: Name of density map (if none is provided, the name of the
        file is used)
        :return:
        """
        if name is None:
            name = os.path.basename(filename).split('.')[0]

        try:
            with mrcfile.open(filename) as mrc:
                # deepcopy everything because file attributes are read-only
                voxels = deepcopy(mrc.data)
                voxel_size = deepcopy(mrc.voxel_size)
                nxstart = deepcopy(mrc.header.nxstart)
                nystart = deepcopy(mrc.header.nystart)
                nzstart = deepcopy(mrc.header.nzstart)
                origin = deepcopy(mrc.header.origin)

                # Get the axis ordering, this specifies the Cartesian axis for the
                # respective C, R, and S values. Values of 1, 2, and 3 correspond
                # to the X, Y, and Z axes respectively. Subtract 1 from the value
                # to align the value with a zero-based index.
                axis_order = (mrc.header.mapc - 1, mrc.header.mapr - 1, mrc.header.maps - 1)
                # If invalid axis order data, use default ordering
                if ((0 not in axis_order) or (1 not in axis_order) or (2 not in axis_order)):
                    axis_order = (0, 1, 2)

            # The mrcfile library "does not attempt to swap the axes and simply assigns the
            # columns to X, rows to Y and sections to Z. (The data array is indexed in C style,
            # so data values can be accessed using mrc.data[z][y][x].)"
            # [https://mrcfile.readthedocs.io/en/latest/usage_guide.html#data-dimensionality]
            # So we need transpose the voxel array and metadata to a consistent order.
            if axis_order[0] != 0 or axis_order[1] != 1 or axis_order[2] != 2:
                # Calculate the axis conversion
                converter = [None, None, None]
                converter[axis_order[0]] = 0
                converter[axis_order[1]] = 1
                converter[axis_order[2]] = 2

                # Convert the voxel array
                voxels = np.swapaxes(voxels, 0, 2)  # Swap to data[c][r][s] for conversion
                voxels = np.transpose(voxels, converter)  # Transpose to data[x][y][z]
                voxels = np.swapaxes(voxels, 0, 2)  # Swap back to data[z][y][x] as DeepTracer expects

                # Convert voxel offset metadata
                voxel_offset_crs = deepcopy((nxstart, nystart, nzstart))
                nxstart = voxel_offset_crs[converter[0]]
                nystart = voxel_offset_crs[converter[1]]
                nzstart = voxel_offset_crs[converter[2]]

            return DensityMap(
                name,
                filename,
                data=voxels,
                shape=voxels.shape,
                origin=origin,
                voxel_size=voxel_size,
                nxstart=nxstart,
                nystart=nystart,
                nzstart=nzstart
            )
        except ValueError as e:
            raise ValueError('Error occurred opening %s (%s)' % (filename, str(e)))

    def update_from_file(self, filename: str):
        """Update density map with contents of given file"""
        with mrcfile.open(filename) as mrc:
            self.filename = filename
            self.data = deepcopy(mrc.data)
            self.shape = mrc.data.shape
            self.origin = mrc.header.origin
            self.voxel_size = mrc.voxel_size
            self.nxstart = mrc.header.nxstart
            self.nystart = mrc.header.nystart
            self.nzstart = mrc.header.nzstart

    def ijk_to_xyz(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """Converts indices i, j, and k into x, y, and z coordinates"""
        return (
            ((k + self.nxstart) * self.voxel_size.x) + self.origin.x,
            ((j + self.nystart) * self.voxel_size.y) + self.origin.y,
            ((i + self.nzstart) * self.voxel_size.z) + self.origin.z
        )

    def xyz_to_ijk(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Converts x, y, and z coordinates into indices i, j, and k """
        return (
            max(0, min(self.shape[0] - 1, int(round((z - self.origin.z) / self.voxel_size.z)) + self.nzstart)),
            max(0, min(self.shape[1] - 1, int(round((y - self.origin.y) / self.voxel_size.y)) + self.nystart)),
            max(0, min(self.shape[2] - 1, int(round((x - self.origin.x) / self.voxel_size.x)) + self.nxstart))
        )

    def save(self, filename: str, compress: bool = False):
        """
        Saves density map as MRC file to given path

        :param filename: Path to which file will be saved to
        :param compress: Reduce file size by re-sampling data on grid with voxel size of 1.5
        """
        if self.data is None:
            raise AttributeError('MRC data cannot be None')

        with mrcfile.new(filename, overwrite=True) as mrc:
            mrc.set_data(self.data)
            if self.origin is not None:
                mrc.header.origin = self.origin
            if self.voxel_size is not None:
                mrc.voxel_size = self.voxel_size
            if self.nxstart is not None and self.nystart is not None and self.nzstart is not None:
                mrc.header.nxstart = self.nxstart
                mrc.header.nystart = self.nystart
                mrc.header.nzstart = self.nzstart
            mrc.update_header_stats()

        return self
