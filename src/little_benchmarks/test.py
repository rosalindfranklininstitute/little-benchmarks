# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

import h5py

import numpy as np


with h5py.File("./test.hdf5", "w") as h5:
    width = 1024
    blob = np.random.rand(width, width, width)

    ds = h5.create_dataset("data", shape=blob.shape, dtype=blob.dtype)
    ds[:, :, :] = blob[:, :, :]
