import h5py

import numpy as np


with h5py.File("./test.hdf5", "w") as h5:
    width = 1024
    blob = np.random.rand(width, width, width)

    ds = h5.create_dataset("data", shape=blob.shape, dtype=blob.dtype)
    ds[:, :, :] = blob[:, :, :]
