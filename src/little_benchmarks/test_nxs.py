# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from nexusformat.nexus import nxload
import hdf5plugin

# from ms_nexus_tools.lib.chunking import chunker

import numpy as np


from icecream import ic


with nxload("./test.nxs", "w").nxfile as h5:
    width = 1024
    blob = np.random.rand(width, width, width) * 1000
    ic(blob.dtype)

    ds = h5.create_dataset(
        "data",
        shape=(1, *blob.shape),
        dtype=np.int32,
        # shuffle=False,
        compression=hdf5plugin.Blosc(),
        chunks=(1, width / 2, width / 2, width / 2),
    )
    ds[0, :, :, :] = blob[:, :, :]
