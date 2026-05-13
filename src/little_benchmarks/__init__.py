from ms_nexus_tools.lib.utils import format_bytes
import time
from typing import Any
import itertools
from pathlib import Path
import concurrent.futures as cfutures

from ms_nexus_tools.lib.bounds import Chunk
from ms_nexus_tools.lib.chunking import Chunker
import h5py
import zarr
import numpy as np
import hdf5plugin
from tqdm import tqdm, trange
# from icecream import ic

from .result_reader import read as read_results

import warnings

fle = None


def write(s):
    global fle
    if fle is not None:
        fle.write(s)
        fle.write("\n")
    else:
        ic(fle)
    print(s)


def comp_str(compression: Any, compression_opts: Any) -> str:
    if compression is None:
        return "None"
    elif isinstance(compression, hdf5plugin.Blosc):
        return "blosc"
    else:
        return f"{compression}-{compression_opts}"


def write_hdf5(
    filename: Path,
    memory: Chunker,
    chunker: Chunker,
    compression: Any,
    compression_opts: Any,
    data,
) -> float:

    start = stop = 0.0

    match compression:
        case "none":
            compressor = None
        case "blosc":
            compressor = hdf5plugin.Blosc(
                cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
            )
        case "gzip":
            compressor = "gzip"
        case _:
            raise NotImplementedError(
                f"Specified compression unsupported: {compression}"
            )

    memory_chunks = [c for c in memory.chunks()]
    start = time.monotonic()

    with h5py.File(filename, "w") as h5:
        ds = h5.create_dataset(
            "data",
            shape=memory.data_shape,
            dtype=np.int32,
            compression=compressor,
            compression_opts=compression_opts,
            chunks=chunker.chunk_shape,
        )
        for ii, c in enumerate(tqdm(memory_chunks, desc="Writing")):
            ds[*c] = data[*c]
        h5.close()

    stop = time.monotonic()
    return stop - start


def write_zarr(
    filename: Path,
    memory: Chunker,
    chunker: Chunker,
    compression: Any,
    compression_opts: Any,
    data,
) -> float:

    start = stop = 0.0

    match compression:
        case "none":
            compressor = None
        case "blosc":
            compressor = zarr.codecs.BloscCodec(
                cname="lz4", clevel=5, shuffle="shuffle"
            )
        case "gzip":
            compressor = zarr.codecs.GzipCodec(level=compression_opts)
        case _:
            raise NotImplementedError(
                f"Specified compression unsupported: {compression}"
            )

    memory_chunks = [c for c in memory.chunks()]
    with warnings.catch_warnings():
        # This is partcularly for the memory buffer smaller than the chunk
        warnings.simplefilter("ignore")
        start = time.monotonic()

        # with zarr.storage.LocalStore(filename) as store:
        with zarr.storage.ZipStore(filename, mode="w") as store:
            z = zarr.create_array(
                store=store,
                shape=memory.data_shape,
                chunks=chunker.chunk_shape,
                dtype=np.int32,
                compressors=compressor,
            )
            for ii, c in enumerate(tqdm(memory_chunks, desc="Writing")):
                z[*c] = data[*c]
            store.close()

        stop = time.monotonic()
    return stop - start


def main() -> None:

    out_path = Path("C:/Workspace/data/out/little-benchmark/")
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "output.log"
    width = 256

    rng = np.random.default_rng(seed=1298)

    data = rng.random((width, width, width)) * 1000

    # compressions: dict[Any, list[Any]] = dict(gzip=[4, 9], none=[None])
    compressions: dict[Any, list[Any]] = dict(gzip=[4], none=[None], blosc=[None])

    chunk_sizes = [(2**i) for i in range(9, 25, 1)]
    memory_counts = [0.5, 1, 1.5, 2, 3]
    memory_counts_dict = {count: ii for ii, count in enumerate(memory_counts)}
    max_memory_buffer = np.prod(data.shape)

    memory_combos = [
        (chunk_size, memory_count)
        for chunk_size, memory_count in itertools.product(chunk_sizes, memory_counts)
    ]

    compression_combos = []
    for compression, opts in compressions.items():
        for compression_opts in opts:
            compression_combos.append((compression, compression_opts))

    total_count = len(compression_combos) * len(memory_combos)
    current_count = 0

    global fle
    with open(log_file, "w") as fle:
        for compression, compression_opts in compression_combos:
            for chunk_size, memory_count in memory_combos:
                chunker = Chunker.from_max_item_count(
                    data_shape=data.shape,
                    priorities=(1, 1, 1),
                    items_per_chunk=chunk_size,
                )
                if np.all(
                    [
                        c * memory_count > d
                        for c, d in zip(chunker.chunk_shape, chunker.data_shape)
                    ]
                ):
                    continue

                memory = Chunker.from_chunk_shape(
                    data_shape=data.shape,
                    chunk_shape=tuple(
                        [int(round(c * memory_count)) for c in chunker.chunk_shape]
                    ),
                )
                memory_chunks = [c for c in memory.chunks()]

                filename = f"{comp_str(compression, compression_opts)} m-{np.prod(memory.chunk_shape)} c-{np.prod(chunker.chunk_shape)}"
                hdf_file: Path = out_path / f"{filename}.hdf5"
                zarr_file: Path = out_path / f"{filename}.zarr"

                hdf_time = write_hdf5(
                    hdf_file,
                    memory,
                    chunker,
                    compression,
                    compression_opts,
                    data,
                )
                time.sleep(1)
                zarr_time = write_zarr(
                    zarr_file,
                    memory,
                    chunker,
                    compression,
                    compression_opts,
                    data,
                )
                for _ in trange(5, desc="pause"):
                    time.sleep(1)

                print(
                    f" --- {current_count}/{total_count} = {float(current_count) / total_count * 100:.1f}% ---"
                )
                write(f"Wrote: {filename}")
                write(f" hdf_time: {hdf_time:.1f} seconds.")
                write(f" hdf_size: {format_bytes(hdf_file.stat().st_size)}")
                write(f" zarr_time: {zarr_time:.1f} seconds.")
                write(f" zarr_size: {format_bytes(zarr_file.stat().st_size)}")
                write(f" chunk_shape: {chunker.chunk_shape}")
                write(f" chunk_size: {format_bytes(np.prod(chunker.chunk_shape) * 4)}")
                write(f" memory_count: {memory_count}")
                write(f" memory_shape: {memory.chunk_shape}")
                write(f" memory_size: {format_bytes(np.prod(memory.chunk_shape) * 4)}")

                hdf_file.unlink()
                zarr_file.unlink()
                current_count += 1


def read():
    read_results()
