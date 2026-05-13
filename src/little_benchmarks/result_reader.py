import copy
from matplotlib.ticker import FuncFormatter
import bisect
from typing import Any, NamedTuple, Callable, Literal
from pathlib import Path
from dataclasses import dataclass, asdict
import itertools

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ms_nexus_tools.lib.bounds import Shape
from ms_nexus_tools.lib.utils import parse_bytes, format_bytes


@dataclass
class FileResult:
    filename: str
    compression: str
    filetype: str
    time: float
    size: int
    chunk_shape: Shape
    chunk_size: int
    memory_count: float
    memory_shape: Shape
    memory_size: int


class Col(NamedTuple):
    col: str
    prefix: str
    label: str


AxesScale = Literal["semilogx", "semilogy", "loglog", "plot"]


def set_scale(axis: plt.Axes, axis_style: AxesScale):
    match axis_style:
        case "loglog":
            axis.set_xscale("log", base=2)
            axis.set_yscale("log", base=2)
        case "semilogx":
            axis.set_xscale("log", base=2)
        case "semilogy":
            axis.set_yscale("log", base=2)
        case "plot":
            pass
        case _:
            raise ValueError(f"Did not understand scale {axis_style}")


class PlotData(NamedTuple):
    row: str
    x_data: np.ndarray[tuple[int], np.dtype[np.int32]]
    y_data_min: np.ndarray[tuple[int], np.dtype[np.int32]]
    y_data_mean: np.ndarray[tuple[int], np.dtype[np.int32]]
    y_data_max: np.ndarray[tuple[int], np.dtype[np.int32]]

    def filter(self, x_min, x_max) -> "PlotData":
        if x_min is not None:
            min_i = bisect.bisect_left(self.x_data, x_min)
        else:
            min_i = 0

        if x_max is not None:
            max_i = bisect.bisect_right(self.x_data, x_max)
        else:
            max_i = len(self.x_data)

        return PlotData(
            row=self.row,
            x_data=self.x_data[min_i:max_i],
            y_data_min=self.y_data_min[min_i:max_i],
            y_data_mean=self.y_data_mean[min_i:max_i],
            y_data_max=self.y_data_max[min_i:max_i],
        )

    def plot(self, axis, axis_style, label, color, marker, linestyle):
        set_scale(axis, axis_style)
        axis.plot(
            self.x_data,
            self.y_data_mean,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
        )

    def plot_agg(self, axis, axis_style, label, color, marker, linestyle):
        set_scale(axis, axis_style)
        axis.fill_between(
            self.x_data,
            self.y_data_min,
            self.y_data_max,
            color=color,
            alpha=0.4,
        )
        axis.plot(
            self.x_data,
            self.y_data_mean,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
        )


def accept_all(row: dict[str, Any]) -> bool:
    return True


def accept_compressed(row: dict[str, Any]) -> bool:
    return "None" not in row["compression"]


def accept_blosc(row: dict[str, Any]) -> bool:
    return row["compression"] == "blosc"


def accept_zarr(row: dict[str, Any]) -> bool:
    return row["filetype"] == "zarr"


def reject_gzip(row: dict[str, Any]) -> bool:
    return "gzip" not in row["compression"]


def reject_gzip9(row: dict[str, Any]) -> bool:
    return row["compression"] != "gzip-9"


def plot_x_sizes(ax: plt.Axes, plot_data: PlotData):
    max_x = np.max(plot_data.x_data)
    min_x = np.min(plot_data.x_data)
    items = 0
    i = 3
    while items <= max_x:
        if items < min_x:
            items = 2 ** (2 * i)
            i += 1
            continue
        ax.axvline(items, color="black", linestyle="-.", linewidth=0.5)
        ax.annotate(
            format_bytes(items),
            xy=(items, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            va="top",
            rotation=270,
            fontsize=10,
            color="black",
            backgroundcolor="white",
        )
        items = 2 ** (2 * i)
        i += 1


@dataclass
class DataFilter:
    x_min: None | float | int = None
    x_max: None | float | int = None
    run: Callable[[dict[str, Any]], bool] = accept_all


def plot_on_axis(
    ax: plt.Axes,
    data,
    x_col: Col,
    y_col: Col,
    lines: list[Col] = [Col("compression", "comp", "")],
    callbacks: list[Callable[[plt.Axes, PlotData], None]] = [],
    filters: list[DataFilter] = [],
    axis_style: AxesScale = "semilogx",
    aggrigate: bool = False,
):
    cols = [pl.col(l.col) for l in lines]
    rows = data.select(*cols).unique().sort(by=cols)

    markers = "o^sp*dP"
    colors_and_markers = [
        (c, m) for m, c in itertools.product(markers, mcolors.TABLEAU_COLORS)
    ]

    x_min = None
    x_max = None
    for filter in filters:
        if filter.x_min is not None:
            x_min = filter.x_min if x_min is None else min(x_min, filter.x_min)
        if filter.x_max is not None:
            x_max = filter.x_max if x_max is None else max(x_max, filter.x_max)

    for ii, row in enumerate(rows.iter_rows(named=True)):
        # current_color = colors[ii]
        current_color, current_marker = colors_and_markers[ii]

        run_loop = True
        for filter in filters:
            run_loop &= filter.run(row)

        if not run_loop:
            continue

        dt = (
            data.filter(*[pl.col(l.col) == row[l.col] for l in lines])
            .group_by(
                pl.col(x_col.col),
            )
            .agg(
                pl.col(x_col.col).mean().alias("x"),
                pl.col(y_col.col).min().alias("min"),
                pl.col(y_col.col).mean().round(2).alias("mean"),
                pl.col(y_col.col).max().alias("max"),
            )
            .sort(by=pl.col("x"))
        )

        plot_data = PlotData(
            row=row,
            x_data=dt["x"].to_numpy(),
            y_data_min=dt["min"].to_numpy(),
            y_data_mean=dt["mean"].to_numpy(),
            y_data_max=dt["max"].to_numpy(),
        )
        plot_data = plot_data.filter(x_min, x_max)
        row_strs = [f"{l.prefix}: {row[l.col]}" for l in lines]
        str_s = f"{y_col.prefix}: {' '.join(row_strs)}"
        linestyle = "--"
        if aggrigate:
            plot_data.plot_agg(
                ax, axis_style, str_s, current_color, current_marker, linestyle
            )
        else:
            plot_data.plot(
                ax, axis_style, str_s, current_color, current_marker, linestyle
            )

        for callback in callbacks:
            callback(ax, plot_data)

    ax.set_xlabel(x_col.label)
    ax.set_ylabel(y_col.label)


def plot_single(
    data,
    x_col: Col,
    y_col: Col,
    lines: list[Col] = [Col("compression", "comp", "")],
    callbacks: list[Callable[[plt.Axes, PlotData], None]] = [],
    filters: list[DataFilter] = [],
    subplots_adjust: None | dict[str, Any] = None,
    filename: None | Path = None,
    axis_style: AxesScale = "semilogx",
    aggrigate: bool = False,
):
    fig, ax = plt.subplots(figsize=(16, 12))
    if subplots_adjust:
        fig.subplots_adjust(**subplots_adjust)

    plot_on_axis(
        ax=ax,
        data=data,
        x_col=x_col,
        y_col=y_col,
        lines=lines,
        callbacks=callbacks,
        filters=filters,
        axis_style=axis_style,
        aggrigate=aggrigate,
    )

    ax.legend()

    if filename is not None:
        ic(filename)
        fig.savefig(filename)
        plt.close(fig)


def plot_vertical(
    data,
    x_col: Col,
    y_col: Col,
    lines: list[Col] = [Col("compression", "comp", "")],
    callbacks: list[Callable[[plt.Axes, PlotData], None]] = [],
    filters: list[DataFilter] = [],
    subplots_adjust: None | dict[str, Any] = None,
    filename: None | Path = None,
    axis_style: AxesScale = "semilogx",
    aggrigate: bool = False,
):
    nrows = len(filters)
    fig, axs = plt.subplots(nrows=nrows, figsize=(16, nrows * 8))
    if subplots_adjust:
        fig.subplots_adjust(**subplots_adjust)

    for ii, filter in enumerate(filters):
        plot_on_axis(
            axs[ii],
            data,
            x_col,
            y_col,
            lines,
            callbacks,
            [filter],
            axis_style,
            aggrigate,
        )

        axs[ii].legend()

    if filename is not None:
        ic(filename)
        fig.savefig(filename)
        plt.close(fig)


def yaxis_format_bytes(ax: plt.Axes, plot_data: PlotData):
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos=None: format_bytes(int(round(x))))
    )


def xaxis_format_bytes(ax: plt.Axes, plot_data: PlotData):
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos=None: format_bytes(int(round(x))))
    )


def xaxis_format_float(ax: plt.Axes, plot_data: PlotData):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos=None: f"{x}"))


def plot_size_vs_chunk_size(data, base_dir):
    plot_single(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("size", "", "file size"),
        lines=[
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
        ],
        callbacks=[yaxis_format_bytes, plot_x_sizes],
        filename=base_dir / "size full.png",
        aggrigate=True,
    )
    plot_single(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("size", "", "file size"),
        lines=[
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
        ],
        callbacks=[yaxis_format_bytes, plot_x_sizes],
        filters=[DataFilter(run=accept_compressed)],
        filename=base_dir / "size compressed.png",
        aggrigate=True,
    )
    plot_single(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("size", "", "file size"),
        lines=[
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
        ],
        callbacks=[yaxis_format_bytes, plot_x_sizes],
        filters=[DataFilter(run=accept_blosc)],
        filename=base_dir / "size blosc.png",
        aggrigate=True,
    )


def plot_time_vs_chunk_size(data, base_dir):
    plot_single(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("time", "", "write time (seconds)"),
        lines=[
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
        ],
        callbacks=[plot_x_sizes],
        filename=base_dir / "time full.png",
        aggrigate=True,
    )
    plot_vertical(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("time", "", "write time (seconds)"),
        lines=[
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
        ],
        callbacks=[plot_x_sizes],
        filters=[
            DataFilter(run=accept_compressed),
            DataFilter(x_min=16 * 1024, run=accept_compressed),
        ],
        filename=base_dir / "time compressed.png",
        aggrigate=True,
    )
    plot_vertical(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("time", "", "write time (seconds)"),
        lines=[
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
        ],
        callbacks=[plot_x_sizes],
        filters=[
            DataFilter(run=accept_blosc),
            DataFilter(x_min=16 * 1024, run=accept_blosc),
        ],
        filename=base_dir / "time blosc.png",
        aggrigate=True,
    )


def plot(data, base_dir):

    plot_size_vs_chunk_size(data, base_dir)
    plot_time_vs_chunk_size(data, base_dir)

    plot_vertical(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("time", "", "write time (seconds)"),
        lines=[
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
            Col("memory_count", "mc", ""),
        ],
        callbacks=[plot_x_sizes],
        filters=[
            DataFilter(run=lambda x: accept_blosc(x) and accept_zarr(x)),
            DataFilter(
                x_min=16 * 1024, run=lambda x: accept_blosc(x) and accept_zarr(x)
            ),
            DataFilter(
                x_min=16 * 1024,
                run=lambda x: (
                    accept_blosc(x) and accept_zarr(x) and x["memory_count"] >= 1.0
                ),
            ),
        ],
        filename=base_dir / "time zarr blosc.png",
        aggrigate=True,
    )

    plot_vertical(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("time", "", "write time (seconds)"),
        lines=[
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
            Col("memory_count", "mc", ""),
        ],
        callbacks=[plot_x_sizes],
        filters=[
            DataFilter(run=lambda x: accept_blosc(x) and (not accept_zarr(x))),
            DataFilter(
                x_min=16 * 1024, run=lambda x: accept_blosc(x) and (not accept_zarr(x))
            ),
            DataFilter(
                x_min=16 * 1024,
                run=lambda x: (
                    accept_blosc(x)
                    and (not accept_zarr(x))
                    and x["memory_count"] >= 1.0
                ),
            ),
        ],
        filename=base_dir / "time hdf blosc.png",
        aggrigate=True,
    )

    plot_vertical(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("size", "", "file size"),
        lines=[
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
            Col("memory_count", "mc", ""),
        ],
        callbacks=[plot_x_sizes, yaxis_format_bytes],
        filters=[
            DataFilter(run=lambda x: accept_blosc(x) and accept_zarr(x)),
            DataFilter(
                x_min=16 * 1024, run=lambda x: accept_blosc(x) and accept_zarr(x)
            ),
            DataFilter(
                x_min=16 * 1024,
                run=lambda x: (
                    accept_blosc(x) and accept_zarr(x) and x["memory_count"] >= 1.0
                ),
            ),
        ],
        filename=base_dir / "size zarr blosc.png",
        aggrigate=True,
    )

    plot_vertical(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("size", "", "file size"),
        lines=[
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
            Col("memory_count", "mc", ""),
        ],
        callbacks=[plot_x_sizes, yaxis_format_bytes],
        filters=[
            DataFilter(run=lambda x: accept_blosc(x) and (not accept_zarr(x))),
            DataFilter(
                x_min=16 * 1024, run=lambda x: accept_blosc(x) and (not accept_zarr(x))
            ),
            DataFilter(
                x_min=16 * 1024,
                run=lambda x: (
                    accept_blosc(x)
                    and (not accept_zarr(x))
                    and x["memory_count"] >= 1.0
                ),
            ),
        ],
        filename=base_dir / "size hdf blosc.png",
        aggrigate=True,
    )

    # plt.show()


def read() -> None:

    in_path = Path("C:/Workspace/data/out/little-benchmark/output_zarr.log")
    base_dir = Path(in_path.parent / "plots")
    base_dir.mkdir(parents=True, exist_ok=True)

    results: list[FileResult] = []
    with open(in_path, "r") as results_file:
        current_data: dict[str, Any] = {}
        for line_no, line in enumerate(results_file):
            try:
                parts = [l_part.strip() for l_part in line.strip().split()]
                name = parts[0][:-1]
                if line.startswith("Wrote:"):
                    if len(current_data) != 0:
                        hdf = (current_data["hdf_time"], current_data["hdf_size"])
                        zarr = (current_data["zarr_time"], current_data["zarr_size"])

                        new_data = copy.copy(current_data)
                        del new_data["hdf_time"]
                        del new_data["hdf_size"]
                        del new_data["zarr_time"]
                        del new_data["zarr_size"]

                        hdf_data = new_data | dict(
                            time=hdf[0], size=hdf[1], filetype="hdf"
                        )
                        zarr_data = new_data | dict(
                            time=zarr[0], size=zarr[1], filetype="zarr"
                        )
                        results.append(FileResult(**hdf_data))
                        results.append(FileResult(**zarr_data))
                    else:
                        assert len(current_data) == 0

                    current_data = dict(filename=" ".join(parts[1:]))
                    compression = current_data["filename"].split(maxsplit=1)[0]
                    current_data["compression"] = compression
                elif "shape" in name:
                    end = " ".join(parts[1:])
                    end = end.replace("(", "").replace(")", "")
                    shape = tuple([int(value.strip()) for value in end.split(",")])
                    current_data[name] = shape
                elif "size" in name:
                    size = parse_bytes(parts[-1])
                    current_data[name] = size
                elif "time" in name:
                    current_data[name] = float(parts[1])
                elif name == "memory_count":
                    current_data[name] = float(parts[1])
                else:
                    raise ValueError(
                        f"Unknown result part: {parts[0]} on line {line_no}"
                    )
            except:
                print(f"Error on line: {line_no}")
                print(line)
                raise

    max_memory_size = -1
    min_count = 1e7

    for r in results:
        if r.memory_size > max_memory_size:
            max_memory_size = r.memory_size
            min_count = 1e7
        elif r.memory_size == max_memory_size:
            min_count = min(r.memory_count, min_count)

    filtered_results = []
    for r in results:
        if r.memory_size < max_memory_size:
            filtered_results.append(r)
        elif r.memory_count <= min_count:
            filtered_results.append(r)

    data = pl.DataFrame([asdict(r) for r in filtered_results])
    ic(data["time"].sum())

    plot(data, base_dir)
