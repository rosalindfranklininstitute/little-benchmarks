# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

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
    compression: str
    filetype: str
    time: float
    size: int
    data_size: int
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
    return "none" not in row["compression"]


def accept_blosc(row: dict[str, Any]) -> bool:
    return "blosc" in row["compression"]


def accept_zarr(row: dict[str, Any]) -> bool:
    return "zarr" in row["filetype"]


def accept_deflate_zarr(row: dict[str, Any]) -> bool:
    return "deflate_zarr" == row["filetype"]


def accept_store_zarr(row: dict[str, Any]) -> bool:
    return "store_zarr" == row["filetype"]


def accept_hdf(row: dict[str, Any]) -> bool:
    return "hdf" == row["filetype"]


def accept_int_mc(row: dict[str, Any]) -> bool:
    mc = row["memory_count"]
    return int(mc) == mc


def reject_gzip(row: dict[str, Any]) -> bool:
    return "gzip" not in row["compression"]


def accept_gzip(row: dict[str, Any]) -> bool:
    return "gzip" in row["compression"]


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


FilterFunc = Callable[[dict[str, Any]], bool]


@dataclass
class DataFilter:
    x_min: None | float | int = None
    x_max: None | float | int = None
    run: FilterFunc = accept_all


def get_colors_for_columns(data, cols: list[pl.Expr], sort: bool):
    counts = [len(data.select(c).unique()) for c in cols]
    values = [[v[0] for v in data.select(c).unique().sort(by=c).rows()] for c in cols]
    if sort:
        inx_col = np.argsort(counts)[::-1]
    else:
        inx_col = np.arange(len(values))

    sorted_values = [values[ii] for ii in inx_col]
    sorted_counts = [counts[ii] for ii in inx_col]

    inv_col = np.zeros_like(inx_col)
    inv_col[inx_col] = np.arange(len(inx_col))

    colors = itertools.cycle(mcolors.TABLEAU_COLORS)
    linestyles = itertools.cycle(["-", "--", "-.", ":"])
    markers = itertools.cycle("o^sp*dP")

    if np.any([c == 0 for c in counts]):
        print("WARNING: not values provided.")
        return {}

    color_choices = [next(colors) for _ in sorted_values[0]]
    if len(values) > 1:
        linestyle_choices = [next(linestyles) for _ in sorted_values[1]]
        if len(values) > 2:
            marker_choices = [next(markers) for _ in sorted_values[2]]
        else:
            marker_choices = ["o"]
    else:
        marker_choices = ["o"]
        linestyle_choices = ["-"]

    current_color = color_choices[0]
    current_marker = marker_choices[0]
    current_linestyle = linestyle_choices[0]

    styles = {}

    for ii, value in enumerate(itertools.product(*sorted_values)):
        for jj in range(1, min(len(values) + 1, 4)):
            rate_of_change = int(np.prod(sorted_counts[jj:]))
            inx = (ii // rate_of_change) % sorted_counts[jj - 1]
            match (jj - 1) % 3:
                case 0:
                    current_color = color_choices[inx]
                case 1:
                    current_linestyle = linestyle_choices[inx]
                case 2:
                    current_marker = marker_choices[inx]
        style = (current_color, current_marker, current_linestyle)
        key = tuple([value[inv] for inv in inv_col])
        styles[key] = style
    return styles


def format_col(col, value):
    if "size" in col:
        return format_bytes(value)
    else:
        return str(value)


def plot_on_axis(
    ax: plt.Axes,
    data,
    x_col: Col,
    y_col: Col,
    lines: list[Col] = [Col("compression", "comp", "")],
    callbacks: list[Callable[[plt.Axes, PlotData], None]] = [],
    filter_cols: list[tuple[Col, Any]] = [],
    filters: list[DataFilter] = [],
    axis_style: AxesScale = "semilogx",
    aggrigate: bool = False,
):
    cols = [pl.col(l.col) for l in lines]
    if len(filter_cols) > 0:
        filtered_data = data.filter(*[pl.col(c.col) == v for c, v in filter_cols])
    else:
        filtered_data = data

    rows = filtered_data.select(*cols).unique().sort(by=cols)

    styles = get_colors_for_columns(filtered_data, cols, sort=False)

    x_min = None
    x_max = None
    for filter in filters:
        if filter.x_min is not None:
            x_min = filter.x_min if x_min is None else min(x_min, filter.x_min)
        if filter.x_max is not None:
            x_max = filter.x_max if x_max is None else max(x_max, filter.x_max)

    for ii, row in enumerate(rows.iter_rows(named=True)):
        key = tuple([row[l.col] for l in lines])
        current_color, current_marker, current_linestyle = styles[key]

        run_loop = True
        for filter in filters:
            run_loop &= filter.run(row)

        if not run_loop:
            continue

        dt = (
            filtered_data.filter(*[pl.col(l.col) == row[l.col] for l in lines])
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
        row_strs = [f"{l.prefix}: {format_col(l.col, row[l.col])}" for l in lines]
        str_s = f"{y_col.prefix}: {' '.join(row_strs)}"
        if aggrigate:
            plot_data.plot_agg(
                ax, axis_style, str_s, current_color, current_marker, current_linestyle
            )
        else:
            plot_data.plot(
                ax, axis_style, str_s, current_color, current_marker, current_linestyle
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
    filter_cols: list[tuple[Col, Any]] = [],
    filters: list[DataFilter] = [],
    subplots_adjust: None | dict[str, Any] = None,
    title: None | str = None,
    filename: None | Path = None,
    axis_style: AxesScale = "semilogx",
    aggrigate: bool = False,
):
    fig, ax = plt.subplots(figsize=(16, 12))
    if subplots_adjust:
        fig.subplots_adjust(**subplots_adjust)

    if title is not None:
        fig.suptitle(title)

    plot_on_axis(
        ax=ax,
        data=data,
        x_col=x_col,
        y_col=y_col,
        lines=lines,
        callbacks=callbacks,
        filter_cols=filter_cols,
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
    filter_cols: list[tuple[Col, Any]] = [],
    filters: list[DataFilter] = [],
    subplots_adjust: None | dict[str, Any] = None,
    title: None | str = None,
    filename: None | Path = None,
    axis_style: AxesScale = "semilogx",
    aggrigate: bool = False,
):
    nrows = len(filters)
    fig, axs = plt.subplots(nrows=nrows, figsize=(16, nrows * 8), sharex=True)
    if nrows == 1:
        axs = [axs]
    if subplots_adjust:
        fig.subplots_adjust(**subplots_adjust)

    if title is not None:
        fig.suptitle(title)

    for ii, filter in enumerate(filters):
        plot_on_axis(
            axs[ii],
            data,
            x_col,
            y_col,
            lines,
            callbacks,
            filter_cols,
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
            Col("data_size", "dt", ""),
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
            Col("data_size", "dt", ""),
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
            Col("data_size", "dt", ""),
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
            Col("data_size", "dt", ""),
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
        ],
        callbacks=[plot_x_sizes],
        title="write time vs chunk size",
        filename=base_dir / "time full.png",
        aggrigate=True,
    )
    plot_vertical(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("time", "", "write time (seconds)"),
        lines=[
            Col("data_size", "dt", ""),
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
        ],
        callbacks=[plot_x_sizes],
        filters=[
            DataFilter(run=accept_compressed),
            DataFilter(x_min=16 * 1024, run=accept_compressed),
        ],
        title="write time vs chunk size with varius compressions",
        filename=base_dir / "time compressed.png",
        aggrigate=True,
    )
    plot_vertical(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("time", "", "write time (seconds)"),
        lines=[
            Col("data_size", "dt", ""),
            Col("compression", "comp", ""),
            Col("filetype", "ft", ""),
        ],
        callbacks=[plot_x_sizes],
        filters=[
            DataFilter(run=accept_blosc),
            DataFilter(x_min=16 * 1024, run=accept_blosc),
        ],
        title="write time vs chunk size with blosc",
        filename=base_dir / "time blosc.png",
        aggrigate=True,
    )


def plot_size_and_time(
    data,
    base_dir: Path,
    filetype: None | str,
    compression: None | str,
    data_size: None | int,
    run_filter: FilterFunc = accept_all,
):
    lines: list[Col] = []
    if filetype is None:
        lines.append(Col("filetype", "dt", ""))
    if compression is None:
        lines.append(Col("compression", "dt", ""))
    lines.append(Col("memory_count", "mc", ""))
    if data_size is None:
        lines.append(Col("data_size", "dt", ""))

    filter_cols: list[tuple[Col, Any]] = []
    if compression is not None:
        filter_cols.append((Col("compression", "", ""), compression))
    if filetype is not None:
        filter_cols.append((Col("filetype", "", ""), filetype))
    if data_size is not None:
        filter_cols.append((Col("data_size", "", ""), data_size))
    filters = [
        DataFilter(run=run_filter),
        DataFilter(x_min=255 * 1024, run=run_filter),
        DataFilter(x_min=255 * 1024, run=lambda x: accept_int_mc(x) and run_filter(x)),
        DataFilter(
            x_min=255 * 1024,
            run=lambda x: x["memory_count"] >= 2 and run_filter(x),
        ),
    ]

    data_size_str = "all" if data_size is None else format_bytes(data_size)

    plot_vertical(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("time", "", "write time (seconds)"),
        lines=lines,
        callbacks=[plot_x_sizes],
        filter_cols=filter_cols,
        filters=filters,
        title=f"write time vs chunk size for {filetype} with {compression} on {data_size_str}",
        filename=base_dir / f"time {filetype} {compression} {data_size_str}.png",
        aggrigate=True,
    )

    plot_vertical(
        data,
        Col("chunk_size", "", "chunk size (bytes)"),
        Col("size", "", "file size"),
        lines=lines,
        callbacks=[plot_x_sizes, yaxis_format_bytes],
        filter_cols=filter_cols,
        filters=filters,
        title=f"file size vs chunk size for {filetype} with {compression} on {data_size_str}",
        filename=base_dir / f"size {filetype} {compression} {data_size_str}.png",
        aggrigate=True,
    )


def plot(data, base_dir):

    plot_size_vs_chunk_size(data, base_dir)
    plot_time_vs_chunk_size(data, base_dir)

    for filetype in [None, "deflate_zarr", "store_zarr", "hdf"]:
        for compression in [None, "blosc", "gzip-4"]:
            for data_size in [None, 64 * 1024 * 1024, 512 * 1024 * 1024]:
                if filetype is None:
                    run_filter = lambda x: x["filetype"] in ["store_zarr", "hdf"]
                else:
                    run_filter = accept_all

                plot_size_and_time(
                    data, base_dir, filetype, compression, data_size, run_filter
                )

    # plt.show()


def read() -> None:

    in_path = Path("C:/Workspace/data/out/little-benchmark/output_sizes.log")
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
                        szarr = (current_data["szarr_time"], current_data["szarr_size"])
                        dzarr = (current_data["dzarr_time"], current_data["dzarr_size"])

                        new_data = copy.copy(current_data)
                        del new_data["hdf_time"]
                        del new_data["hdf_size"]
                        del new_data["szarr_time"]
                        del new_data["szarr_size"]
                        del new_data["dzarr_time"]
                        del new_data["dzarr_size"]

                        hdf_data = new_data | dict(
                            time=hdf[0], size=hdf[1], filetype="hdf"
                        )
                        szarr_data = new_data | dict(
                            time=szarr[0], size=szarr[1], filetype="store_zarr"
                        )
                        dzarr_data = new_data | dict(
                            time=dzarr[0], size=dzarr[1], filetype="deflate_zarr"
                        )
                        results.append(FileResult(**hdf_data))
                        results.append(FileResult(**szarr_data))
                        results.append(FileResult(**dzarr_data))
                    else:
                        assert len(current_data) == 0

                    current_data = {}
                    filename = " ".join(parts[1:])
                    compression = filename.split(maxsplit=1)[0]
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

    for col in data.get_columns():
        vals = col.unique()
        print(f"{col.name}: {vals.shape}")

    # res = ic(
    #     data.filter(
    #         pl.col("memory_count") == 1.0,
    #         pl.col("compression") == "gzip-4",
    #         pl.col("chunk_size") == 1024 * 1024 * 64,
    #         pl.col("filetype") == "hdf",
    #     )
    # )
    #
    # for col in res.get_columns():
    #     vals = col.unique()
    #     print(f"{col.name}: {vals.shape}")

    plot(data, base_dir)
