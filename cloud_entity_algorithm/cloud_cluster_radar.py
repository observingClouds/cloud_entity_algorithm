"""
Cloud entity analysis
"""
import argparse
import datetime
import glob
import logging
import os
import subprocess
import sys
from time import ctime

import _dataset_creator as dc  # noqa: E402
import _entity_helpers as ent_hp  # noqa: E402
import ccam  # noqa: E402
import cftime
import numpy as np
import scipy.ndimage as ndimage
import xarray as xr
from netCDF4 import num2date
from omegaconf import OmegaConf
from skimage.measure import regionprops
from tqdm import tqdm as tqdm

sys.path.append(".")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--inputfilefmt",
        metavar="/path/to/inputfile_{date}.nc",
        required=False,
        help="Provide filenamefmt of the input.",
        default="MMCR__MBR__Spectral_Moments__10s__155m-*km__{date}0?.nc",
    )
    parser.add_argument(
        "-o",
        "--outputfilefmt",
        metavar="/path/to/outputfile_{level}_{date}.nc",
        required=False,
        default="./Radar__BCO__{level}_{date}.nc",
        help="Provide filenamefmt of the output. {type} will"
        " will be replaced by the output type (monotonic,"
        " entity, ...",
    )

    parser.add_argument(
        "-d",
        "--date",
        metavar="YYYYMM",
        help="Provide the desired month, Format: YYYYMM",
        required=True,
        default=None,
    )

    parser.add_argument(
        "-s",
        "--cloud-stencil",
        metavar="1 10",
        help="Stencil connecting cloud pixels" "(height, time)",
        required=False,
        default=(1, 10),
        nargs=2,
        type=int,
    )

    parser.add_argument(
        "-t",
        "--cloud-threshold",
        metavar="-55",
        help="Reflectivity threshold (dBZ) for clouds. Lower"
        "values are ignored during labeling",
        required=False,
        default=-55,
        nargs=1,
        type=int,
    )

    parser.add_argument(
        "-r",
        "--max-range",
        metavar="150",
        help="Highest rangegate of radar that should" "still be included in analysis",
        required=False,
        default=150,
        nargs=1,
        type=int,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        metavar="DEBUG",
        help="Set the level of verbosity [DEBUG, INFO, WARNING, ERROR]",
        required=False,
        default="INFO",
    )

    args = vars(parser.parse_args())

    return args


def setup_logging(verbose):
    assert verbose in ["DEBUG", "INFO", "WARNING", "ERROR"]
    logging.basicConfig(
        level=logging.getLevelName(verbose),
        format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
        handlers=[logging.FileHandler(f"{__file__}.log"), logging.StreamHandler()],
    )


def get_git_version():
    """ Get git version of the current pwd"""
    try:
        version = (
            subprocess.check_output(["git", "describe", "--dirty"]).strip().decode()
        )
    except subprocess.CalledProcessError:
        version = "--"

    return version


git_module_version = get_git_version()
input_args = get_args()
setup_logging(input_args["verbose"])
script_name = os.path.basename(__file__)

# Settings
RESAMPLE = True
LABELING = True
ANALYSIS = True

cloud_threshold = input_args["cloud_threshold"]
max_hgt_idx = input_args["max_range"]
stencil_label = np.ones(input_args["cloud_stencil"])  # height, time
date_YYYYMM = input_args["date"]
date_YYMM = date_YYYYMM[2:]

# Read in config files
cfg_nc = OmegaConf.load("../config/netcdf_templates.yaml")
cfg_args = OmegaConf.create({"stencil_label": np.shape(stencil_label)})
cfg = OmegaConf.merge(cfg_nc, cfg_args)

outputfmt = input_args["outputfilefmt"]
filename_radar_monotonic = outputfmt.format(
    level="Z_gridded", date=date_YYYYMM
)  # f'MBR2__BCO__Zf__10s_monotonic_grid__201801.nc'
filename_label = outputfmt.format(
    level="CloudEntity", date=date_YYYYMM
)  # MBR2__BCO__Labels__10s_monotonic_grid__201801.nc'
filename_entity = outputfmt.format(
    level="CloudEntityInformation", date=date_YYYYMM
)  # 'MBR2__BCO__CloudEntityInformations__201801_.nc'

if RESAMPLE:
    logging.info("Resampling data")
    # Gather files
    input_radar_files = sorted(
        glob.glob(input_args["inputfilefmt"].format(date=date_YYMM))
    )  # data_dir+"MMCR__MBR__Spectral_Moments__10s__155m-*km__18010?.nc"))

    logging.info("Files found: {}".format(len(input_radar_files)))

    # Sort files by datestamp in filename
    #  because parameters of max height change in file
    #  name and mess up the time sorting
    timestamp_files = [file[-9:-3] for file in input_radar_files]

    # Delete days, where time is not monotonically increasing (duplicate instances; bug #1)
    idx_2_del = []
    for time in [
        "180413",
        "150608",
        "150727",
        "150728",
        "150811",
        "151026",
        "160318",
        "170309",
        "170615",
        "150730",
        "180108",
    ]:
        try:
            idx_2_del.append(timestamp_files.index(time))
        except ValueError:
            continue
        else:
            timestamp_files.remove(time)
    for idx in idx_2_del:
        del input_radar_files[idx]

    idx_sort = np.argsort(timestamp_files)

    # Open files
    xr.set_options(file_cache_maxsize=2)
    d = xr.open_mfdataset(np.array(input_radar_files)[idx_sort], combine="by_coords")

    Z = d["Zf"][:, :max_hgt_idx]
    ranges = d["range"][:max_hgt_idx]

    Z.load()

    # RESAMPLE
    Z_ = Z.resample(time="10S").nearest(tolerance="5S")  # .first(skipna=False)
    Z = Z_

    try:
        times_unix = np.int64(Z_.time.values) / 1e9
    except ValueError:
        times_unix = cftime.date2num(Z_.time.values, "seconds since 1970-1-1")
    Z.time.values = times_unix
    times = num2date(times_unix, "seconds since 1970-01-01")

    # Create new dataset
    runtime_cfg = OmegaConf.create(
        {
            "time_dimension": list([float(t) for t in Z.time]),
            "range_dimension": list([float(r) for r in Z.range]),
        }
    )
    ds = dc.create_dataset(OmegaConf.merge(runtime_cfg, cfg.datasets.resampled))
    ds["Zf"].data = Z
    attrs_dict = {
        "source_files_used": np.array(input_radar_files)[idx_sort],
        "creation_date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "created_with": script_name
        + " with its last modification on "
        + ctime(os.path.getmtime(os.path.realpath(__file__))),
        "version": git_module_version,
        "environment": "env:{}, numpy:{}".format(sys.version, np.__version__),
    }
    for attrs, val in attrs_dict.items():
        ds.attrs[attrs] = val

    # Export resampled dataset
    logging.info("File written to {}".format(filename_radar_monotonic))
    ds.to_netcdf(filename_radar_monotonic, encoding={"Zf": {"zlib": True}})

if LABELING:
    logging.info("Finding entities")

    # Label data
    Z_ = xr.open_dataset(filename_radar_monotonic)
    Zz = Z_["Zf"]

    # Apply threshold
    Z = np.where(Zz < cloud_threshold, np.nan, Zz)
    data = np.where(np.isnan(Z), 0, 1).T

    labels = ndimage.label(ndimage.binary_dilation(data, structure=stencil_label))[0]
    labels = np.where(labels == 0, np.nan, labels)
    labels_smoothed = labels

    # Apply labels to original (undilated) data
    labels_original = labels * np.asarray(data, dtype=bool)
    labels_original = np.where(
        labels_original == 0, np.nan, labels_original
    )  # remove zeros which are created by the 0-cloud-labels

    logging.info("Create label file {}".format(filename_label))
    runtime_cfg = OmegaConf.create(
        {
            "time_dimension": list([float(t) for t in Z_.time]),
            "range_dimension": list([float(r) for r in Z_.range]),
            "stencil_label": str(stencil_label.shape),
            "dBZ_threshold": cloud_threshold,
        }
    )
    ds_label = dc.create_dataset(OmegaConf.merge(runtime_cfg, cfg.datasets.labeled))
    attrs_dict = {
        "source_files_used": np.array(input_radar_files)[idx_sort],
        "creation_date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "created_with": script_name
        + " with its last modification on "
        + ctime(os.path.getmtime(os.path.realpath(__file__))),
        "version": git_module_version,
        "environment": "env:{}, numpy:{}".format(sys.version, np.__version__),
    }
    for attrs, val in attrs_dict.items():
        ds_label.attrs[attrs] = val

    ds_label["time"].data = Z_.time
    ds_label["range"].data = Z_.range
    ds_label["label"].data = labels_original.T
    ds_label["Z"].data = data.T

    ds_label.to_netcdf(filename_label)


if ANALYSIS:
    logging.info("Analyse cloud entities")

    def f_range(idx):
        """
        Return range gate for a provided range index
        """
        if np.isnan(idx):
            return np.nan
        else:
            return ranges[np.int(idx)]

    # Vectorize functions
    vf_range = np.vectorize(f_range, otypes=[np.float])
    vf_apply_cloudtype = np.vectorize(ccam.f_apply_cloudtype, otypes=[np.float])

    properties_str = [
        "cloud",
        "entity_start_time",
        "entity_stop_time",
        "entity_cloud_type",
        "entity_len",
        "StSc_extent",
        "Cu_extent",
        "Cu_center_time",
        "Cu_cores_nb",
        "precip_extent",
        "StSc_thickness_min",
        "StSc_thickness_max",
        "StSc_thickness_mean",
        "StSc_thickness_std",
        "Other_occurance",
    ]

    # Loading data
    labels_netCDF = xr.open_dataset(filename_label, decode_times=False)
    labels_netCDF.load()
    labels = labels_netCDF["label"].T
    data = labels_netCDF["Z"].T
    ranges = labels_netCDF["range"] / 1000
    times = num2date(labels_netCDF["time"] / 1e9, "seconds since 1970-01-01")

    # Create slices
    labels_0 = np.where(np.isnan(labels), 0, labels)  # find_objects cannot handle nan
    # labels_0 = np.where(labels < 0, 0, labels)  # find_objects cannot handle nan
    # ndimage.find_obejct finds peaces of the same label and returns
    #  the minimum dimension of the slice containing the labels
    cloud_slices = ndimage.find_objects(
        np.asarray(labels_0, dtype=int)
    )  # first part of slice height dim; second part time

    box_prop = regionprops(labels_0[:, :].astype(int))
    unique_labels = np.unique(labels_0)
    unique_labels = unique_labels[1:]

    # Cloud entity analysis
    cloud_prop = np.empty((len(unique_labels), 15))
    cloud_prop[:] = np.nan

    # Create arrays of the size (time, height) for detailed properties
    CBH = np.empty_like(labels, dtype=float)
    CTH = np.empty_like(labels, dtype=float)
    cloud_type = np.empty_like(labels, dtype=float)
    CBH[:, :] = np.nan
    CTH[:, :] = np.nan
    cloud_type[:, :] = np.nan

    for cloud in tqdm(unique_labels[0:]):

        #  def retrieve_cloud_information(cloud):
        entity_start_time = (
            entity_stop_time
        ) = (
            entity_cloud_type
        ) = (
            Cu_center_time
        ) = (
            StSc_thickness_min
        ) = StSc_thickness_max = StSc_thickness_mean = StSc_thickness_std = np.nan

        StSc_extent = Cu_extent = entity_len = Cu_cores_nb = precip_extent = 0

        box = box_prop[int(cloud) - 1]
        # Indices/Position within label file
        idx = box.coords.T  # (timeidx,hgt_idx)

        entity_start_time = np.int(Z_.time[idx[1].min()].values)
        entity_stop_time = np.int(Z_.time[idx[1].max()].values)
        entity_start_idx = idx[1].min()

        #  Disregard if cloud is too small (number of labels)
        #     if (len(idx[0]) < 20) & (len(idx[0])>30000):
        #         continue

        cloud_slice_idx = cloud_slices[
            int(cloud) - 1
        ]  # cloud slices and labels differ by one index
        cloud_slice = np.where(
            labels_0[cloud_slice_idx] == 0, np.nan, labels_0[cloud_slice_idx]
        )
        first_hgt_idx = cloud_slice_idx[0].start

        cbhs_idx = (
            ccam.f_cloud_edges(np.where(np.isnan(cloud_slice), 0, 1)) + first_hgt_idx
        )
        cloud_edges = vf_range(cbhs_idx)

        entity_label = box.label
        cbhs, cths = ent_hp.calculate_cbh_n_cth(cloud_edges, remove_nan=False)
        for ii, i in enumerate(range(idx[1].min(), idx[1].max())):
            CBH[idx[0][np.where(idx[1] == i)], i] = cbhs[ii]
            CTH[idx[0][np.where(idx[1] == i)], i] = cths[ii]
        cbhs = cbhs[~np.isnan(cbhs)]
        cths = cths[~np.isnan(cths)]

        entity_len = len(cbhs)
        entity_cloud_type, StSc_idx, Cu_idx, ND_len = ent_hp.estimate_cloud_type(
            cbhs, cths
        )
        StSc_extent = len(StSc_idx)
        Cu_extent = len(Cu_idx)

        if Cu_extent > 0:
            # Number of cumulus cores
            Cu_cores_nb = ent_hp.get_nb_Cu_cores(Cu_idx, threshold=1)

            # In case there is only one Cu, where would be its center (time)
            if Cu_cores_nb == 1:
                Cu_center_time = entity_start_idx + np.nanmean(Cu_idx).round(0).astype(
                    int
                )

            # Precipitation
            precip_extent = ent_hp.get_precip(data, idx)

        if StSc_extent > 0:
            (
                StSc_thickness_min,
                StSc_thickness_max,
                StSc_thickness_mean,
                StSc_thickness_std,
            ) = ent_hp.get_stsc_macrophysics(StSc_idx, cbhs, cths)

        # Write results to array
        properties = [
            cloud,
            entity_start_time,
            entity_stop_time,
            entity_cloud_type,
            entity_len,
            StSc_extent,
            Cu_extent,
            Cu_center_time,
            Cu_cores_nb,
            precip_extent,
            StSc_thickness_min,
            StSc_thickness_max,
            StSc_thickness_mean,
            StSc_thickness_std,
            ND_len,
        ]
        #     return properties
        for p, prop in enumerate(properties):
            cloud_prop[int(cloud) - 1, p] = prop

    # Write analysis data to Dataset
    cloud_data_xr = xr.Dataset({}, coords={"entity": cloud_prop[:, 0]})
    for v, variable in enumerate(properties_str[1:]):
        cloud_data_xr[variable] = ("entity", cloud_prop[:, v + 1].astype("float32"))

    # Merge analysis and label data
    cloud_data_merged = xr.merge([cloud_data_xr, Z_])
    cloud_data_merged["label"] = (["range", "time"], labels_original)
    cloud_data_merged["label"].attrs["stencil"] = str(stencil_label.shape)
    cloud_data_merged["label"].attrs["reflectiviy_threshold"] = cloud_threshold
    cloud_data_merged["label"].attrs["description"] = (
        "Cloud entities created by applying a stencil "
        "to radar reflectiviy field greater than defined "
        "threshold"
    )
    cloud_data_merged["label"].attrs["units"] = ""

    cloud_data_merged["CBH"] = (["range", "time"], CBH)
    cloud_data_merged["CBH"].attrs[
        "description"
    ] = "Cloud base height for each profile within a cloud entity."
    cloud_data_merged["CTH"] = (["range", "time"], CTH)
    cloud_data_merged["CTH"].attrs[
        "description"
    ] = "Cloud top height for each profile within a cloud entity."

    # Prepare netCDF metainformation
    cloud_data_merged.attrs["author"] = cfg.user.author_str
    cloud_data_merged.attrs["institute"] = cfg.user.institute
    cloud_data_merged.attrs["created_on"] = datetime.datetime.utcnow().strftime(
        "%d/%m/%Y %H:%M UTC"
    )
    cloud_data_merged.attrs["created_with"] = (
        script_name
        + " with its last modification on "
        + ctime(os.path.getmtime(os.path.realpath(__file__)))
    )
    cloud_data_merged.attrs["version"] = git_module_version
    cloud_data_merged.attrs["environment"] = "env:{}, numpy:{}".format(
        sys.version, np.__version__
    )
    cloud_data_merged.attrs["source"] = "{} created with version {}".format(
        filename_label, labels_netCDF.version
    )
    cloud_data_merged.attrs["Conventions"] = "CF-1.7"

    # Write analysis output
    cloud_data_merged.to_netcdf(
        filename_entity,
        encoding={
            "Zf": {"zlib": True},
            "label": {"zlib": True},
            "CTH": {"zlib": True},
            "CBH": {"zlib": True},
        },
        unlimited_dims=["time", "entity"],
    )

    logging.info("File {} successfully written".format(filename_entity))
