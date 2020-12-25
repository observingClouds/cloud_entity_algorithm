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
from collections import defaultdict
from time import ctime

import ccam
import cftime
import numpy as np
import scipy.ndimage as ndimage
import xarray as xr
from netCDF4 import Dataset, num2date
from skimage.measure import regionprops
from tqdm import tqdm as tqdm


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


try:
    git_module_version = (
        subprocess.check_output(["git", "describe", "--dirty"]).strip().decode()
    )
except subprocess.CalledProcessError:
    git_module_version = "--"

input_args = get_args()
setup_logging(input_args["verbose"])

# Settings
RESAMPLE = True
LABELING = True
ANALYSIS = True

cloud_threshold = input_args["cloud_threshold"]
max_hgt_idx = input_args["max_range"]
stencil_label = np.ones(input_args["cloud_stencil"])  # height, time
date_YYYYMM = input_args["date"]
date_YYMM = date_YYYYMM[2:]

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

    # Check and remove duplicate timesteps
    logging.info(
        "Non unique timestamps: {}".format(len(Z.time) - len(np.unique(Z.time)))
    )

    # and remove where they occur
    D = defaultdict(list)
    for i, item in enumerate(list(Z.time.values)):
        D[item].append(i)
    D_ = {k: v for k, v in D.items() if len(v) > 1}
    del D
    D = D_

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
    ds = xr.Dataset()
    ds["Zf"] = Z
    ds["time"].attrs["standard_name"] = "time"
    ds["time"].attrs["axis"] = "T"
    ds["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00 UTC"
    ds["time"].attrs["calendar"] = "gregorian"
    ds.attrs["author"] = "Hauke Schulz (hauke.schulz@mpimet.mpg.de)"
    ds.attrs["title"] = (
        "Radar reflectivity on equidistant monotonic" " increasing time grid"
    )
    ds.attrs["source_files_used"] = np.array(input_radar_files)[idx_sort]
    ds.attrs["creation_date"] = datetime.datetime.utcnow().strftime(
        "%d/%m/%Y %H:%M UTC"
    )
    ds.attrs["created_with"] = (
        os.path.basename(__file__)
        + " with its last modification on "
        + ctime(os.path.getmtime(os.path.realpath(__file__)))
    )
    ds.attrs["environment"] = "env:{}, numpy:{}".format(sys.version, np.__version__)
    ds.attrs["version"] = git_module_version
    ds.attrs["Conventions"] = "CF-1.7"

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
    labels_original = labels[np.asarray(data, dtype=bool)]
    labels_original = labels * np.asarray(data, dtype=bool)
    labels_original = np.where(
        labels_original == 0, np.nan, labels_original
    )  # remove zeros which are created by the 0-cloud-labels

    logging.info("Create label file {}".format(filename_label))
    nc = Dataset(filename_label, "w")
    dim_t = nc.createDimension("time", len(Z_.time))
    dim_r = nc.createDimension("range", len(Z_.range))

    var_t = nc.createVariable("time", np.float64, ("time",), zlib=True)
    var_t.units = "seconds since 1970-01-01 00:00:00"
    var_t.calendar = "standard"
    var_h = nc.createVariable("range", np.float, ("range",))
    var_h.description = "height"
    var_h.units = "m"
    var_label = nc.createVariable("label", np.int, ("range", "time"), zlib=True)
    var_label.stencil = str(stencil_label.shape)
    var_label.reflectiviy_threshold = cloud_threshold
    var_label.description = (
        "Cloud entities created by applying a stencil"
        "to radar reflectiviy field greater than defined"
        "threshold"
    )
    var_label.units = ""
    var_Z = nc.createVariable("Z", np.float, ("range", "time"), zlib=True)
    var_Z.units = ""

    nc.title = "Cloud entities on equidistant monotonic increasing time grid"
    nc.description = (
        "Cloud labels derived from radar reflectivity smoothed"
        "before calculation by stencil to interconnect close-by"
        "clouds"
    )
    nc.stencil_shape = str(stencil_label.shape)
    nc.source_files_used = np.array(input_radar_files)[idx_sort]
    nc.author = "Hauke Schulz (hauke.schulz@mpimet.mpg.de)"
    nc.institute = "Max Planck Institute for Meteorology, Hamburg, Germany"
    nc.creation_date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    nc.created_with = (
        os.path.basename(__file__)
        + " with its last modification on "
        + ctime(os.path.getmtime(os.path.realpath(__file__)))
    )
    nc.version = git_module_version
    nc.environment = "env:{}, numpy:{}".format(sys.version, np.__version__)

    var_t[:] = Z_.time
    var_h[:] = Z_.range
    var_label[:, :] = labels_original
    var_Z[:, :] = data

    nc.close()


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

    def estimate_cloud_type(cbhs, cths):
        """
        Estimate the cloud type of a cloud entity and
        for each profile within the entity (optional)
        and return the specific indices.
        """

        StSc_idx = np.where((cbhs > 1) & (cbhs < 3))[0]
        Cu_idx = np.where((cbhs > 0) & (cbhs < 1))[0]
        ND = sum(~np.isnan(cbhs)) - len(Cu_idx) - len(StSc_idx)

        possible_cloud_types = [-999, len(Cu_idx), len(StSc_idx), -999, ND]
        # Sort cloud_type by occurrence
        sorted_types = np.argsort(possible_cloud_types)[-2:]
        entity_cloud_type = 0
        for c_type in sorted_types:
            if possible_cloud_types[c_type] > 0.1 * np.sum(~np.isnan(cbhs)):
                entity_cloud_type += c_type
            else:
                continue

        return entity_cloud_type, StSc_idx, Cu_idx, ND

    def calculate_cbh_n_cth(cloud_edges, remove_nan=True):
        """
        Calculate cloud base height (cbh) and cloud
        top height (cth) for every single profile
        in the entities

        IMPORTANT
        ---------
        Check that the cth is calculated correctly
        """
        if remove_nan:
            cbhs = cloud_edges[1][~np.isnan(cloud_edges[1])]
            cths = cloud_edges[0][~np.isnan(cloud_edges[0])]
        else:
            cbhs = cloud_edges[1]
            cths = cloud_edges[0]

        return cbhs, cths

    def get_stsc_macrophysics(StSc_idx, cbhs, cths):
        """
        Calculate thickness, extent and other
        macrophysical paramters of the StSc part
        of the entity
        """
        # Thickness of each single StSc profile
        # within entity
        thicknesses = cths[StSc_idx] - cbhs[StSc_idx]

        return (
            np.min(thicknesses),
            np.max(thicknesses),
            np.mean(thicknesses),
            np.std(thicknesses),
        )

    def get_nb_Cu_cores(cu_idx, threshold=1):
        """
        Retrieve number of cumulus cores
        within the cloud entity

        Input
        -----
        threshold : defines, how many different cloud
            profiles other than Cu are allowed before
            the Cu core are splitted. Default is 1
            meaning no other cloud type between
            Cu profiles is allowed.
        >>> get_nb_Cu_cores([1,2,5,6,7,8,9,11,12,13],1)
        3
        """
        cu_cores = np.count_nonzero(np.diff(cu_idx) > threshold) + 1
        return cu_cores

    def get_precip(Z, idx):
        """
        Length of surface precip events

        Note
        ----
        This method do not care about different
        Cu cores, the complete time of rain
        underneath a entity is integrated.

        >>>get_precip(np.array([[np.nan,np.nan,np.nan],[0,10,0],[-50,-40,-30]]))
        1
        """
        # precip_len = np.count_nonzero(np.nanmean(Z[idx[0],0:3]) > -30)
        precip_len = np.count_nonzero(np.nanmean(Z[idx[0], 0:3], axis=1) > 0)
        return precip_len

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
    labels = labels_netCDF["label"]
    data = labels_netCDF["Z"]
    ranges = labels_netCDF["range"] / 1000
    times = num2date(labels_netCDF["time"] / 1e9, "seconds since 1970-01-01")

    # Create slices
    labels_0 = np.where(np.isnan(labels), 0, labels)  # find_objects cannot handle nan
    labels_0 = np.where(labels < 0, 0, labels)  # find_objects cannot handle nan
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
        cbhs, cths = calculate_cbh_n_cth(cloud_edges, remove_nan=False)
        for ii, i in enumerate(range(idx[1].min(), idx[1].max())):
            CBH[idx[0][np.where(idx[1] == i)], i] = cbhs[ii]
            CTH[idx[0][np.where(idx[1] == i)], i] = cths[ii]
        cbhs = cbhs[~np.isnan(cbhs)]
        cths = cths[~np.isnan(cths)]

        entity_len = len(cbhs)
        entity_cloud_type, StSc_idx, Cu_idx, ND_len = estimate_cloud_type(cbhs, cths)
        StSc_extent = len(StSc_idx)
        Cu_extent = len(Cu_idx)

        if Cu_extent > 0:
            # Number of cumulus cores
            Cu_cores_nb = get_nb_Cu_cores(Cu_idx, threshold=1)

            # In case there is only one Cu, where would be its center (time)
            if Cu_cores_nb == 1:
                Cu_center_time = entity_start_idx + np.nanmean(Cu_idx).round(0).astype(
                    int
                )

            # Precipitation
            precip_extent = get_precip(data, idx)

        if StSc_extent > 0:
            (
                StSc_thickness_min,
                StSc_thickness_max,
                StSc_thickness_mean,
                StSc_thickness_std,
            ) = get_stsc_macrophysics(StSc_idx, cbhs, cths)

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

    # cloud_data_merged['time'].attrs['units'] = 'seconds since 1970-01-01 00:00:00 UTC'
    # cloud_data_merged['time'].attrs['calendar'] = 'standard'

    # Prepare netCDF metainformation
    cloud_data_merged.attrs["author"] = "Hauke Schulz (hauke.schulz@mpimet.mpg.de)"
    cloud_data_merged.attrs[
        "institute"
    ] = "Max Planck Institute for Meteorology, Hamburg, Germany"
    cloud_data_merged.attrs["created_on"] = datetime.datetime.utcnow().strftime(
        "%d/%m%/%Y %H:%M UTC"
    )
    cloud_data_merged.attrs["created_with"] = (
        os.path.basename(__file__)
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
