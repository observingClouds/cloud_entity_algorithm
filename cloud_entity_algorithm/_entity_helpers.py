"""Entity helpers"""
import numpy as np


def estimate_cloud_type(cbhs, cths):
    """Estimate the cloud type of a cloud entity and
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
