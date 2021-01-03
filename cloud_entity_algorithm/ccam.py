"""
Cloud Cluster Analysis Module

This module contains functions for cloud cluster identification
and analysis
"""
import numpy as np
from scipy.ndimage import binary_dilation, label


def add_boundary_to_cloud_slice(c_slice):
    """
    Expand cloud slice by two rows/colums of zeros.

    Enables to calculate the gradient and therefor detect
    the edges of the field (1 extra row) and also in case
    the edge is directly on the original (c_slice) field.

    Note: in principle the extra columns are not necessary to
    add but that is the standard behaviour of np.pad

    >>> add_boundary_to_cloud_slice(np.array([[2,2,2],[2,2,2]]))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 2, 2, 2, 0, 0],
           [0, 0, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    """
    return np.pad(c_slice, 2, mode="constant")


def f_cloud_edges(c_slice, verbose=False):
    """
    Algorithm to detect cloud base and cloud top.

    For a layered cloud like
    array([[0., 3., 3., 3.],
       [0., 3., 3., 3.],
       [0., 0., 0., 0.],
       [3., 3., 3., 3.],
       [0., 3., 0., 3.]])

    the highest and lowest idx are returned.

    In case of cloud column gaps the returned index is "nan".

    >>> f_cloud_edges(np.array([[0., 3., 0., 3.],\
       [0., 3., 0., 3.],\
       [3., 3., 0., 3.],\
       [3., 3., 0., 3.],\
       [0., 3., 0., 3.]]))
    array([[ 3.,  4., nan,  4.],
           [ 2.,  0., nan,  0.]])
    """
    height, width = np.shape(c_slice)

    # Add boundary of zeros to calculate gradient on the edges
    bounded_cloud = add_boundary_to_cloud_slice(c_slice)
    # Calculate gradient/difference between single items along hgt-axis
    diff = np.diff(bounded_cloud, axis=0)
    # First maximum difference in column is the cloud top
    top_idx = np.nanargmax(diff, axis=0)[2:-2]  # exclude padding width(left and right)
    # Last minimum difference is the cloud base
    #  reverse index listing to find the last index easier
    base_idx_reverse = np.nanargmin(diff[::-1, :], axis=0)[
        2:-2
    ]  # counting from below to get lowest cbh in case of layer

    # Exclude indices which are 0 or height in the padded version
    #  these are non-cloudy columns
    base_idx_reverse = np.where(base_idx_reverse == 0, np.nan, base_idx_reverse)
    top_idx = np.where(top_idx == 0, np.nan, top_idx)

    base_idx = height - base_idx_reverse  # counting from the top
    top_idx = top_idx - 1

    if verbose:
        print("Cloud slice")
        print(c_slice)
        print("Added boundaries for easier calculation")
        print(bounded_cloud)
        print("Difference between column items for border detection")
        print(diff)
        print("Indices")
        print(base_idx, top_idx)

    return np.array([base_idx, top_idx])


def detect_precip(c_slice, lowest_idx, threshold=3):
    """

    Input
    lowest_idx: lowest height index of the original grid
        necessary to put the c_slice values into their
        context.
        As precip is present in the case the lowest two
        range gates show a return signal, the lowest two
        range gates of the c_slice by themselves could
        also be in the high troposphere. Therefore the lowest_idx
    """
    if lowest_idx < threshold:
        precip_idx = np.where(np.all(~np.isnan(c_slice[0:2, :]), axis=0), 1, 0)
        return precip_idx


def f_apply_cloudtype(label, dictionary):
    if np.isnan(label):
        return np.nan
    else:
        return dictionary[label]


def prepare_labels(labels):
    labels_0 = np.where(np.logical_or((np.isnan(labels)), (labels < 0)), 0, labels)
    unique_labels = np.unique(labels)[1:]
    return unique_labels, labels_0


def label_cluster(cluster_mask, stencil=np.ones((1, 1))):
    """
    Find coherent clusters.

    Method to find and label coherent clusters.
    The stencil can be used to count also masked points
    which are not direct neighbours to the cluster.
    """
    labels = label(binary_dilation(cluster_mask, structure=stencil))[0]
    return labels
