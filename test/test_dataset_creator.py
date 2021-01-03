import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from xarray.testing import assert_equal

import cloud_entity_algorithm._dataset_creator as dc

# Create random config files
cfg_1 = OmegaConf.create({})
cfg_2 = OmegaConf.create(
    {"global_attrs": {"title": "some awesome title", "author": "unknown"}}
)
cfg_3 = OmegaConf.create({"coordinates": {"time": {"dimension": 4}}})
cfg_4 = OmegaConf.create({"coordinates": {"time": {"dimension": [0, 2, 4, 6, 8]}}})
cfg_5 = OmegaConf.create(
    {
        "coordinates": {"time": {"dimension": [0, 2, 4, 6, 8]}},
        "variables": {"x": {"coordinates": ["time"]}},
    }
)
cfg_6 = OmegaConf.create(
    {
        "coordinates": {"time": {"dimension": [0, 2, 4, 6, 8]}},
        "variables": {"x": {"coordinates": ["time"], "attrs": {"units": ""}}},
    }
)
cfg_7 = OmegaConf.create(
    {"coordinates": {"time": {"dimension": 4, "attrs": {"units": "seconds"}}}}
)


def test_create_dataset():
    assert_equal(dc.create_dataset(cfg_1), xr.Dataset())
    ds = xr.Dataset()
    ds.attrs = {"global_attrs": {"title": "some awesome title", "author": "unknown"}}
    assert_equal(dc.create_dataset(cfg_2), ds)
    ds = xr.Dataset()
    ds = ds.assign_coords(time=range(4))
    assert_equal(dc.create_dataset(cfg_3), ds)
    ds = xr.Dataset(coords={"time": np.arange(0, 10, 2)})
    assert_equal(dc.create_dataset(cfg_4), ds)
    ds["x"] = xr.DataArray(None, coords=ds.coords)
    assert_equal(dc.create_dataset(cfg_5), ds)
    ds["x"].attrs["units"] = ""
    assert_equal(dc.create_dataset(cfg_6), ds)
    ds = xr.Dataset()
    ds = ds.assign_coords(time=range(4))
    ds["time"].attrs["units"] = "seconds"
    assert_equal(dc.create_dataset(cfg_7), ds)
