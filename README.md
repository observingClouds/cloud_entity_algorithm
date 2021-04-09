# Cloud Entity Algorithm
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4415834.svg)](https://doi.org/10.5281/zenodo.4415834)

This algorithm has been developed to detect individual clouds in radar time-series,
especially those captured at the
[Barbados Cloud Observatory](https://barbados.mpimet.mpg.de/).

Besides the identification of single cloud entities, a simple classification of the
cloud type as well as the calculation of additional parameters like *entity length*, *cloud base height*, *cloud top height* is also implemented.

![Algorithm Example](docs/images/cloud_classification_example.png?raw=true "Algorithm overview example")

## Installation
```sh
# Install latest version from pypi
pipenv install cloud-entity-algorithm

# or directly from github
pipenv install -e git+https://github.com/observingClouds/cloud_entity_algorithm.git
```
## Configuration
The metadata of output is mostly configured in [config/netcdf_templates.yaml](config/netcdf_templates.yaml).
Other configuration is given via command line arguments.

## Execution
The worker is `cloud_cluster_radar.py`. In the MPI compute environment, this script can be executed as:
```shell script
python cloud_cluster_radar.py -i /pool/OBS/BARBADOS_CLOUD_OBSERVATORY/Level_1/B_Reflectivity/Ka-Band/MBR2/10s/202010/*{date}??.nc -d 202010 -o ./Radar__BCO__{level}__{date}.nc
```
and creates files of different processing levels (`{level}`).

### Gridded reflectivities (level: Z_gridded)
This algorithm is written for two different Ka-Band radars (KATRIN and CORAL) whose range gates are not identical and 
can also change during the time of operation. To have a consistent height grid over the complete time-period the
data is regridded.

```
netcdf Radar__BCO__Z_gridded__202010 {
dimensions:
	time = 202009 ;
	range = 150 ;
variables:
	double time(time) ;
		time:_FillValue = NaN ;
		time:units = "seconds since 1970-01-01 00:00:00" ;
		time:calendar = "standard" ;
		time:axis = "T" ;
		time:standard_name = "time" ;
	double range(range) ;
		range:_FillValue = NaN ;
		range:units = "m" ;
	float Zf(time, range) ;
		Zf:_FillValue = NaNf ;
		Zf:units = "" ;

// global attributes:
        :title = "Radar reflectivity on equidistant monotonic increasing time grid" ;
        :author = "Hauke Schulz" ;
        :institute = "Max Planck Institute for Meteorology, Hamburg, Germany" ;
        :Conventions = "CF-1.6" ;
        string :source_files_used = "/pool/OBS/BARBADOS_CLOUD_OBSERVATORY/Level_1/B_Reflectivity/Ka-Band/MBR2/10s/202010/MMCR__MBR2__Spectral_Moments__10s__155m-18km__201008.nc", ...;
        :creation_date = "27/12/2020 19:02" ;
        :created_with = "cloud_cluster_radar.py with its last modification on Sun Dec 27 00:48:29 2020" ;
        :version = "--" ;
        :environment = "env:3.6.7 | packaged by conda-forge | (default, Feb 28 2019, 09:07:38) \n[GCC 7.3.0], numpy:1.18.1" ;
}
```

### Cloud entities (level: CloudEntity)
The actual cloud entities are created in this processing step. The cloud entities `label` have
the same dimension as the radar reflectivity `Zf`: (time, range) and therefore attribute
each measurement to a specific cloud entity.

```
netcdf Radar__BCO__CloudEntity__202010 {
dimensions:
	time = 202009 ;
	range = 150 ;
variables:
	int64 time(time) ;
		time:axis = "T" ;
		time:standard_name = "time" ;
		time:units = "seconds since 2020-10-08 14:51:50" ;
		time:calendar = "proleptic_gregorian" ;
	double range(range) ;
		range:_FillValue = NaN ;
		range:units = "m" ;
		range:standard_name = "height" ;
	double label(time, range) ;
		label:_FillValue = NaN ;
		label:stencil = "(1, 10)" ;
		label:reflectiviy_threshold = -55LL ;
		label:description = "Cloud entities created by applying a stencil to radar reflectivity field greater than defined threshold" ;
		label:units = "" ;
	int64 Z(time, range) ;
		Z:units = "" ;

// global attributes:
		:title = "Cloud entities on equidistant monotonic increasing time grid" ;
		:description = "Cloud labels derived from radar reflectivity smoothed before calculation by stencil to interconnect close-by clouds" ;
		:author = "Hauke Schulz" ;
		:institute = "Max Planck Institute for Meteorology, Hamburg, Germany" ;
		:Conventions = "CF-1.6" ;
		string :source_files_used = "/pool/OBS/BARBADOS_CLOUD_OBSERVATORY/Level_1/B_Reflectivity/Ka-Band/MBR2/10s/202010/MMCR__MBR2__Spectral_Moments__10s__155m-18km__201008.nc", ...;
		:creation_date = "27/12/2020 19:23" ;
		:created_with = "cloud_cluster_radar.py with its last modification on Sun Dec 27 00:48:29 2020" ;
		:version = "--" ;
		:environment = "env:3.6.7 | packaged by conda-forge | (default, Feb 28 2019, 09:07:38) \n[GCC 7.3.0], numpy:1.18.1" ;
}
```

### Cloud entity classification and parameters (level: CloudEntityInformation)
**Note: the methods to get the classification and parameters is very dependent on the
research question and should be handled with care and only used after studying the code.**

At this processing step the cloud type is identified based on its cloud base heights:
- cumulus gene (type=1; dark blue): Cloud base height below 1 km
- stratiform (type=2; light blue): Cloud base height between 1 km and 2.5 km
- none (type=4): Cloud base height above 2.5 km
- mixtures are classified, in case several of the above conditions are met for more than 10% of a
cloud entity e.g. cumulus (>10%) with stratiform layer (>10%) is cloud type 3 (neon-green)

These method is similar to Lamer et. al (2015) who classified individual clouds by measured at the BCO also according to their cloud base heights (CBH).

In addition, several parameters describing the cloud entities are calculated for convenience.
The output file also includes also the reflectivities and labels from the previous processing steps.
```
netcdf Radar__BCO__CloudEntityInformation__202010 {
dimensions:
	time = UNLIMITED ; // (202009 currently)
	entity = UNLIMITED ; // (4345 currently)
	range = 150 ;
variables:
	double entity(entity) ;
		entity:_FillValue = NaN ;
	float entity_start_time(entity) ;
		entity_start_time:_FillValue = NaNf ;
	float entity_stop_time(entity) ;
		entity_stop_time:_FillValue = NaNf ;
	float entity_cloud_type(entity) ;
		entity_cloud_type:_FillValue = NaNf ;
	float entity_len(entity) ;
		entity_len:_FillValue = NaNf ;
	float StSc_extent(entity) ;
		StSc_extent:_FillValue = NaNf ;
	float Cu_extent(entity) ;
		Cu_extent:_FillValue = NaNf ;
	float Cu_center_time(entity) ;
		Cu_center_time:_FillValue = NaNf ;
	float Cu_cores_nb(entity) ;
		Cu_cores_nb:_FillValue = NaNf ;
	float precip_extent(entity) ;
		precip_extent:_FillValue = NaNf ;
	float StSc_thickness_min(entity) ;
		StSc_thickness_min:_FillValue = NaNf ;
	float StSc_thickness_max(entity) ;
		StSc_thickness_max:_FillValue = NaNf ;
	float StSc_thickness_mean(entity) ;
		StSc_thickness_mean:_FillValue = NaNf ;
	float StSc_thickness_std(entity) ;
		StSc_thickness_std:_FillValue = NaNf ;
	float Other_occurance(entity) ;
		Other_occurance:_FillValue = NaNf ;
	double time(time) ;
		time:_FillValue = NaN ;
		time:axis = "T" ;
		time:standard_name = "time" ;
		time:units = "seconds since 1970-01-01" ;
		time:calendar = "standard" ;
	double range(range) ;
		range:_FillValue = NaN ;
		range:units = "m" ;
	float Zf(time, range) ;
		Zf:_FillValue = NaNf ;
		Zf:units = "" ;
	double label(range, time) ;
		label:_FillValue = NaN ;
		label:stencil = "(1, 10)" ;
		label:reflectiviy_threshold = -55LL ;
		label:description = "Cloud entities created by applying a stencil to radar reflectiviy field greater than defined threshold" ;
		label:units = "" ;
	double CBH(range, time) ;
		CBH:_FillValue = NaN ;
		CBH:description = "Cloud base height for each profile within a cloud entity." ;
	double CTH(range, time) ;
		CTH:_FillValue = NaN ;
		CTH:description = "Cloud top height for each profile within a cloud entity." ;

// global attributes:
		:author = "Hauke Schulz" ;
		:institute = "Max Planck Institute for Meteorology, Hamburg, Germany" ;
		:created_on = "27/12/2020 18:25 UTC" ;
		:created_with = "cloud_cluster_radar.py with its last modification on Sun Dec 27 00:48:29 2020" ;
		:version = "--" ;
		:environment = "env:3.6.7 | packaged by conda-forge | (default, Feb 28 2019, 09:07:38) \n[GCC 7.3.0], numpy:1.18.1" ;
		:source = "Radar__BCO__CloudEntity__202010.nc_delete created with version --" ;
		:Conventions = "CF-1.7" ;
```

## References
Lamer, K.; Kollias, P.; Nuijens, L. Observations of the Variability of Shallow Trade Wind Cumulus Cloudiness and Mass Flux. J. Geophys. Res. Atmos. 2015, 120 (12), 2014JD022950. https://doi.org/10.1002/2014JD022950.
