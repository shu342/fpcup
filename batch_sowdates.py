"""
Playing around with the PCSE implementation of WOFOST.
Based on the example notebook: https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb
"""
from pathlib import Path

data_dir = Path("../pcse_notebooks/data")
output_dir = Path.cwd() / "outputs" / "sowdates"
results_dir = Path.cwd() / "results"

from datetime import datetime
from itertools import product

import yaml

import pcse
from pcse.base import ParameterProvider
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider
from pcse.util import WOFOST72SiteDataProvider

import fpcup

print(f"PCSE version: {pcse.__version__}")

cropd = YAMLCropDataProvider()
soil_dir = data_dir / "soil"
soil_files = [CABOFileReader(soil_filename) for soil_filename in soil_dir.glob("ec*")]
sited = WOFOST72SiteDataProvider(WAV=10)

agro = """
- {date:%Y}-01-01:
    CropCalendar:
        crop_name: 'barley'
        variety_name: 'Spring_barley_301'
        crop_start_date: {date:%Y-%m-%d}
        crop_start_type: sowing
        crop_end_date:
        crop_end_type: maturity
        max_duration: 300
    TimedEvents: null
    StateEvents: null
- {date:%Y}-12-01: null
"""
crop_type = "barley"

weatherdata = fpcup.weather.load_weather_data_NASAPower(latitude=5, longitude=53, return_single=False)

# Sowing dates to simulate
year = 2020
doys = range(1, 222)
sowing_dates = [datetime.strptime(f"{year}-{doy}", "%Y-%j") for doy in doys]

# Set up iterables
sitedata = [sited]
soildata = soil_files
cropdata = [cropd]

parameters_combined = [ParameterProvider(sitedata=site, soildata=soil, cropdata=crop) for site, soil, crop in product(sitedata, soildata, cropdata)]
agromanagementdata = [yaml.safe_load(agro.format(date=date)) for date in sowing_dates]

# Loop over input data
all_runs = product(parameters_combined, weatherdata, agromanagementdata)
nruns = len(parameters_combined) * len(weatherdata) * len(agromanagementdata)
print(f"Number of runs: {nruns}")
# (this does not work when the inputs are all generators)

# Run the simulation ensemble
outputs, summary = fpcup.run_pcse_ensemble(all_runs, nr_runs=nruns)

# Write the summary results to a CSV file
fpcup.io.save_ensemble_summary(summary, output_dir / "summary.csv")

# Write the individual outputs to CSV files
fpcup.io.save_ensemble_results(outputs, output_dir)
