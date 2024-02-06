"""
Run PCSE for plots within the BRP.
"""
from tqdm import tqdm
import datetime as dt

import fpcup

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run PCSE for plots within the BRP.")
parser.add_argument("brp_filename", help="file to load the BRP from", type=fpcup.io.Path)
parser.add_argument("-c", "--crop", help="crop to run simulations on (or all)", default="All", choices=("barley", "maize", "wheat"), type=str.lower)
parser.add_argument("-p", "--province", help="crop to run simulations on (or all)", default="All", choices=fpcup.province.province_names+["All"], type=str.title)
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=fpcup.io.Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-o", "--output_dir", help="folder to save PCSE outputs to", type=fpcup.io.Path, default=None)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Get the year from the BRP filename
year = int(args.brp_filename.stem.split("_")[-1].split("-")[0])

# Set up a default output folder if a custom one was not provided
if args.output_dir is None:
    args.output_dir = fpcup.settings.DEFAULT_OUTPUT / f"brp{year}-{args.crop}"

    if args.verbose:
        print(f"Default save folder: {args.output_dir.absolute()}")

# Make the output folder if it does not exist yet
fpcup.io.makedirs(args.output_dir, exist_ok=True)

# Load the BRP file
brp = fpcup.io.read_gpd(args.brp_filename)
if args.verbose:
    print(f"Loaded BRP data from {args.brp_filename.absolute()} -- {len(brp)} sites")

# If we are only doing one crop, select only the relevant lines from the BRP file
if args.crop != "All":
    brp = brp.loc[brp["crop_species"] == args.crop]

    if args.verbose:
        print(f"Selected only plots growing {args.crop} -- {len(brp)} sites")

# If we are only doing one province, select only the relevant lines from the BRP file
if args.province != "All":
    if args.province == "Friesland":
        args.province = "Fryslân"
    brp = brp.loc[brp["province"] == args.province]

    if args.verbose:
        print(f"Selected only plots in {args.province} -- {len(brp)} sites")

# Pre-load data (to be improved)
soil_dir = args.data_dir / "soil"
soildata = fpcup.soil.load_folder(soil_dir)[0]
cropdata = fpcup.crop.default

sowdate = dt.date(year, 2, 1)
agromanagementdata = fpcup.agro.load_formatted(fpcup.agro.template_springbarley_date, date=sowdate)

outputs = []
# Run the simulations (minimum working example)
for i, row in tqdm(brp.iterrows(), total=len(brp), desc="Running PCSE", unit="plot"):
    coords = (row["longitude"], row["latitude"])

    # Fetch site data
    sitedata = fpcup.site.example(coords)

    # Fetch weather data
    weatherdata = fpcup.weather.load_weather_data_NASAPower(coords)

    # Bundle parameters
    params_agro = fpcup.model.ParameterProvider(sitedata=sitedata, soildata=soildata, cropdata=cropdata)
    run = (params_agro, weatherdata, agromanagementdata)

    # Run model
    output = fpcup.model.run_pcse_single(run, run_id_function=lambda *args: str(i))

    outputs.append(output)

summaries_individual = [o.summary for o in outputs]
outputs, summaries_individual, n_filtered_out = fpcup.model.filter_ensemble_outputs(outputs, summaries_individual)
summary = fpcup.model.Summary(summaries_individual)

# Write the summary results to a CSV file
summary_filename = args.output_dir / "summary.csv"
fpcup.io.save_ensemble_summary(summary, summary_filename)
if args.verbose:
    print(f"Saved ensemble summary to {summary_filename.absolute()}")

# Write the individual outputs to CSV files
fpcup.io.save_ensemble_results(outputs, args.output_dir)
if args.verbose:
    print(f"\nSaved individual ensemble results to {args.output_dir.absolute()}")
