# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

"""
This is the prescribe.py script for a simple example prescriptor that
generates IP schedules that trade off between IP cost and cases.

The prescriptor is "blind" in that it does not consider any historical
data when making its prescriptions.

The prescriptor is "greedy" in that it starts with all IPs turned off,
and then iteratively turns on the unused IP that has the least cost.

Since each subsequent prescription is stricter, the resulting set
of prescriptions should produce a Pareto front that highlights the
trade-off space between total IP cost and cases.

Note this file has significant overlap with ../random/prescribe.py.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from covid_xprize.examples.prescriptors.neat.utils import (
    CASES_COL, IP_COLS, IP_MAX_VALUES, PRED_CASES_COL, add_geo_id,
    get_predictions, prepare_historical_df)

NUM_PRESCRIPTIONS = 10

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def load_npi_importances(path=ROOT_DIR/'npi_importances.csv'):
    df = pd.read_csv(path, low_memory=False)

    return df

def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_hist_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:

    # Load historical IPs, just to extract the geos
    # we need to prescribe for.
    hist_df = pd.read_csv(path_to_hist_file,
                          parse_dates=['Date'],
                          encoding="ISO-8859-1",
                          keep_default_na=False,
                          error_bad_lines=True)

    # Load the IP weights, so that we can use them
    # greedily for each geo.
    weights_df = pd.read_csv(path_to_cost_file, keep_default_na=False)

    # Generate prescriptions
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    prescription_dict = {
        'PrescriptionIndex': [],
        'CountryName': [],
        'RegionName': [],
        'Date': []
    }
    for ip in IP_COLS:
        prescription_dict[ip] = []

    npi_importances = load_npi_importances()
    for country_name in hist_df['CountryName'].unique():
        country_df = hist_df[hist_df['CountryName'] == country_name]
        for region_name in country_df['RegionName'].unique():

            # Sort IPs for this geo by weight
            geo_weights_df = weights_df[(weights_df['CountryName'] == country_name) &
                                        (weights_df['RegionName'] == region_name)][IP_COLS]

            geo_id = country_name
            if region_name != '':
                geo_id = f'{geo_id} / {region_name}'
            geo_npi_importances = npi_importances.query(f'GeoID == "{geo_id}"').copy()
            geo_npi_importances.drop(columns='GeoID', inplace=True)

            ip_names = list(geo_weights_df.columns)
            ip_weights = geo_weights_df[ip_names].values[0]
            sorted_ips = [ip for _, ip in sorted(zip(ip_weights, ip_names))]

            ip_weights = ip_weights*geo_npi_importances[ip_names]
            ip_weights = 12*ip_weights.values[0]

            sorted_ips = [ip for _, ip in sorted(zip(ip_weights, ip_names))]

            # Initialize the IPs to all turned off
            curr_ips = {ip: 0 for ip in IP_MAX_VALUES}

            for prescription_idx in range(NUM_PRESCRIPTIONS):
                # Turn on the next IP
                next_ip = sorted_ips[prescription_idx]
                curr_ips[next_ip] = IP_MAX_VALUES[next_ip]

                # Use curr_ips for all dates for this prescription
                for date in pd.date_range(start_date, end_date):
                    prescription_dict['PrescriptionIndex'].append(prescription_idx)
                    prescription_dict['CountryName'].append(country_name)
                    prescription_dict['RegionName'].append(region_name)
                    prescription_dict['Date'].append(date.strftime("%Y-%m-%d"))
                    for ip in IP_COLS:
                        prescription_dict[ip].append(curr_ips[ip])

    # Create dataframe from dictionary.
    prescription_df = pd.DataFrame(prescription_dict)

    # Create the directory for writing the output file, if necessary.
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save output csv file.
    prescription_df.to_csv(output_file_path, index=False)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD."
                             "For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD."
                             "For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_past",
                        dest="prev_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans")
    parser.add_argument("-c", "--intervention_costs",
                        dest="cost_file",
                        type=str,
                        required=True,
                        help="Path to a .csv file containing the cost of each IP for each geo")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    print(f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    prescribe(args.start_date, args.end_date, args.prev_file, args.cost_file, args.output_file)
    print("Done!")
