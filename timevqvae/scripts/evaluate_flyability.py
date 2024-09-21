import argparse
import glob
import os
import pickle
import re
from typing import Any, Dict, List, Tuple

import cartopy.crs as ccrs
import cartopy.feature
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.basemap import Basemap
from traffic.core import Traffic

from timevqvae.evaluation.flyability_utils import (clean, discret_frechet, e_dtw, e_edr, e_erp,
                          e_hausdorff, e_lcss, e_sspd, frechet, s_dtw, s_edr,
                          s_erp, s_hausdorff, s_lcss, s_sspd, simulate)
from timevqvae.utils import extract_geographic_info

from traffic.data import airports
def extract_airport_coordinates(
    training_data_path: str,
) -> Tuple[float, float]:

    training_data = Traffic.from_file(training_data_path)

    if len(training_data.data["ADES"].unique()) > 1:
        raise ValueError("There are multiple destination airports in the training data")
    if len(training_data.data["ADES"].unique()) == 0:
        raise ValueError("There are no destination airports in the training data")

    ADES_code = training_data.data["ADES"].value_counts().idxmax()

    ADES_code = training_data.data["ADES"].value_counts().idxmax()
    ADES_lat = airports[ADES_code].latitude
    ADES_lon = airports[ADES_code].longitude
    return ADES_lat, ADES_lon

def get_longest_non_outlier_flight_duration(
    training_data_path: str, outlier_threshold: float = 5.0
) -> str:
    # Load the data
    training_data = Traffic.from_file(training_data_path)

    # Group data by flight_id and calculate the maximum duration for each flight
    flight_durations = training_data.groupby(["flight_id"]).agg(
        duration=("timedelta", "max")
    )
    flight_durations = flight_durations.sort_values(by="duration", ascending=False)

    # Calculate IQR and determine outliers
    Q1 = flight_durations["duration"].quantile(0.25)
    Q3 = flight_durations["duration"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + outlier_threshold * IQR

    # Filter out outliers
    non_outliers = flight_durations[flight_durations["duration"] <= upper_bound]

    # Select the longest duration that is not an outlier
    if not non_outliers.empty:
        longest_non_outlier_duration = non_outliers.iloc[0]["duration"]
    else:
        raise ValueError("All flights are outliers based on the current threshold.")

    # Convert the duration to a more readable format
    longest_duration_timedelta = pd.to_timedelta(longest_non_outlier_duration, unit="s")
    hours, remainder = divmod(longest_duration_timedelta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_duration = f"{hours:02}:{minutes:02}:{seconds:02}"

    # Print and return the formatted duration
    print(f"Longest non-outlier flight duration: {formatted_duration}")
    return formatted_duration


def get_most_common_ac_type(training_data_path: str) -> str:
    training_data = Traffic.from_file(training_data_path)

    # value counts of AC types
    v_c = training_data.data["AC Type"].value_counts()

    ac_types = training_data.groupby(["AC Type"]).agg(count=("AC Type", "count"))
    ac_types = ac_types.sort_values(by="count", ascending=False)

    most_common_ac_type = ac_types.index[0]
    print(f"Most common AC type: {most_common_ac_type}")
    return most_common_ac_type


def filter_simulated_traffic( simulated_trajectories: Traffic, training_data_path: str) -> Traffic:
    """
    filter simulated traffic data based on their proximity to the destination airport,
    i.e., remove redundant simulation points after the aircraft has landed at its destination.

    Args:
    simulated_trajectories (Traffic): Traffic object containing simulated trajectories.
    training_data_path (str): Path to training data to extract airport coordinates.

    Returns:
    Traffic: Filtered Traffic object containing simulated trajectories.
    """
    # Extract target airport coordinates
    ADES_lat, ADES_lon = extract_airport_coordinates(training_data_path)

    # Filter traffic based on minimum distance to the airport and remove all subsequent points

    simulated_trajectories.data["distance"] = np.sqrt(
        (simulated_trajectories.data["latitude"] - ADES_lat) ** 2
        + (simulated_trajectories.data["longitude"] - ADES_lon) ** 2
    )

    filtered_results = []
    for flight_id, group in simulated_trajectories.data.groupby("flight_id"):
        min_index = group["distance"].idxmin()
        filtered_results.append(group.loc[:min_index])

    filtered_df = pd.concat(filtered_results).reset_index(drop=True)

    return Traffic(filtered_df)



def run(training_data_path: str, synthetic_data_path: str) -> None:

    simulation_time = get_longest_non_outlier_flight_duration(training_data_path)
    # most_common_ac_type = get_most_common_ac_type(training_data_path)
    most_common_ac_type = "A319"
    # GenTrajs_list = load_and_edit_generated_trajectories(
    #     args.gen_dir, "TCVAE", most_common_ac_type
    # )
    generated_trajectories = Traffic.from_file(synthetic_data_path)

    #  traj.data["AC Type"] = AC_type
    generated_trajectories.data["AC Type"] = most_common_ac_type

    simulation_config = {
        "delta": 2000,
        "batch_size": 256,
        "early_batch_stop": False,
        "logs_directory": os.path.expanduser("~/bluesky/output/"),
        "simulation_time": simulation_time,
    }

    # SimuTrajs_list = simulate_traffic(GenTrajs_list, simulation_config)
    # SimTrajs_list.append(simulate(traffic, config))
    simulated_trajectories = simulate(generated_trajectories, simulation_config)
    clean()

    # SimuTrajs_list = filter_simulated_traffic(SimuTrajs_list, args.data_path)
    simulated_trajectories = filter_simulated_traffic(simulated_trajectories, training_data_path)

    # save simulated trajectories to the same path as the synthetic data with "simulated" appended to the filename
    simulated_data_path = synthetic_data_path.replace(".pkl", "_simulated.pkl")
    simulated_trajectories.to_pickle(simulated_data_path)



def main():
    parser = argparse.ArgumentParser(description="Evaluate the synthetic trajectories.")
    parser.add_argument("--dataset_file", type=str, help="Path to the training data file.")
    parser.add_argument("--synthetic_data_file", type=str, help="Path to the generated data file.")

    args = parser.parse_args()
    run(args.dataset_file, args.synthetic_data_file)

if __name__ == "__main__":
    main()

# example usage
# python evaluate_flyability.py --dataset_file /path/to/training_data.pkl --synthetic_data_file /path/to/generated_data.pkl