import glob
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from cartopy.crs import EuroPP
from scipy.stats import zscore
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from traffic.core import Flight, Traffic
from traffic.data import airports

from timevqvae.utils import (calculate_consecutive_distances,
                             calculate_final_distance,
                             calculate_initial_distance)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in  
 decimal degrees) using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.   

        lon2 (float): Longitude of the second point.

    Returns:
        float: The distance between the two points in kilometers.  

    """

    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)  

    lon2_rad = np.radians(lon2)  


    # Differences in latitude and longitude
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1  
 - a))

    # Distance in kilometers
    distance = R * c
    return distance 



def assign_flight_ids(opensky_data: pd.DataFrame, window: int = 6) -> pd.DataFrame:

    # Create a unique key for each (icao24, callsign) combination
    opensky_data['key'] = opensky_data['icao24'] + '_' + opensky_data['callsign']

    # Calculate the time difference between consecutive rows with the same key
    time_diff = opensky_data.groupby('key')['timestamp'].diff().dt.total_seconds() / 3600

    # Create a new group when the time difference exceeds the window
    group = (time_diff > window).cumsum()

    # Generate flight IDs
    opensky_data['flight_id'] = (
        opensky_data['key'] + '_' +
        opensky_data.groupby(['key', group])['timestamp'].transform('first').dt.strftime('%Y%m%d_%H%M%S')
    )

    # Drop the temporary 'key' column
    opensky_data = opensky_data.drop('key', axis=1)

    # print the number of unique flight ids
    num_flights = opensky_data["flight_id"].nunique()
    print(f"Number of unique flight ids: {num_flights}")
    print(f"Number of rows: {len(opensky_data)}")
    return opensky_data

def remove_outliers(
    opensky_data: pd.DataFrame, thresholds: List[float], ADES_code: str
) -> Tuple[pd.DataFrame, float]:

    # print the number of unique flight ids
    num_flights = opensky_data["flight_id"].nunique()
    print(f"Number of unique flight ids before removing outliers: {num_flights}")

    def find_outliers_zscore(df, column, threshold=2.5):
        # Calculate z-scores
        df["z_score"] = zscore(df[column])

        # Filter and return outlier rows
        outliers = df[df["z_score"].abs() > threshold]
        return outliers.drop(columns="z_score")

    (
        consecutive_distance_threshold,
        altitude_threshold,
        lowest_sequence_length_threshold,
    ) = thresholds

    consecutive_distance_outliers = calculate_consecutive_distances(
        opensky_data, distance_threshold=consecutive_distance_threshold
    )
    print(
        f"Found {len(consecutive_distance_outliers)} flights with excessive consecutive distances."
    )


    # ADES_code = opensky_data["ADES"].value_counts().idxmax()
    ADES_lat_lon = airports[ADES_code].latlon
 
    final_distance_outliers = calculate_final_distance(
        opensky_data, ADES_lat_lon, distance_threshold=10
    )
    print(
        f"Found {len(final_distance_outliers)} flights with excessive final distances."
    )
    print(
        f"Number of unique flight ids in final distance outliers that are in consecutive distance outliers: {len(set(final_distance_outliers).intersection(set(consecutive_distance_outliers)))}"
    )


    altitude_outliers = find_outliers_zscore(
        opensky_data, "altitude", threshold=altitude_threshold
    )
    print(
        f"Found {len(altitude_outliers)} outliers in column 'altitude', with threshold {altitude_threshold}"
    )
    print(altitude_outliers[["flight_id", "altitude"]])
    # print(altitude_outliers['flight_id'].unique())
    print(
        f"Number of unique flight ids in altitude outliers: {altitude_outliers['flight_id'].nunique()}\n"
    )

    # drop rows with altitude outliers
    print("Dropping rows with altitude outliers...")
    opensky_data = opensky_data.drop(altitude_outliers.index).reset_index(drop=True)

    # drop flights with consecutive distance outliers
    print("Dropping flights with consecutive distance outliers...")
    opensky_data = opensky_data[
        ~opensky_data["flight_id"].isin(consecutive_distance_outliers)
    ]


    final_distance_outliers = [
        flight_id
        for flight_id in final_distance_outliers
        if flight_id not in consecutive_distance_outliers
  
    ]
    print("Dropping flights with final distance outliers...")
    opensky_data = opensky_data[
        ~opensky_data["flight_id"].isin(final_distance_outliers)
    ]

    # reset the index
    opensky_data = opensky_data.reset_index(drop=True)

    # find the average number of rows in each flight with unique flight_id
    avg_sequence_length = opensky_data.groupby("flight_id").size().mean()
    # make it even number (for Fourier transform)
    avg_sequence_length = int(avg_sequence_length) if int(avg_sequence_length) % 2 == 0 else int(avg_sequence_length) - 1

    # count the number of rows in each flight with unique flight_id, and make it a dataframe
    size = opensky_data.groupby("flight_id").size().reset_index(name="counts")

    # calculate z-scores for the counts
    size["z_score"] = zscore(size["counts"])

    # drop flights with lowest sequence length
    low_counts_outliers = size[size["z_score"] < lowest_sequence_length_threshold]
    print(
        f"Found {len(low_counts_outliers)} outliers in column 'counts', with threshold {lowest_sequence_length_threshold}"
    )

    # drop the low counts outliers
    opensky_data = opensky_data[
        ~opensky_data["flight_id"].isin(low_counts_outliers["flight_id"])
    ]
    # reset the index
    opensky_data = opensky_data.reset_index(drop=True)

    # remove flights with duplicate rows of the same timestamp. To address: ValueError: cannot reindex on an axis with duplicate labels
    duplicate_rows = opensky_data[
        opensky_data.duplicated(subset=["flight_id", "timestamp"], keep=False)
    ]
    duplicate_flights = duplicate_rows["flight_id"].unique()
    print(f"Found {len(duplicate_flights)} flights with duplicate rows")
    opensky_data = opensky_data[~opensky_data["flight_id"].isin(duplicate_flights)]

    return opensky_data, avg_sequence_length



def load_OpenSky_flights_points(
    base_path: str,  ADES_code: str
) -> pd.DataFrame:

    # look for any .csv files in the base_path
    files = glob.glob(os.path.join(base_path, "*.csv"))
    print(f"Found {len(files)} files in the directory: {base_path}")
    print(files)

    # select only the files that contain the ADEP and ADES codes: e.g. opensky_EHAM_LIMC_2019-01-01_2023-01-01.csv
    print(f"Looking for files with ADES code: {ADES_code}")
    files = [file for file in files if ADES_code in file]
    print(
            f"Found {len(files)} files with ADES codes in the directory: {base_path}"
        )
        # select the first file
    file = files[0]  # TODO: process all files

    print(f"Processing file: {file}")



    distance_threshold = 100
    # Initialize an empty DataFrame to store the final result
    landing_data_total = pd.DataFrame()

    # Set the chunk size
    chunk_size = 15000000  # Adjust based on memory capacity
    for chunk in pd.read_csv(file, chunksize=chunk_size):
        processed_chunk = process_chunk(chunk, ADES_code, distance_threshold)
        landing_data_total = pd.concat([landing_data_total, processed_chunk])


    print("Got landing data, now removing outliers...")

    # remove nan values
    landing_data_total.dropna(inplace=True)


    # remove outliers
    landing_data, avg_sequence_length = remove_outliers(
        landing_data_total, thresholds=[50, 2.2, -1], ADES_code=ADES_code
    )  # [consecutive_distance_threshold, altitude_threshold, lowest_sequence_length_threshold]
    print("Removed outliers, now getting trajectories...")
    return landing_data, avg_sequence_length




def process_chunk(chunk, ADES_code, distance_threshold):
    # Drop unnecessary column and rows with NaN values
    chunk = chunk.dropna()
    
    # Keep rows with non-negative altitude values
    chunk = chunk.query("altitude >= 0")
    
    # Rename columns
    # chunk = chunk.rename(columns={"estdepartureairport": "ADEP", "estarrivalairport": "ADES"})
    
    # Convert 'timestamp' column to datetime
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
    chunk.sort_values("timestamp", inplace=True)
    
    # Assign flight ids
    chunk = assign_flight_ids(chunk, window=6)
    
    # Get landing data
    landing_data = get_landing_data(chunk, ADES_code, distance_threshold)
    
    return landing_data

def get_landing_data(opensky_data: pd.DataFrame, ADES_code: str, distance_threshold: int) -> pd.DataFrame:

    # get the latitude and longitude of the destination airport
    ADES_lat_lon = airports[ADES_code].latlon

    # calculate the distance between the destination airport and each point in the open sky data
    opensky_data["distance_to_destination"] = opensky_data.apply(
        lambda row: haversine_distance(row["latitude"], row["longitude"], ADES_lat_lon[0], ADES_lat_lon[1]), axis=1
    )
    landing_data = opensky_data[opensky_data["distance_to_destination"] <= distance_threshold]


    return landing_data



def get_trajectories(flights_points: pd.DataFrame) -> Traffic:


    flights_points.loc[:, "timestamp"] = pd.to_datetime(
        flights_points["timestamp"], format="%d-%m-%Y %H:%M:%S", utc=True
    )

    grouped_flights = flights_points.groupby("flight_id")
    flights_list = [Flight(group) for _, group in grouped_flights]
    trajectories = Traffic.from_flights(flights_list)



    return trajectories


def prepare_trajectories(
    trajectories: Traffic, n_samples: int, n_jobs: int, douglas_peucker_coeff: float
) -> Traffic:

    if douglas_peucker_coeff is not None:
        print("Simplification...")
        trajectories = trajectories.simplify(tolerance=1e3).eval(desc="")

    trajectories = Traffic.from_flights(
        flight.assign(
            timedelta=lambda r: (r.timestamp - flight.start).apply(
                lambda t: t.total_seconds() if t.total_seconds() != 0 else 0.001  # Assign small value if timedelta is 0
            )
        )
        for flight in trajectories
    )


    # clustering
    print("Clustering...")
    np.random.seed(
        199
    )  
    nb_samples = (n_samples -1) if n_samples < 1000 else 1000
    print(f"Number of samples for clustering: {nb_samples}, Number of samples for resampling: {n_samples}")

    trajectories = trajectories.clustering(
        nb_samples=nb_samples,
        projection=EuroPP(),
        features=["latitude", "longitude"],
        clustering=GaussianMixture(n_components=5),
        transform=StandardScaler(),
    ).fit_predict()

    # Resample trajectories for uniformity
    print("Resampling...")
    trajectories = (
        trajectories.resample(n_samples).unwrap().eval(max_workers=n_jobs, desc="resampling")
    )
    return trajectories


def get_args() -> ArgumentParser:

    parser = ArgumentParser()
    parser.add_argument("--ADES", dest="ADES", type=str, default="ENGM")
    parser.add_argument(
        "--raw_data_dir", dest="base_path", type=str, default="../raw_data"
    )
    # save directory for the prepared trajectories
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="./data/real/")

    args = parser.parse_args()
    return args


def main():
    args = get_args()


    landing_data, avg_sequence_length = load_OpenSky_flights_points(
        os.path.join(args.base_path, "landing"),
        args.ADES
    )

    # Create Traffic object from flight points
    trajectories = get_trajectories(landing_data)
    del landing_data

    # Prepare trajectories for training
    trajectories = prepare_trajectories(
        trajectories, int(avg_sequence_length), n_jobs=7, douglas_peucker_coeff=None
    )

    # Save the prepared trajectories to a pickle file
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir, f"landing_{args.ADES}.pkl"
    )
    trajectories.to_pickle(Path(save_path))
    print(f"Saved trajectories to {save_path}")


if __name__ == "__main__":
    main()
