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


def load_flights_points(
    file_flights: str, flight_points_file_path: str, ADEP_code: str, ADES_code: str
) -> pd.DataFrame:
    # Load flights data from a CSV file
    flights_df = pd.read_csv(file_flights)

    # Filter flights originating from ADEP_code and destined for ADES_code
    flights = flights_df[
        (flights_df["ADEP"] == ADEP_code) & (flights_df["ADES"] == ADES_code)
    ]
    print(f"Number of flights from {ADEP_code} to {ADES_code}: {len(flights)}")

    # Load flight points data from another CSV file
    flight_points = pd.read_csv(flight_points_file_path)

    # Merge data to only include flight points for the filtered flights
    flights_points = flight_points[flight_points["ECTRL ID"].isin(flights["ECTRL ID"])]

    # Add relevant flight information to the flight points
    flights_points = flights_points.merge(
        flights[["ECTRL ID", "ADEP", "ADES", "AC Type"]], on="ECTRL ID"
    )

    # Calculate the average sequence length of the flights
    sequence_lengths = flights_points.groupby("ECTRL ID").size()
    avg_sequence_length = sequence_lengths.mean()
    print(f"Average sequence length: {avg_sequence_length}")

    return flights_points


def load_EuroControl_flights_points(
    base_path: str, ADEP_code: str, ADES_code: str
) -> pd.DataFrame:
    directories = glob.glob(os.path.join(base_path, "20????"))

    all_flights_points = []

    for directory in directories:
        print(f"\nProcessing directory: {directory}")
        file_flights = glob.glob(os.path.join(directory, "Flights_*.csv"))[0]
        flight_points_file_path = glob.glob(
            os.path.join(directory, "Flight_Points_Actual_*.csv")
        )[0]
        print(
            f"file_flight: {file_flights} ; flight_points_file_path: {flight_points_file_path}"
        )
        flights_points = load_flights_points(
            file_flights, flight_points_file_path, ADEP_code, ADES_code
        )

        if not flights_points.empty:
            all_flights_points.append(flights_points)
        else:
            print("No data to process in this directory.")

    all_flights_points = pd.concat(all_flights_points, ignore_index=True)

    print("***************** Done processing all directories *****************")

    # Remove flights with duplicate rows of the same timestamp
    duplicate_rows_df = all_flights_points[
        all_flights_points.duplicated(subset=["ECTRL ID", "Time Over"])
    ]
    duplicate_ids = duplicate_rows_df["ECTRL ID"].unique()
    print(f"Number of duplicate rows: {len(duplicate_rows_df)}")

    all_flights_points = all_flights_points[
        ~all_flights_points["ECTRL ID"].isin(duplicate_ids)
    ]

    # callsign equals to the ECTRL ID
    all_flights_points["callsign"] = all_flights_points["ECTRL ID"]
    # icas24 equals to the ECTRL ID
    all_flights_points["icao24"] = all_flights_points["ECTRL ID"]

    # Calculate the average sequence length of all flights
    sequence_lengths = all_flights_points.groupby("ECTRL ID").size()
    avg_sequence_length = sequence_lengths.mean()
    # make sure that the sequence length is an even integer: since fourier transform return even sequence length !!
    avg_sequence_length = (
        int(avg_sequence_length)
        if int(avg_sequence_length) % 2 == 0
        else int(avg_sequence_length) - 1
    )
    print(f"Average sequence length of all flights: {avg_sequence_length}")

    # Rename columns for consistency and clarity in the Traffic library
    all_flights_points = all_flights_points.rename(
        columns={
            "ECTRL ID": "flight_id",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Flight Level": "altitude",
            "Time Over": "timestamp",
        }
    )

    # Convert flight level to altitude in feet
    all_flights_points["altitude"] = all_flights_points["altitude"] * 100

    # Drop unnecessary column and reset index
    all_flights_points = all_flights_points.drop(
        columns=["Sequence Number"]
    ).reset_index(drop=True)

    return all_flights_points, avg_sequence_length


# def assign_flight_ids(opensky_data: pd.DataFrame, window: int = 6) -> pd.DataFrame:

#     # Initialize the flight_id column and a dictionary to track the last time per (icao24, callsign) df['flight_id'] = None
#     opensky_data["flight_id"] = None
#     last_flight_times = {}

#     # Function to determine flight id based on past flight times and a 6-hour window
#     def assign_flight_id_fn(row, window=window):
#         key = (row["icao24"], row["callsign"])
#         current_time = row["timestamp"]
#         if (
#             key in last_flight_times
#             and (current_time - last_flight_times[key]["time"]).total_seconds() / 3600
#             <= window
#         ):
#             # If within 6 hours of the last flight with the same icao24 and callsign, use the same flight id
#             return last_flight_times[key]["flight_id"]
#         else:
#             # Otherwise, create a new flight id
#             formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
#             # new_flight_id = f"{row['icao24']}_{row['callsign']}_{current_time.isoformat()}"
#             new_flight_id = f"{row['icao24']}_{row['callsign']}_{formatted_time}"
#             last_flight_times[key] = {"time": current_time, "flight_id": new_flight_id}
#             return new_flight_id

#     # Apply the function to each row df['flight_id'] = df.apply(assign_flight_id, axis=1)
#     opensky_data["flight_id"] = opensky_data.apply(assign_flight_id_fn, axis=1)

#     return opensky_data


# def assign_flight_ids(opensky_data: pd.DataFrame, window: int = 6) -> pd.DataFrame:

#     # Sort data by timestamp to ensure correct processing order
#     opensky_data = opensky_data.sort_values("timestamp")

#     # Create a unique identifier for each (icao24, callsign) combination
#     opensky_data['flight_key'] = opensky_data['icao24'] + '_' + opensky_data['callsign'].astype(str)

#     # Group by the unique identifier and calculate time differences between consecutive rows
#     grouped = opensky_data.groupby('flight_key')
#     opensky_data['time_diff'] = grouped['timestamp'].diff().dt.total_seconds() / 3600

#     # Create a new flight ID whenever the time difference exceeds the window or it's the first row in a group
#     opensky_data['new_flight'] = (opensky_data['time_diff'] > window) | opensky_data['time_diff'].isna()
#     opensky_data['flight_group'] = opensky_data.groupby('flight_key')['new_flight'].cumsum()

#     # Generate unique flight IDs based on the flight group and timestamp
#     opensky_data['flight_id'] = opensky_data['flight_key'] + '_' + opensky_data['flight_group'].astype(str) + '_' + opensky_data['timestamp'].dt.strftime('%Y%m%d_%H%M%S')

#     # Drop temporary columns
#     opensky_data = opensky_data.drop(columns=['flight_key', 'time_diff', 'new_flight', 'flight_group'])

#     return opensky_data


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
    opensky_data: pd.DataFrame, thresholds: List[float]
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

    ADEP_code = opensky_data["ADEP"].value_counts().idxmax()
    ADES_code = opensky_data["ADES"].value_counts().idxmax()
    ADEP_lat_lon = airports[ADEP_code].latlon
    ADES_lat_lon = airports[ADES_code].latlon
    # # 48.1179° N, 16.5663° E
    # # 51.4680° N, 0.4551° W
    # ADEP_lat_lon = (48.1179, 16.5663)
    # ADES_lat_lon = (51.4680, -0.4551)

    # find outliers where the distance between the first point in the flight and the origin airport is greater than 100 km
    initial_distance_outliers = calculate_initial_distance(
        opensky_data, ADEP_lat_lon, distance_threshold=100
    )
    print(
        f"Found {len(initial_distance_outliers)} flights with excessive initial distances."
    )
    print(
        f"Number of unique flight ids in initial distance outliers that are in consecutive distance outliers: {len(set(initial_distance_outliers).intersection(set(consecutive_distance_outliers)))}"
    )

    # find outliers where the distance between the last point in the flight and the destination airport is greater than 100 km
    final_distance_outliers = calculate_final_distance(
        opensky_data, ADES_lat_lon, distance_threshold=100
    )
    print(
        f"Found {len(final_distance_outliers)} flights with excessive final distances."
    )
    print(
        f"Number of unique flight ids in final distance outliers that are in consecutive distance outliers: {len(set(final_distance_outliers).intersection(set(consecutive_distance_outliers)))}"
    )
    print(
        f"Number of unique flight ids in final distance outliers that are in initial distance outliers: {len(set(final_distance_outliers).intersection(set(initial_distance_outliers)))}"
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

    # drop flights with initial distance outliers that are not dropped by consecutive distance outliers
    initial_distance_outliers = [
        flight_id
        for flight_id in initial_distance_outliers
        if flight_id not in consecutive_distance_outliers
    ]
    print("Dropping flights with initial distance outliers...")
    opensky_data = opensky_data[
        ~opensky_data["flight_id"].isin(initial_distance_outliers)
    ]

    # drop flights with final distance outliers that are not dropped by consecutive distance outliers or initial distance outliers
    final_distance_outliers = [
        flight_id
        for flight_id in final_distance_outliers
        if flight_id not in consecutive_distance_outliers
        and flight_id not in initial_distance_outliers
    ]
    print("Dropping flights with final distance outliers...")
    opensky_data = opensky_data[
        ~opensky_data["flight_id"].isin(final_distance_outliers)
    ]

    # reset the index
    opensky_data = opensky_data.reset_index(drop=True)

    # find the average number of rows in each flight with unique flight_id
    avg_sequence_length = opensky_data.groupby("flight_id").size().mean()

    # count the number of rows in each flight with unique flight_id, and make it a dataframe
    size = opensky_data.groupby("flight_id").size().reset_index(name="counts")

    # calculate z-scores for the counts
    size["z_score"] = zscore(size["counts"])

    # drop flights with lowest sequence length
    low_counts_outliers = size[size["z_score"] < lowest_sequence_length_threshold]
    print(
        f"Found {len(low_counts_outliers)} outliers in column 'counts', with threshold {lowest_sequence_length_threshold}"
    )
    # print(low_counts_outliers)

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
    base_path: str, ADEP_code: str, ADES_code: str
) -> pd.DataFrame:

    # look for any .csv files in the base_path
    files = glob.glob(os.path.join(base_path, "*.csv"))
    print(f"Found {len(files)} files in the directory: {base_path}")

    # select only the files that contain the ADEP and ADES codes: e.g. opensky_EHAM_LIMC_2019-01-01_2023-01-01.csv

    files = [file for file in files if ADEP_code in file and ADES_code in file]
    print(
            f"Found {len(files)} files with ADEP and ADES codes in the directory: {base_path}"
        )
        # select the first file
    file = files[0]  # TODO: process all files

    print(f"Processing file: {file}")
    opensky_data = pd.read_csv(file)
    # drop Unnamed: 0 column
    opensky_data = opensky_data.drop(columns=["Unnamed: 0"])
    # opensky_data = opensky_data.drop(columns=['groundspeed', 'track', 'geoaltitude'])

    # drop the rows with Nan values and reset the index
    opensky_data = opensky_data.dropna().reset_index(drop=True)

    # drop rows with negative values in the 'altitude' column
    opensky_data = opensky_data[opensky_data["altitude"] >= 0]

    # change the column names to match the ectrl data: estdepartureairport	estarrivalairport, to ADEP and ADES
    opensky_data = opensky_data.rename(
        columns={"estdepartureairport": "ADEP", "estarrivalairport": "ADES"}
    )

    # Convert the 'timestamp' column to datetime
    opensky_data["timestamp"] = pd.to_datetime(opensky_data["timestamp"])
    # df.sort_values('timestamp', inplace=True)
    opensky_data.sort_values("timestamp", inplace=True)

    print("Loaded OpenSky data, now assigning flight ids...")
    # assign flight ids
    opensky_data = assign_flight_ids(opensky_data, window=6)

    # remove outliers
    opensky_data, avg_sequence_length = remove_outliers(
        opensky_data, thresholds=[50, 2.2, -1.4]
    )  # [consecutive_distance_threshold, altitude_threshold, lowest_sequence_length_threshold]
    print("Removed outliers, now getting trajectories...")


    # make sure that the sequence length is an even integer: since fourier transform return even sequence length !!
    avg_sequence_length = (
        int(avg_sequence_length)
        if int(avg_sequence_length) % 2 == 0
        else int(avg_sequence_length) - 1
    )
    print(f"Average sequence length of all flights: {avg_sequence_length}")

    return opensky_data, avg_sequence_length


def get_trajectories(flights_points: pd.DataFrame) -> Traffic:

    # Convert timestamp to datetime object
    flights_points["timestamp"] = pd.to_datetime(
        flights_points["timestamp"], format="%d-%m-%Y %H:%M:%S", utc=True
    )

    # Create Flight objects for each unique flight ID
    grouped_flights = flights_points.groupby("flight_id")
    flights_list = [Flight(group) for _, group in grouped_flights]

    # Create a Traffic object containing all the flighits
    trajectories = Traffic.from_flights(flights_list)
    return trajectories


def prepare_trajectories(
    trajectories: Traffic, n_samples: int, n_jobs: int, douglas_peucker_coeff: float
) -> Traffic:

    # trajectories = trajectories.compute_xy(projection=EuroPP())

    # Simplify trajectories with Douglas-Peucker algorithm if a coefficient is provided
    if douglas_peucker_coeff is not None:
        print("Simplification...")
        trajectories = trajectories.simplify(tolerance=1e3).eval(desc="")

    # Add elapsed time since start for each flight
    trajectories = Traffic.from_flights(
        flight.assign(
            timedelta=lambda r: (r.timestamp - flight.start).apply(
                lambda t: t.total_seconds()
            )
        )
        for flight in trajectories
    )
 
    # clustering
    print("Clustering...")
    np.random.seed(
        199
    )  # random seed for reproducibility (has big impact on the clustering shape)
    # nb_samples = n_samples if n_samples < len(trajectories) else len(trajectories)
    # print(f"Number of samples: {nb_samples}, Number of trajectories: {len(trajectories)}, n_samples for resampling: {n_samples}")
    nb_samples = n_samples if n_samples < 1000 else 1000
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
    parser.add_argument("--ADEP", dest="ADEP", type=str, default="EHAM")
    parser.add_argument("--ADES", dest="ADES", type=str, default="LIMC")
    parser.add_argument(
        "--raw_data_dir", dest="base_path", type=str, default="../raw_data/"
    )
    # source of data: Either Eurocontrol or OpenSky
    parser.add_argument(
        "--data_source", dest="data_source", type=str, default="EuroControl"
    )
    # save directory for the prepared trajectories
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="./data/real/")

    args = parser.parse_args()
    return args



def main():
    args = get_args()

    # if args.data_source == "EuroControl":
    # if args.data_source caontains EuroControl, then load EuroControl data (ignore capitalization)
    if "eurocontrol" in args.data_source.lower():
        flights_points, avg_sequence_length = load_EuroControl_flights_points(
            os.path.join(args.base_path, "EuroControl"),
            args.ADEP, args.ADES
        )
    # elif args.data_source == "OpenSky":
    elif "opensky" in args.data_source.lower():
        flights_points, avg_sequence_length = load_OpenSky_flights_points(
            os.path.join(args.base_path, "OpenSky"),
            args.ADEP, args.ADES
        )
    else:
        raise ValueError(
            f"Invalid data source: {args.data_source}. For now, only EuroControl and OpenSky are supported."
        )

    # Create Traffic object from flight points
    trajectories = get_trajectories(flights_points)
    del flights_points

    # Prepare trajectories for training
    trajectories = prepare_trajectories(
        trajectories, int(avg_sequence_length), n_jobs=7, douglas_peucker_coeff=None
    )

    # Save the prepared trajectories to a pickle file
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir, f"{args.data_source}_{args.ADEP}_{args.ADES}.pkl"
    )
    trajectories.to_pickle(Path(save_path))
    print(f"Saved trajectories to {save_path}")


if __name__ == "__main__":
    main()
