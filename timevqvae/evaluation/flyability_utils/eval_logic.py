import argparse
import glob
import importlib.util
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
from geopy.distance import geodesic
from tqdm import tqdm
from traffic.core import Traffic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Some useful methods for evaluation
def setup_simulator():
    """Install the bluesky simulator using pypi"""
    # Install BlueSky
    # Do with process: !pip install bluesky
    # !pip install bluesky-simulator
    package_name = "bluesky"

    if importlib.util.find_spec(package_name) is None:
        print("Installing BlueSky ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "bluesky-simulator"]
        )

    output_folder = os.path.expanduser("~/bluesky/output")

    return output_folder


# Headings
def get_heading(waypoint1, waypoint2):
    """Gives the heading to follow from waypoint 1 to waypoing2 (2D)

    Parameters
    ----------
    waypoint1 (dict):
        First waypoint with latitude and longitude
    waypoint2 (dict):
        Second waypoint with latitude and longitude

    Returns
    -------pp
        float: Heading to follow in degrees
    """
    # Assuming you have two waypoints as (lat1, lon1) and (lat2, lon2)
    lat1, lon1 = math.radians(waypoint1["latitude"]), math.radians(
        waypoint1["longitude"]
    )
    lat2, lon2 = math.radians(waypoint2["latitude"]), math.radians(
        waypoint2["longitude"]
    )

    # Calculate difference in coordinates
    dLon = lon2 - lon1

    # Calculate heading
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        dLon
    )
    heading = math.atan2(y, x)

    # Convert heading from radians to degrees
    heading = math.degrees(heading)

    # Ensure heading is in range 0-360
    heading = (heading + 360) % 360

    return heading


def get_initial_heading(flight):
    """Gives the initial heading to follow for a flight to the first waypoint

    Parameters
    ----------
    flight (Flight):
        Flight object with waypoints

    Returns
    -------
        float: Heading to follow in degrees
    """
    # Get first two waypoints
    waypoint1 = flight.data.iloc[0]
    waypoint2 = flight.data.iloc[1]

    # Calculate heading
    heading = get_heading(waypoint1, waypoint2)

    return heading


def get_speed(waypoint1, waypoint2):
    """Gives the speed between two waypoints

    Parameters
    ----------
    waypoint1 (dict):
        First waypoint with latitude, longitude and timestamp
    waypoint2 (dict):
        Second waypoint with latitude, longitude and timestamp

    Returns
    -------
        float: Speed in knots
    """
    # Calculate distance in nautical miles
    coords_1 = (waypoint1["latitude"], waypoint1["longitude"])
    coords_2 = (waypoint2["latitude"], waypoint2["longitude"])
    distance = geodesic(coords_1, coords_2).nm

    # Calculate time difference in hours
    time_diff = (waypoint2["timestamp"] - waypoint1["timestamp"]).total_seconds() / 3600

    # Calculate speed in knots, handling division by zero
    # speed = distance / time_diff
    speed = distance / (time_diff + 1e-6)

    return speed


def get_initial_speed(flight):
    """Gives the initial speed to follow for a flight to the first waypoint

    Parameters
    ----------
    flight (Flight):
        Flight object with waypoints

    Returns
    -------
        float: Speed in knots
    """
    # Get first two waypoints
    waypoint1 = flight.data.iloc[0]
    waypoint2 = flight.data.iloc[1]

    # Calculate distance in nautical miles
    coords_1 = (waypoint1["latitude"], waypoint1["longitude"])
    coords_2 = (waypoint2["latitude"], waypoint2["longitude"])
    distance = geodesic(coords_1, coords_2).nm

    # Calculate time difference in hours
    time_diff = (waypoint2["timestamp"] - waypoint1["timestamp"]).total_seconds() / 3600

    # Calculate speed in knots, handling division by zero
    # speed = distance / time_diff
    speed = distance / (time_diff + 1e-6)

    return speed


# TODO: Retrieve groundspeed from model or update this to a better method
def add_ground_speed(flight):
    """Adds groundspeed to a flight object

    Parameters
    ----------
    flight (Flight):
        Flight object with waypoints

    Returns
    -------
        Flight: Flight object with groundspeed added
    """
    # Get initial speed and heading
    groundspeed = []
    groundspeed.append(get_initial_speed(flight))
    for i in range(1, len(flight.data)):
        # Get previous and current waypoints
        waypoint1 = flight.data.iloc[i - 1]
        waypoint2 = flight.data.iloc[i]

        # Calculate speed
        speed = get_speed(waypoint1, waypoint2)

        # Add ground speed to list
        groundspeed.append(speed)

    # Add speeds to dataframe
    flight = flight
    flight.data.loc[:, "groundspeed"] = groundspeed

    return flight


# Removing similar points
def remove_neighbours(flight):
    """Removes the next waypoint if coordinates are equal to the current waypoint"""
    flight.data = flight.data[
        flight.data["latitude"] != flight.data["latitude"].shift()
    ]
    flight.data = flight.data[
        flight.data["longitude"] != flight.data["longitude"].shift()
    ]

    return flight


def build_scenario(flight):
    """Builds a BlueSky simulator scenario file for a flight object and outputs it in output_`fid`.scn

    Parameters
    ----------
    flight (Flight):
        Flight object with waypoints
    """

    fid = flight.data.iloc[0]["flight_id"]
    fname = os.path.join(BASE_DIR, f"scenarios/output_{fid}.scn")
    if not os.path.exists(os.path.join(BASE_DIR, "scenarios")):
        os.makedirs(os.path.join(BASE_DIR, "scenarios"))
    init_point = flight.data.iloc[0]
    with open(fname, "w") as f:
        f.write(
            f"00:00:00.00>CRE {fid} {flight.data.iloc[0]['AC Type']} {init_point['latitude']} {init_point['longitude']} {get_initial_heading(flight)} {init_point['altitude']} {get_initial_speed(flight)}\n"
        )
        # remove AC Type
        # f.write(f"00:00:00.00>CRE {fid} {init_point['latitude']} {init_point['longitude']} {get_initial_heading(flight)} {init_point['altitude']} {get_initial_speed(flight)}\n")

        for index, row in flight.data.iloc[1:].iterrows():
            line = f"00:00:00.00>DEFWPT WPTZ{index},{row['latitude']}, {row['longitude']}\n"
            f.write(line)
            line = f"00:00:00.00>{fid} ADDWPT WPTZ{index} {row['altitude']} {row['groundspeed']}\n"
            f.write(line)
        f.write(f"00:00:00.00>{fid} LNAV ON\n")
        f.write(f"00:00:00.00>{fid} VNAV ON\n")
        f.write(f"00:00:00.00>{fid} AT WPTZ{index} QUIT\n")


def assemble_scenarios(flight_ids, simulation_time, debug=False):
    """Assembles a BlueSky simulator scenario file given a list of flight ids and outputs it in evaluation_scenario.scn"""
    with open(os.path.join(BASE_DIR, "evaluation_scenario.scn"), "w") as f:
        for fid in flight_ids:
            line = f"00:00:00.00>PCALL {os.path.join(BASE_DIR, 'scenarios', f'output_{fid}.scn')}\n"
            f.write(line)
        f.write(
            f"00:00:00.00>PCALL {os.path.join(BASE_DIR, 'evaluation_logger.scn')}\n"
        )
        f.write("00:00:03.00>FF\n")  # Fast forward with delay to avoid bugs
        if debug:
            f.write("00:01:00.00>OP\n")  # Resume normal speed to avoid clsosing bug
            f.write("00:01:00.10>CLOSE\n")  # Close the simulation
        else:
            f.write(
                f"{simulation_time}.00>OP\n"
            )  # Resume normal speed to avoid closing bug
            f.write(f"{simulation_time}.10>CLOSE\n")  # Close the simulation


def build_logger(dt=10, variables=["traf.lat", "traf.lon", "traf.alt", "traf.id"]):
    """Builds a BlueSky simulator logger file for a flight object and outputs it in evaluation_logger.scn"""
    with open(os.path.join(BASE_DIR, "evaluation_logger.scn"), "w") as f:
        # creating logger
        f.write(f"0:00:00.00>CRELOG EVALLOG {dt}\n")
        # adding logging items
        values = ""
        for var in variables:
            values += var + " "
        f.write(f"0:00:00.00>EVALLOG ADD {values}\n")
        # starting logger
        f.write("0:00:00.00>EVALLOG ON\n")


###################
# Extracting Logs #
###################


def logs_to_df(fname: os.PathLike) -> pd.DataFrame:
    # Read the log file into a DataFrame
    df = pd.read_csv(
        fname,
        comment="#",
        names=["relt", "latitude", "longitude", "altitude", "flight_id"],
    )

    df = df.sort_values(by=["flight_id", "relt"])

    return df


def annotate_logs(logs, traffic: Traffic):
    """Annotates the logs with flight information"""
    tdf = (
        traffic.data.sort_values(by=["flight_id", "timestamp"])
        .drop_duplicates(subset=["flight_id"], keep="first")
        .drop(columns=["altitude", "longitude", "latitude"])
    )  # Drop columns that are not needed

    merged_df = tdf.merge(logs, on="flight_id")
    merged_df["timestamp"] = merged_df["timestamp"] + pd.to_timedelta(
        merged_df["relt"], unit="s"
    )

    return Traffic(merged_df)


def get_last_fname(directory: str):
    files = glob.glob(os.path.join(directory, "*"))
    # files = [f for f in glob.glob(os.path.join(directory, "*")) if os.path.isfile(f)]

    if files:  # If there are files in the directory
        latest_file = max(files, key=os.path.getctime)
        return latest_file
    else:
        raise FileNotFoundError(
            "No files in the directory provided ; Directory given was : ", directory
        )


def simulate(traffic: Traffic, config: dict, debug=False):
    delta = config["delta"]
    batch_size = config["batch_size"]
    # trajectories_file = config['trajectories_file']
    # early_batch_stop = config.get("early_batch_stop", False)
    # logs_directory = os.path.expanduser(os.path.join("~", config["logs_directory"]))
    # make logs directory
    # logs_directory = os.path.join(BASE_DIR, "logs")
    # os.makedirs(logs_directory, exist_ok=True)
    simulation_time = config["simulation_time"]
    logs_directory = config["logs_directory"]
    # os.makedirs(logs_directory, exist_ok=True)

    print("### Flights pre-processing ###")

    # df = pd.read_csv(trajectories_file)
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    # traffic = Traffic(df)
    # traffic = Traffic.from_file(trajectories_file)

    flights = []
    for flight in traffic:
        flight.data = flight.data[
            ~((flight.data["altitude"] == 0) & (flight.data["altitude"].shift(-1) == 0))
        ]

        flight = remove_neighbours(flight)
        flight = add_ground_speed(flight)

        flights.append(flight)
    traffic = Traffic.from_flights(flights)

    print("### Building scenarios ###")
    # Build logger
    build_logger()
    # Build scenarios
    flight_ids = []
    reconstructions = []
    for i in tqdm(range(0, len(traffic), batch_size), desc="Batch evaluation..."):
        batch = traffic[i : i + batch_size]
        for flight in batch:
            s_flight = flight.simplify(delta)
            build_scenario(s_flight)
            flight_ids.append(s_flight.data.iloc[0]["flight_id"])

        assemble_scenarios(np.unique(batch.data["flight_id"]), simulation_time, debug)

        cmd = f"bluesky --headless --scenfile {os.path.join(BASE_DIR, 'evaluation_scenario.scn')}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)
        process.wait()
        logs_fname = get_last_fname(logs_directory)
        # print(f'### Processing logs from {logs_fname} ###')
        logs = logs_to_df(logs_fname)
        recon = annotate_logs(logs, traffic)
        reconstructions.append(recon.data)
        # Deleting log file:
        os.remove(logs_fname)

    # df = pd.concat(reconstructions, axis=0).drop(columns=['index'])
    df = pd.concat(reconstructions, axis=0)  # .drop(columns=['index'])
    df.index.rename("index", inplace=True)
    recons = Traffic(df)
    return recons


def parse_config():
    parser = argparse.ArgumentParser(
        description="Runs the evaluation of trajectories generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=f"{os.path.join(BASE_DIR, 'evaluation_config.yaml')}",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return config


def clean():
    """Cleans the evaluation directory from generated scenario and log files"""
    try:
        os.remove(os.path.join(BASE_DIR, "evaluation_scenario.scn"))
    except:
        pass
    try:
        os.remove(os.path.join(BASE_DIR, "evaluation_logger.scn"))
    except:
        pass
    try:
        for f in glob.glob(os.path.join(BASE_DIR, "scenarios", "*")):
            os.remove(f)
        os.rmdir(os.path.join(BASE_DIR, "scenarios"))
    except Exception as e:
        print(e)
